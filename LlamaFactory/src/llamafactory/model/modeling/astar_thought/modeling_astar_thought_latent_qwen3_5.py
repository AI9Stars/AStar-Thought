from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5PreTrainedModel,
    Qwen3_5Model,
    Qwen3_5CausalLMOutputWithPast,
)

class Qwen3_5ForConditionalGeneration(Qwen3_5PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3_5Config

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # A*-Thought
        self.latent_begin_id = config.latent_begin_id
        self.latent_end_id = config.latent_end_id
        self.latent_token_base_id = config.latent_token_base_id
        self.structure_weight = getattr(config, 'structure_weight', 1.0)
        self.num_latent_clusters = getattr(config, 'num_latent_clusters', 64)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def _interior_token_counts_between_latent_bounds(self, raw_input_ids: torch.Tensor) -> list[int]:
        """按顺序扫描序列，对每个 latent_begin..latent_end 区间，返回二者之间的 token 数（不含 begin/end）。"""
        ids = raw_input_ids.detach().cpu().tolist()
        counts: list[int] = []
        i, n = 0, len(ids)
        while i < n:
            if ids[i] == self.latent_begin_id:
                j = i + 1
                while j < n and ids[j] != self.latent_end_id:
                    j += 1
                counts.append(j - (i + 1))
                i = j + 1 if j < n else n
            else:
                i += 1
        return counts

    def _collect_per_span_n_latent(self, latent_input_ids, latent_spans) -> list[int]:
        """与 all_latent_spans_embeds 展开顺序一致：逐 sample、逐 span 的 n_latent。"""
        all_n: list[int] = []
        for raw_input_ids, raw_latent_spans in zip(latent_input_ids, latent_spans):
            if not raw_latent_spans:
                continue
            counts = self._interior_token_counts_between_latent_bounds(raw_input_ids)
            for s_idx, span in enumerate(raw_latent_spans):
                ni = counts[s_idx] if s_idx < len(counts) else len(span)
                if ni <= 0:
                    ni = len(span) if len(span) > 0 else 1
                all_n.append(int(ni))
        return all_n
    
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        # latent
        latent_input_ids: Optional[torch.LongTensor] = None,
        latent_spans: Optional[list] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3_5CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        >>> model = Qwen3_5ForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        >>> output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(output_text)
        ```
        """

        # A*-Thought-Latent
        if input_ids is None and inputs_embeds is None:
            # Step 1: Collect all samples' information for batch processing
            batch_size = len(latent_input_ids)
            all_raw_input_embeds = []
            all_latent_spans_embeds = []  # List of lists: [sample_idx][span_idx]
            all_latent_masks = []  # List of lists: [sample_idx][latent_token_positions]
            sample_idx_to_spans = {}  # Map: sample_idx -> list of span indices in all_latent_spans_embeds
            
            # Process all samples to collect embeddings and masks
            current_span_idx = 0
            for sample_idx, (raw_input_ids, raw_latent_spans) in enumerate(zip(latent_input_ids, latent_spans)):
                if raw_latent_spans:
                    spans_lens = [len(span) for span in raw_latent_spans]
                    spans_sets = torch.tensor(
                        [item for span in raw_latent_spans for item in span],
                        dtype=torch.long,
                        device=raw_input_ids.device,
                    )
                    combined_ids = torch.cat([raw_input_ids, spans_sets], dim=0)
                    embeds = self.model.get_input_embeddings()(combined_ids)
                    raw_input_embeds = embeds[:raw_input_ids.size(0), :]
                    latent_spans_embeds = [embeds[raw_input_ids.size(0) + sum(spans_lens[:i]): raw_input_ids.size(0) + sum(spans_lens[:i+1]), :] for i in range(len(spans_lens))]
                    latent_mask = ((raw_input_ids >= self.latent_token_base_id) & (raw_input_ids < self.latent_token_base_id + self.num_latent_clusters)).cpu().nonzero(as_tuple=True)[0].tolist()
                    all_raw_input_embeds.append(raw_input_embeds)
                    all_latent_spans_embeds.extend(latent_spans_embeds)
                    all_latent_masks.append(latent_mask)
                    sample_idx_to_spans[sample_idx] = list(range(current_span_idx, current_span_idx + len(latent_spans_embeds)))
                    current_span_idx += len(latent_spans_embeds)
                else:
                    raw_input_embeds = self.model.get_input_embeddings()(raw_input_ids)
                    all_raw_input_embeds.append(raw_input_embeds)
                    all_latent_masks.append([])
                    sample_idx_to_spans[sample_idx] = []
            
            # Step 2: Batch process all latent spans if any exist
            if all_latent_spans_embeds:
                all_span_n_latent = self._collect_per_span_n_latent(latent_input_ids, latent_spans)

                def _span_to_n_embeds(span_embeds, n_latent):
                    L = span_embeds.size(0)
                    if L >= n_latent:
                        seg_ends = [L * (i + 1) // n_latent for i in range(n_latent)]
                        seg_starts = [0] + seg_ends[:-1]
                        segment_embeds = [span_embeds[s:e].mean(dim=0) for s, e in zip(seg_starts, seg_ends)]
                    else:
                        segment_embeds = [span_embeds[i] if i < L else span_embeds.mean(dim=0) for i in range(n_latent)]
                    return torch.stack(segment_embeds, dim=0)  # (n_latent, hidden_size)

                pooled = [
                    _span_to_n_embeds(se, n)
                    for se, n in zip(all_latent_spans_embeds, all_span_n_latent)
                ]
                max_n = max(all_span_n_latent)
                hidden_size = pooled[0].size(-1)
                batch_avg_embeds = pooled[0].new_zeros(len(pooled), max_n, hidden_size)
                for i, (emb, n) in enumerate(zip(pooled, all_span_n_latent)):
                    batch_avg_embeds[i, :n, :] = emb

                # Step 3: Scatter 仅写回各 span 有效长度 n，跳过 padding（直接用 pooled 嵌入，不过 backbone）
                for sample_idx in range(batch_size):
                    if all_latent_masks[sample_idx]:
                        span_indices = sample_idx_to_spans[sample_idx]
                        latent_positions = all_latent_masks[sample_idx]
                        pos_cursor = 0
                        for global_span_idx in span_indices:
                            n = all_span_n_latent[global_span_idx]
                            positions = latent_positions[pos_cursor : pos_cursor + n]
                            if len(positions) != n:
                                raise ValueError(
                                    f"latent scatter mismatch: n={n}, len(positions)={len(positions)}. "
                                    f"Likely cutoff_len truncation or invalid latent tokens. Please increase cutoff_len."
                                )
                            pos_cursor += n
                            sample_hidden_states = batch_avg_embeds[global_span_idx, :n, :]
                            target_embeds = all_raw_input_embeds[sample_idx]
                            sample_hidden_states = sample_hidden_states.to(
                                device=target_embeds.device, dtype=target_embeds.dtype
                            )
                            target_embeds[positions, :] = sample_hidden_states
            # Step 4: Stack all samples' embeddings
            inputs_embeds = torch.stack(all_raw_input_embeds, dim=0)

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)

            # 1. 找出哪些是 Latent Token，哪些是普通 Token
            latent_target_mask = (shift_labels >= self.latent_token_base_id) & (shift_labels < self.latent_token_base_id + self.num_latent_clusters)
            normal_target_mask = ~latent_target_mask & (shift_labels != -100) # 排除 padding (-100)

            # 2. 计算普通 Token 的标准交叉熵 Loss
            weights = torch.ones(
                self.config.text_config.vocab_size, 
                device=shift_logits.device, 
                dtype=shift_logits.dtype
            )
            weights[self.latent_begin_id] = self.structure_weight
            weights[self.latent_end_id] = self.structure_weight
            
            normal_loss_fct = CrossEntropyLoss(weight=weights, reduction='sum')
            normal_loss = normal_loss_fct(shift_logits[normal_target_mask], shift_labels[normal_target_mask])

            # 3. 计算 Latent Token 的软标签 Loss（label mean pooling，n_latent 与 v2.12 嵌入路径一致）
            latent_loss = 0.0
            if latent_target_mask.any() and latent_input_ids is not None and latent_spans is not None:
                current_latent_logits = shift_logits[latent_target_mask]

                vocab_size = self.config.text_config.vocab_size
                all_spans = [span for spans in latent_spans if spans for span in spans]
                all_span_n_latent = self._collect_per_span_n_latent(latent_input_ids, latent_spans)
                assert len(all_spans) > 0, "Expected all_spans to be non-empty based on data pipeline guarantees!"
                assert len(all_span_n_latent) == len(all_spans), "n_latent list must align with flattened spans!"

                def _span_to_n_soft_labels(span_tokens_list, n_latent):
                    span_tokens = torch.tensor(span_tokens_list, dtype=torch.long, device=shift_logits.device)
                    L = span_tokens.size(0)
                    one_hot = F.one_hot(span_tokens.clamp(0, vocab_size - 1), num_classes=vocab_size).float()
                    if L >= n_latent:
                        seg_ends = [L * (i + 1) // n_latent for i in range(n_latent)]
                        seg_starts = [0] + seg_ends[:-1]
                        soft_labels = [one_hot[s:e].mean(dim=0) for s, e in zip(seg_starts, seg_ends)]
                    else:
                        soft_labels = [one_hot[i] if i < L else one_hot.mean(dim=0) for i in range(n_latent)]
                    return torch.stack(soft_labels, dim=0)  # (n_latent, vocab_size)

                span_soft_labels = []
                for span, n_latent in zip(all_spans, all_span_n_latent):
                    span_soft_labels.append(_span_to_n_soft_labels(span, n_latent))
                target_latent_labels = torch.cat(span_soft_labels, dim=0)  # (total_num_spans * n_latent, vocab_size)
                
                min_len = min(current_latent_logits.size(0), target_latent_labels.size(0))
                soft_loss_fct = CrossEntropyLoss(reduction='none')
                raw_latent_loss = soft_loss_fct(
                    current_latent_logits[:min_len],
                    target_latent_labels[:min_len]
                )
                latent_loss = (raw_latent_loss * self.structure_weight).sum()

            # 4. 合并并求平均
            total_valid_tokens = normal_target_mask.sum() + latent_target_mask.sum()
            loss = (normal_loss + latent_loss) / total_valid_tokens.clamp(min=1)


        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )