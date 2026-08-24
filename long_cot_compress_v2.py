import argparse
import tiktoken
import re
from tqdm import tqdm
import json
import torch
import multiprocessing
import os
import copy
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(
    model_path: str,
    *,
    device_map: str,
    torch_dtype: torch.dtype,
    trust_remote_code: bool = True,
):
    """使用 transformers 标准方式加载因果 LM 与 tokenizer（与 PromptCompressor 解耦）。"""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3_5":
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        pad = getattr(config, "pad_token_id", None)
        tokenizer.pad_token_id = pad if pad is not None else tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def parse_angle_threshold_deg(value):
    """夹角阈值（度），要求在 [0, 180]，与问答一致：偏大则更多是「偏离 q→solution 主轴」的步骤被压缩。"""
    x = float(value)
    if not (0.0 <= x <= 180.0):
        raise argparse.ArgumentTypeError(
            "pca_angle_threshold 必须在区间 [0, 180] 内（例如 120）"
        )
    return x


def flags_from_trajectory_cache_row(item, *, angle_threshold_deg, num_steps):
    if num_steps == 0:
        return []
    if item.get("pca_trajectory_error"):
        return [1] * num_steps

    angles = item.get("pca_step_angles_deg")
    if angles is not None and len(angles) == num_steps:
        return [0 if float(a) > angle_threshold_deg else 1 for a in angles]

    return [1] * num_steps


def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def save_to_jsonl(data, file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def _pca3_numpy(x: np.ndarray) -> np.ndarray:
    """x: (n, d) -> (n, 3)，中心化后取前三大主成分（与 analysis_representation._pca3_numpy 一致）。"""
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comp = min(3, vt.shape[0])
    res = x @ vt[:comp].T
    if comp < 3:
        res = np.pad(res, ((0, 0), (0, 3 - comp)), mode="constant")
    return res


def _angle_degrees(u: np.ndarray, v: np.ndarray) -> float:
    """两向量夹角，范围 [0, 180] 度。"""
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _extract_user_content_span(full_text: str, question: str) -> tuple[int, int]:
    patterns = [
        r"<\|im_start\|>user\s*\n([\s\S]*?)\s*<\|im_end\|>",
        r"<\|im_start\|>user\s*\n([\s\S]*?)\s*<\|redacted_im_end\|>",
        r"<\|start_header_id\|>user<\|end_header_id\|>\s*\n([\s\S]*?)\s*(?:<\|eot_id\|>|<\|im_end\|>)",
    ]
    for pat in patterns:
        m = re.search(pat, full_text)
        if m:
            return m.start(1), m.end(1)
    i = full_text.find(question)
    if i >= 0:
        return i, i + len(question)
    raise ValueError("无法在对话文本中定位 user / question 区间")


def _char_span_to_token_indices(offset_mapping, seq_len: int, c0: int, c1: int) -> list[int]:
    out: list[int] = []
    for ti, span in enumerate(offset_mapping):
        if ti >= seq_len:
            break
        if not span or span[0] is None:
            continue
        a, b = int(span[0]), int(span[1])
        if b > c0 and a < c1:
            out.append(ti)
    return out


def _mean_hidden_for_tokens(
    hidden: torch.Tensor, token_indices: list[int]
) -> np.ndarray:
    if not token_indices:
        return np.zeros(hidden.shape[-1], dtype=np.float64)
    ix = torch.tensor(token_indices, device=hidden.device, dtype=torch.long)
    return hidden[ix].float().detach().mean(dim=0).cpu().numpy()


def _last_hidden_sequence(model_out) -> torch.Tensor:
    """
    取 batch 第 0 条序列的最后一层 hidden，形状 (seq_len, dim)。
    Hugging Face 的 CausalLMOutputWithPast 一般没有 last_hidden_state，而是 output_hidden_states=True 时的 hidden_states[-1]。
    """
    lhs = getattr(model_out, "last_hidden_state", None)
    if lhs is not None:
        return lhs[0]
    hs = getattr(model_out, "hidden_states", None)
    if hs is not None and len(hs) > 0:
        return hs[-1][0]
    raise RuntimeError(
        "模型输出不含 last_hidden_state 或 hidden_states；请将 forward(..., output_hidden_states=True)"
    )


def compute_pca_angle_step_flags(
    model: torch.nn.Module,
    tokenizer,
    *,
    device: torch.device,
    system: str,
    question: str,
    thought_steps: list[str],
    solution: str,
    thought_begin_tag: str,
    thought_end_tag: str,
    angle_threshold_deg: float,
) -> tuple[list[int], list[float], list[float] | None, list[list[float]]]:
    """
    对 question -> steps -> solution 各段 last hidden 做 mean pool，再 3D PCA；
    z_0 = PCA 空间中 (P_solution - P_question)；
    z_i（与第 i 个 step 对齐）为：i=1 时 P_S1−P_Q；i>1 时 P_S_i−P_S_{i−1}；最后一步（i=N）为 P_S_N−P_S_{N−1}，不含「S_N→solution」段落。
    若 angle(z_i, z_0) > angle_threshold_deg 则该 step 压缩 latent（flag=0），否则 flag=1。
    返回 (flags, angles, z0_pc3, z_step_segments_pc3)；z0 与每步 z 均为 3 维 PCA 坐标下的位移向量。
    """
    num_steps = len(thought_steps)
    if num_steps == 0:
        return [], [], None, []

    thought_core = "\n\n".join(thought_steps)
    assistant_body = (
        f"{thought_begin_tag}\n\n{thought_core}\n\n{thought_end_tag}\n\n{solution}"
    )
    messages = []
    if system and str(system).strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": assistant_body})

    tok = tokenizer
    if getattr(tok, "chat_template", None):
        full_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    else:
        full_text = f"{question}\n{assistant_body}"

    enc = tok(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    om0 = enc["offset_mapping"][0]
    if hasattr(om0, "tolist"):
        offset_mapping = om0.tolist()
    else:
        offset_mapping = list(om0)
    seq_len = int(input_ids.shape[1])

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
        )
    h = _last_hidden_sequence(out)

    u0, u1 = _extract_user_content_span(full_text, question)
    q_tokens = _char_span_to_token_indices(offset_mapping, seq_len, u0, u1)
    if not q_tokens:
        probe = min(64, seq_len)
        q_tokens = list(range(probe))
    h_q = _mean_hidden_for_tokens(h, q_tokens)

    tb = full_text.find(thought_begin_tag)
    if tb < 0:
        raise ValueError("assistant 中未找到 thought_begin_tag")
    scan = tb + len(thought_begin_tag)
    while scan < len(full_text) and full_text[scan] in "\n\r \t":
        scan += 1

    step_spans: list[tuple[int, int]] = []
    for si, step in enumerate(thought_steps):
        idx = full_text.find(step, scan)
        if idx < 0:
            raise ValueError(f"无法在 full_text 中定位第 {si} 步内容")
        step_spans.append((idx, idx + len(step)))
        scan = idx + len(step)

    te = full_text.find(thought_end_tag, scan)
    if te < 0:
        raise ValueError("assistant 中未找到 thought_end_tag")

    sol_start = te + len(thought_end_tag)
    while sol_start < len(full_text) and full_text[sol_start] in "\n\r \t":
        sol_start += 1
    sol_char = full_text.find(solution, sol_start)
    if sol_char < 0:
        sol_char = sol_start

    sol_tokens = _char_span_to_token_indices(
        offset_mapping, seq_len, sol_char, sol_char + len(solution)
    )
    if not sol_tokens:
        sol_tokens = list(range(max(0, seq_len - min(128, seq_len)), seq_len))
    h_sol = _mean_hidden_for_tokens(h, sol_tokens)

    h_steps: list[np.ndarray] = []
    for (a, b) in step_spans:
        tix = _char_span_to_token_indices(offset_mapping, seq_len, a, b)
        if not tix:
            mid = (a + b) // 2
            nearest = []
            best = None
            for tj, span in enumerate(offset_mapping):
                if tj >= seq_len or not span or span[0] is None:
                    continue
                sa, sb = int(span[0]), int(span[1])
                d = max(0, mid - sb, sa - mid)
                if best is None or d < best:
                    best = d
                    nearest = [tj]
                elif best is not None and d == best:
                    nearest.append(tj)
            tix = nearest[:8] if nearest else [min(seq_len - 1, tb + 8)]
        h_steps.append(_mean_hidden_for_tokens(h, tix))

    anchors = np.stack([h_q] + h_steps + [h_sol], axis=0)
    p = _pca3_numpy(anchors)
    z0 = p[-1] - p[0]
    z0_pc3 = [float(x) for x in z0.tolist()]

    angles: list[float] = []
    flags: list[int] = []
    z_step_segments_pc3: list[list[float]] = []
    # p[0]=Q，p[1..N]=S_1..S_N，p[N+1]=SOL；第 si 步（si=0..N-1）：z=P_{S_{si+1}}−P_{S_si}
    for si in range(num_steps):
        zi = p[si + 1] - p[si]
        z_step_segments_pc3.append([float(x) for x in zi.tolist()])
        ang = _angle_degrees(zi, z0)
        angles.append(ang)
        flags.append(0 if ang > angle_threshold_deg else 1)

    return flags, angles, z0_pc3, z_step_segments_pc3


def compressed_thought_from_flags(thought_steps: list[str], flags: list[int]) -> str:
    return "\n\n".join(
        step for step, flag in zip(thought_steps, flags) if flag == 1
    )

def load_data(data_path): # deepseek-v3.2-speciale-openr1-math-3k
    original_data = load_jsonl_file(data_path)
    thought_begin_tag = args.thought_begin_tag
    thought_end_tag = args.thought_end_tag
    data = []
    for i, row_data in enumerate(original_data):
        system = row_data["messages"][0]["content"]
        user = row_data["messages"][1]["content"]
        assistant = row_data["messages"][2]["content"]
        thought_begin_index = assistant.find(thought_begin_tag)
        thought_end_index = assistant.find(thought_end_tag)
        if thought_begin_index == -1 or thought_end_index == -1:
            raise ValueError(f"Row {i}: missing thought tags")
        assistant_thought = assistant[
            thought_begin_index + len(thought_begin_tag): thought_end_index
        ].strip()
        assistant_solution = assistant[
            thought_end_index + len(thought_end_tag):
        ].strip()
        data.append({"system": system, "conversations": [{"from": "user", "value": user}, {"from": "assistant", "value": assistant}], "assistant_thought": assistant_thought, "assistant_solution": assistant_solution})
    return data


def work(data, num_gpus, process_id):
    try:
        print(f"Process {process_id + 1} Started")
        if "cpu" not in args.device_map:
            local_map = f"cuda:{process_id % num_gpus}"
            torch_dtype = torch.bfloat16
        else:
            local_map = "cpu"
            torch_dtype = torch.float32

        model, tokenizer = load_causal_lm(
            args.model_path,
            device_map=local_map,
            torch_dtype=torch_dtype,
            trust_remote_code=not args.no_trust_remote_code,
        )
        device = next(model.parameters()).device

        output_data = []
        for i, row_data in tqdm(enumerate(data), total=len(data), desc=f"Process {process_id + 1}"):
            torch.cuda.empty_cache()
            thought_steps = [
                item.strip() for item in row_data["assistant_thought"].split("\n\n") if item.strip()
            ]
            question = row_data["conversations"][0]["value"]
            solution = row_data["assistant_solution"]
            err = None
            z0_pc3 = None
            z_step_segments_pc3: list[list[float]] = []
            if not thought_steps:
                angles: list[float] = []
            else:
                try:
                    _, angles, z0_pc3, z_step_segments_pc3 = compute_pca_angle_step_flags(
                        model,
                        tokenizer,
                        device=device,
                        system=row_data.get("system", "") or "",
                        question=question,
                        thought_steps=thought_steps,
                        solution=solution,
                        thought_begin_tag=args.thought_begin_tag,
                        thought_end_tag=args.thought_end_tag,
                        angle_threshold_deg=args.pca_angle_threshold,
                    )
                except Exception as exc:
                    err = repr(exc)
                    angles = []
                    z0_pc3 = None
                    z_step_segments_pc3 = []
            orig_conv = {
                "system": row_data.get("system", "") or "",
                "conversations": copy.deepcopy(row_data["conversations"]),
            }
            if row_data.get("source_messages") is not None:
                orig_conv["source_messages"] = copy.deepcopy(row_data["source_messages"])

            output_data.append(
                {
                    "original_conversation": orig_conv,
                    "question": question,
                    "thought_steps": thought_steps,
                    "solution": solution,
                    "z0_pc3": z0_pc3,
                    "z_step_segments_pc3": z_step_segments_pc3,
                    "pca_step_angles_deg": angles,
                    "pca_trajectory_error": err,
                }
            )
            
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Process {process_id + 1} Complete")
        return output_data

    except Exception as e:
        import sys
        import traceback
        # 1. 把子进程 PID、文件、行号全打出来
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        # 2. 立即 flush 到 stderr，主进程实时可见
        print(f'[worker-{process_id + 1}][pid-{os.getpid()}] CRASH\n{"".join(tb_lines)}',
              file=sys.stderr, flush=True)
        # 3. 把异常原样抛上去，pool.starmap 会立刻终止并 raise
        raise

def main():
    print("========== Data Loading ==========")
    data = load_data(args.data_path)
    total_data = args.max_len if args.max_len != -1 else len(data)
    print(f"Data Length: {total_data}")

    if not os.path.exists(args.trajectory_pca_cache_path):
        subsets = [[] for _ in range(args.works_num)]
        for i in range(total_data):
            part_index = i % args.works_num
            subsets[part_index].append(data[i])

        print("========== Hidden-state 3D PCA + angle scoring ==========")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_map
        num_gpus = len(args.device_map.split(','))
        with multiprocessing.Pool(processes=args.works_num) as pool:
            try:
                processed_subsets = pool.starmap(
                    work,
                    [(subset, num_gpus, i) for i, subset in enumerate(subsets)]
                )
            except Exception as e:
                print(f"Exception caught: {e}")
                pool.terminate()
                pool.join()
                raise

        merged_traj_rows = []
        while any(processed_subsets):
            for subset in processed_subsets:
                if subset:
                    merged_traj_rows.append(subset.pop(0))
        save_to_jsonl(merged_traj_rows, args.trajectory_pca_cache_path)
    else:
        print("========== Trajectory PCA cache loading ==========")
        merged_traj_rows = load_jsonl_file(args.trajectory_pca_cache_path)
        total_rows = args.max_len if args.max_len != -1 else len(merged_traj_rows)
        merged_traj_rows = merged_traj_rows[:total_rows]
        print(f"Data Length: {total_rows}")

    print("========== Build latent / compressed transcripts ==========")
    save_data = []
    oai_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    max_latent_len = 0
    for i, item in tqdm(enumerate(merged_traj_rows), total=len(merged_traj_rows), desc="tiering"):
        # original conv
        base_conv = copy.deepcopy(data[i]["conversations"])
        thought_steps = item["thought_steps"]
        solution = item["solution"]

        chosen_steps_flag = flags_from_trajectory_cache_row(
            item,
            angle_threshold_deg=args.pca_angle_threshold,
            num_steps=len(thought_steps),
        )
        compressed_thought = compressed_thought_from_flags(
            thought_steps, chosen_steps_flag
        )

        # 删掉开头 latent token，强制 text 开头
        try:
            first_one_idx = chosen_steps_flag.index(1)
            chosen_steps_flag[:first_one_idx] = [1] * first_one_idx
        except ValueError:
            # 如果全为 0，则全部翻转为 1
            chosen_steps_flag = [1] * len(chosen_steps_flag)
        
        latent_steps = []
        new_latent_step = "<latent>"
        latent_len = 1
        for step, f in zip(thought_steps, chosen_steps_flag):
            if f == 1:
                if new_latent_step != "<latent>":
                    new_latent_step += "</latent>"
                    latent_steps.append(new_latent_step)
                    new_latent_step = "<latent>"
                latent_steps.append(step)
            elif f == 0:
                new_latent_step += f"<latent_{latent_len}>"
                latent_len += 1
            else:
                raise ValueError(f"Unexpected chosen step flag: {f}")
        
        if new_latent_step != "<latent>":
            new_latent_step += "</latent>"
            latent_steps.append(new_latent_step)
        
        max_latent_len = max(max_latent_len, latent_len)

        if (len(chosen_steps_flag) - sum(chosen_steps_flag)) != latent_len - 1:
            print(i)
            print(f" {(len(chosen_steps_flag) - sum(chosen_steps_flag))}, latent_len: {latent_len - 1}")
        
        latent_thought = "\n\n".join(thought_step for thought_step in latent_steps)
        latent_conv = copy.deepcopy(base_conv)
        if args.solution_begin_tag is not None and args.solution_end_tag is not None:
            latent_conv[1]["value"] = (
                f"{args.thought_begin_tag}\n\n{latent_thought}\n\n{args.thought_end_tag}\n\n"
                f"{args.solution_begin_tag}\n\n{solution}\n\n{args.solution_end_tag}"
            )
        else:
            latent_conv[1]["value"] = (
                f"{args.thought_begin_tag}\n\n{latent_thought}\n\n{args.thought_end_tag}\n\n{solution}"
            )

        origin_tokens = len(oai_tokenizer.encode(data[i]["assistant_thought"]))
        compressed_tokens = len(oai_tokenizer.encode(compressed_thought))
        rate = 1.0 if compressed_tokens == 0 else compressed_tokens / origin_tokens
        data[i]["rate"] =  round(rate, 4)

        save_data.append({
            "system": data[i]["system"],
            "latent_conversations": latent_conv,
            "original_conversations": base_conv,
            "chosen_steps_flag": chosen_steps_flag,
            "steps": thought_steps,
        })
    
    print("========== Data Saving... ==========")
    save_to_jsonl(save_data, args.output_path)
    
    total_compress_rate = sum(item["rate"] for item in data if "rate" in item)
    count = sum(1 for item in data if "rate" in item)
    average_compress_rate = total_compress_rate / count if count > 0 else 0

    print("========== Data Example ==========")
    print(data[0])
    print(f"========== Average Compress Rate: {average_compress_rate * 100 :.2f}% ==========")

    print(f"========== Max Latent Length: {max_latent_len} ==========")

    print("========== Data Saving Path ==========")
    print(args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=str,
        default="openai-community/gpt2",
        help="HF 模型目录或 Hub id（AutoModelForCausalLM）.",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_true",
        default=False,
        help="不向 from_pretrained 透传 trust_remote_code=True（仅限非常规权重）.",
    )
    parser.add_argument("--works_num", type=int, default=1, help="The work number.")
    parser.add_argument("--device_map", type=str, default="0,1,2,3,4,5,6,7", help="The scorer device map.")
    parser.add_argument("--data_path", type=str, default='s1K-1.1/data/train-00000-of-00001.parquet', help="The data directory path.")
    parser.add_argument("--output_path", type=str, default='./res/data/s1K-1.1-compressed.jsonl', help="The output data path.")
    parser.add_argument(
        "--trajectory_pca_cache_path",
        type=str,
        default="./res/cache/s1K-1.1-trajectory-pca.jsonl",
        help="缓存 trajectory：原始会话、question/thought_steps/solution、PCA-3D 下 z0 与各步 z_i、夹角等；不写 hidden tensor。tiering 可仅用夹角与本阈值算 flag；若条目含 extracted 字段则与之对齐。",
    )
    parser.add_argument("--thought_begin_tag", type=str, default='<|begin_of_thought|>', help="The tag for thought begin.")
    parser.add_argument("--thought_end_tag", type=str, default='<|end_of_thought|>', help="The tag for thought end.")
    parser.add_argument("--solution_begin_tag", type=str, default=None, help="The tag for solution begin.")
    parser.add_argument("--solution_end_tag", type=str, default=None, help="The tag for solution end.")
    parser.add_argument(
        "--pca_angle_threshold",
        type=parse_angle_threshold_deg,
        default=90.0,
        help="step 位移 z_i 与 z_0（question→solution 在 PCA-3D 中）夹角阈值（度，范围 [0,180]）；大于则该 step flag=0（压缩为 latent）。",
    )
    parser.add_argument("--max_len", type=int, default=-1, help="The max length for data loading.")
    args = parser.parse_args()

    # LongCoT compress: hidden-state 3D PCA trajectory + angle-based latent compression
    main()