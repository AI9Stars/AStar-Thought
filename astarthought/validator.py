from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
import math
import os
import re
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"

class Validator:
    def __init__(self, model_name: str, device_map: str = "7", beta: float = 0.1):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = device_map
        self.model = LLM(model_name, dtype="float16", tensor_parallel_size=len(device_map.split(',')), gpu_memory_utilization=0.7, enable_prefix_caching=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.beta = beta

    def extract_boxed_content(self, text: str):
        """Extract the content inside the last \boxed{} using regular expression"""
        matches = re.findall(r"\\boxed\{(.*?)\}", text)
        return matches[-1] if matches else None
    
    def evaluate_cost(self, inputs: list, total_question_tokens: int, total_solution_tokens: int):
        cost_g, cost_h = [], []
        sampling_params = SamplingParams(prompt_logprobs=0, max_tokens=1)
        outputs = self.model.generate(inputs, sampling_params, use_tqdm=False)
        for output in outputs:
            thought_token_logprobs = [next(iter(d.values())).logprob for d in output.prompt_logprobs[total_question_tokens: -total_solution_tokens] if d]
            solution_token_logprobs = [next(iter(d.values())).logprob for d in output.prompt_logprobs[-total_solution_tokens + 1:] if d] # skip solution_begin_tag
            thought_avg_logprob = sum(thought_token_logprobs) / len(thought_token_logprobs)
            solution_avg_logprob = sum(solution_token_logprobs) / len(solution_token_logprobs)
            cost_g.append(- self.beta * thought_avg_logprob)
            cost_h.append(- solution_avg_logprob)
        return cost_g, cost_h
    
    def evaluate_solution(self, inputs, solution: str, max_tokens: int = 1024):
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.6, top_p=0.95)
        outputs = self.model.generate(inputs, sampling_params, use_tqdm=False)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            if self.extract_boxed_content(generated_text) == self.extract_boxed_content(solution):
                return True
        return False