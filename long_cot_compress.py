import argparse
import tiktoken
from astarthought.prompt_compressor import PromptCompressor
from astarthought.validator import Validator
from astarthought.a_star import AStar
from datasets import load_dataset
import re
from tqdm import tqdm
import json
import torch
import multiprocessing
import os

def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def save_to_jsonl(data, file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def load_data(data_path):
    original_data = load_dataset("parquet", data_files={'train': data_path}, split="train")
    thought_begin_tag = args.thought_begin_tag
    thought_end_tag = args.thought_end_tag
    solution_begin_tag = args.solution_begin_tag
    solution_end_tag = args.solution_end_tag
    data = []
    for i, row_data in enumerate(original_data):
        user = row_data["conversations"][0]["value"]
        assistant = row_data["conversations"][1]["value"]
        thought_begin_index = assistant.find(thought_begin_tag)
        thought_end_index = assistant.find(thought_end_tag)
        solution_begin_index = assistant.find(solution_begin_tag)
        solution_end_index = assistant.find(solution_end_tag)
        if thought_begin_index != -1 and thought_end_index != -1:
            assistant_thought = assistant[thought_begin_index + len(thought_begin_tag): thought_end_index].strip()
        if solution_begin_index != -1 and solution_end_index != -1:
            assistant_solution = assistant[solution_begin_index + len(solution_begin_tag): solution_end_index].strip()
        data.append({"system": row_data["system"], "conversations": row_data["conversations"], "assistant_thought": assistant_thought, "assistant_solution": assistant_solution})
    return data

def load_s1K_data(data_path):
    original_data = load_dataset("parquet", data_files={'train': data_path}, split="train")
    system = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"
    data = []
    for i, row_data in enumerate(original_data):
        assistant_thought = row_data["deepseek_thinking_trajectory"]
        assistant_solution = row_data["deepseek_attempt"]
        assistant = f"""{args.thought_begin_tag}\n\n{assistant_thought}\n\n{args.thought_end_tag}\n\n{args.solution_begin_tag}\n\n{assistant_solution}\n\n{args.solution_end_tag}"""
        conversations = [{"from": "user", "value": row_data["question"]},
        { "from": "assistant", "value": assistant}]
        data.append({"system": system, "conversations": conversations, "assistant_thought": assistant_thought, "assistant_solution": assistant_solution}) #
    return data

def work(data, num_gpus, process_id):
    try:
        print(f"Process {process_id + 1} Started")
        if "cpu" not in args.scorer_device_map:
            scorer = PromptCompressor(model_name=args.scorer_model_path, device_map=f"cuda:{process_id % num_gpus}", model_config={"torch_dtype": torch.float16})
        else:
            scorer = PromptCompressor(model_name=args.scorer_model_path, device_map="cpu", model_config={"torch_dtype": torch.float16})
    
        output_data = []
        for i, row_data in tqdm(enumerate(data), total=len(data), desc=f"Process {process_id + 1}"):
            torch.cuda.empty_cache()
            bis_sentence_sort = scorer.get_bis_sentence_sort(
                context=row_data["assistant_thought"], 
                question=row_data["conversations"][0]["value"], 
                solution=row_data["assistant_solution"], 
                alpha=args.alpha
            )
            output_data.append(bis_sentence_sort)
            
        print(f"Process {process_id + 1} Complete")
        return output_data
        
    except Exception as e:
        import sys
        import traceback
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        print(f'[worker-{process_id + 1}][pid-{os.getpid()}] CRASH\n{"".join(tb_lines)}',
              file=sys.stderr, flush=True)
        raise

def main():
    print("========== Data Loading ==========")
    if args.load_s1k == True:
        data = load_s1K_data(args.data_path)
    else:
        data = load_data(args.data_path)
    total_data = args.max_len if args.max_len != -1 else len(data)
    print(f"Data Length: {total_data}")

    if not os.path.exists(args.cache_path):
        subsets = [[] for _ in range(args.scorer_works_num)]
        for i in range(total_data):
            part_index = i % args.scorer_works_num
            subsets[part_index].append(data[i])

        print("========== BIS Scoring ==========")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.scorer_device_map
        num_gpus = len(args.scorer_device_map.split(','))
        processed_subsets = [None] * args.scorer_works_num
        with multiprocessing.Pool(processes=args.scorer_works_num) as pool:
            try:
                results = pool.starmap(work, [(subset, num_gpus, i) for i, subset in enumerate(subsets)])
            except Exception as e:
                print(f"Exception caught: {e}")
                pool.terminate()
                pool.join()
        processed_subsets = results

        merged_bis_sent_sort = []
        while any(processed_subsets):
            for subset in processed_subsets:
                if subset:
                    merged_bis_sent_sort.append(subset.pop(0))
        save_to_jsonl(merged_bis_sent_sort, args.cache_path)
    else:
        print("========== BIS Score Cache Loading ==========")
        merged_bis_sent_sort = load_jsonl_file(args.cache_path)
        total_sent_sort = args.max_len if args.max_len != -1 else len(merged_bis_sent_sort)
        merged_bis_sent_sort = merged_bis_sent_sort[:total_sent_sort]
        print(f"Data Length: {total_sent_sort}")
        
    print("========== Validator Loading ==========")
    validator = Validator(args.validator_model_path, args.validator_device_map, args.beta)
    if "Bespoke-Stratos" in args.validator_model_path:
        validator_type = "BS"
    elif "s1" in args.validator_model_path:
        validator_type = "s1"
    else:
        validator_type = "DS"
    
    print("========== A* Searching ==========")
    save_data = []
    oai_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    for i, item in tqdm(enumerate(merged_bis_sent_sort), total=len(merged_bis_sent_sort), desc="A* Search"):
        a_star = AStar(
            thought_steps=item["thought_steps"],
            question=item["question"],
            solution=item["solution"],
            sent_sort=item["sent_sort"],
            expand_size=args.a_star_expand_size,
            min_search_steps=args.min_search_steps,
            max_search_steps=args.max_search_steps,
            validator=validator,
            validator_type=validator_type
        )
        compressed_thought, search_steps = a_star.search()
        compressed_assistant = f"""{args.thought_begin_tag}\n\n{compressed_thought}\n\n{args.thought_end_tag}\n\n{args.solution_begin_tag}\n\n{item["solution"]}\n\n{args.solution_end_tag}"""
        data[i]["conversations"][1]["value"] = compressed_assistant
        data[i]["search_steps"] = search_steps

        origin_tokens = len(oai_tokenizer.encode(data[i]["assistant_thought"]))
        compressed_tokens = len(oai_tokenizer.encode(compressed_thought))
        ratio = 1 if compressed_tokens == 0 else origin_tokens / compressed_tokens
        data[i]["rate"] =  round(1 / ratio, 4)
        save_data.append({"system": data[i]["system"], "conversations": data[i]["conversations"], "rate": round(1 / ratio, 4), "search_steps": search_steps})
    
    del validator
    print("========== Data Saving... ==========")
    save_to_jsonl(save_data, args.output_path)
    
    total_compress_rate = sum(item["rate"] for item in data if "rate" in item)
    count = sum(1 for item in data if "rate" in item)
    average_compress_rate = total_compress_rate / count if count > 0 else 0

    print("========== Data Example ==========")
    print(data[0])
    print(f"========== Average Compress Rate: {average_compress_rate * 100 :.2f}% ==========")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer_model_path", type=str, default='openai-community/gpt2', help="The scorer path.")
    parser.add_argument("--validator_model_path", type=str, default='simplescaling/s1.1-32B', help="The validator path.")
    parser.add_argument("--scorer_works_num", type=int, default=1, help="The work number for scorer.")
    parser.add_argument("--scorer_device_map", type=str, default="0,1,2,3,4,5,6,7", help="The scorer device map.")
    parser.add_argument("--validator_device_map", type=str, default="0,1,2,3,4,5,6,7", help="The validator device map.")
    parser.add_argument("--data_path", type=str, default='s1K-1.1/data/train-00000-of-00001.parquet', help="The data directory path.")
    parser.add_argument("--load_s1k", action="store_true", help="whether load s1K-1.1 data.")
    parser.add_argument("--output_path", type=str, default='./res/data/s1K-1.1-compressed.jsonl', help="The output data path.")
    parser.add_argument("--cache_path", type=str, default='./res/cache/s1K-1.1-bis.jsonl', help="The cached BIS data path.")
    parser.add_argument("--thought_begin_tag", type=str, default='<|begin_of_thought|>', help="The tag for thought begin.")
    parser.add_argument("--thought_end_tag", type=str, default='<|end_of_thought|>', help="The tag for thought end.")
    parser.add_argument("--solution_begin_tag", type=str, default='<|begin_of_solution|>', help="The tag for solution begin.")
    parser.add_argument("--solution_end_tag", type=str, default='<|end_of_solution|>', help="The tag for solution end.")
    parser.add_argument("--alpha", type=float, default=0.5, help="The adjustment factor for BIS.")
    parser.add_argument("--beta", type=float, default=0.1, help="The adjustment factor for cost g.")
    parser.add_argument("--a_star_expand_size", type=int, default=2, help="The expand size for A* Search.")
    parser.add_argument("--min_search_steps", type=int, default=5, help="The min search steps for A* Search.")
    parser.add_argument("--max_search_steps", type=int, default=20, help="The max search steps for A* Search.")
    parser.add_argument("--max_len", type=int, default=-1, help="The max length for data loading.")
    args = parser.parse_args()

    # LongCoT Compress with A* Search
    main()
