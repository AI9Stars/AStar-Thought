import sys
import numpy as np
import torch
from .validator import Validator
import re

class AStar:
    def __init__(self, thought_steps, question, solution, sent_sort, expand_size, min_search_steps, max_search_steps, validator, validator_type):
        self.total_steps = len(thought_steps)
        self.thought_steps = thought_steps
        self.question = question
        self.solution = solution
        self.bis_queue = [i for i, _ in sent_sort]
        self.open_set = []
        self.validator = validator
        self.total_question_tokens = len(validator.tokenizer(question)['input_ids'])
        self.total_solution_tokens = len(validator.tokenizer(solution)['input_ids'])
        self.expand_size = expand_size
        self.min_search_steps = min_search_steps
        self.max_search_steps = max_search_steps
        
        if validator_type == "BS":
            self.thought_begin_tag, self.thought_end_tag, self.solution_begin_tag, self.solution_end_tag = "<|begin_of_thought|>", "<|end_of_thought|>", "<|begin_of_solution|>", "<|end_of_solution|>"
        elif validator_type == "s1":
            self.thought_begin_tag, self.thought_end_tag, self.solution_begin_tag, self.solution_end_tag = "<|im_start|>think", "", "<|im_start|>answer", ""
        else:
            self.thought_begin_tag, self.thought_end_tag, self.solution_begin_tag, self.solution_end_tag = "<think>", "</think>", "", ""
    
    def extract_boxed_content(self, text: str):
        """Extract the content inside the last \boxed{} using regular expression"""
        matches = re.findall(r"\\boxed\{(.*?)\}", text)
        return matches[-1] if matches else None
    
    def cost(self, batch):
        inputs = []
        for item in batch:
            current_thought = "\n\n".join(thought_step for i, thought_step in enumerate(self.thought_steps) if i in item)
            inputs.append(f"{self.question}{self.thought_begin_tag}{current_thought}{self.thought_end_tag}{self.solution_begin_tag}{self.solution}{self.solution_end_tag}")
        batch_cost_g, batch_cost_h = self.validator.evaluate_cost(inputs, self.total_question_tokens, self.total_solution_tokens)
        return batch_cost_g, batch_cost_h
    
    def get_next_step(self, batch):
        batch_span = [list(item.values())[0] for item in batch]
        batch_cost_g, batch_cost_h = self.cost(batch_span)
        min_index = min(
            range(len(batch)),
            key=lambda i: batch_cost_g[i] + batch_cost_h[i]
        )
        return batch[min_index]

    def is_valid_point(self, current):
        current_thought = "\n\n".join(thought_step for i, thought_step in enumerate(self.thought_steps) if i in current)
        input_text = f"{self.question}{self.thought_begin_tag}{current_thought}{self.thought_end_tag}{self.solution_begin_tag}"
        if self.validator.evaluate_solution(input_text, self.solution):
            return True
        return False
    
    def search(self):
        start = self.bis_queue[0]
        self.bis_queue.pop(0)
        self.open_set.append(
            [start] if start == 0 == self.total_steps - 1 else
            [start, start + 1] if start == 0 else
            [start - 1, start] if start == self.total_steps - 1 else
            [start - 1, start, start + 1]
        )

        i = 0
        while len(self.bis_queue) > 0:
            torch.cuda.empty_cache()
            i += 1
            # 1) Verifying
            current = self.open_set.pop(0)
            if i >= self.max_search_steps:
                print("Search steps exceed maximum limit.")
                return "\n\n".join(thought_step for thought_step in self.thought_steps), self.max_search_steps
            if i >= self.min_search_steps:
                if self.extract_boxed_content(self.solution) == None:
                    return "\n\n".join(thought_step for j, thought_step in enumerate(self.thought_steps) if j in current), i
                if self.is_valid_point(current):
                    return "\n\n".join(thought_step for j, thought_step in enumerate(self.thought_steps) if j in current), i
            # 2) Exploring
            expand = [
                {k: [item, item + 1]} if item == 0 else
                {k: [item - 1, item]} if item == self.total_steps - 1 else
                {k: [item - 1, item, item + 1]}
                for k, item in enumerate(self.bis_queue[:self.expand_size])
            ]
            expand = [
                {list(item.keys())[0]: list(set(current + list(item.values())[0]))} 
                for item in expand
            ]
            # 3) Updating
            next_step = self.get_next_step(expand)
            key = list(next_step.keys())[0] # bis_queue index
            value = list(next_step.values())[0]
            index = self.bis_queue[key] # sentence index
            self.bis_queue.pop(key)
            self.open_set.append(value)
            # Prune
            if index - 2 >= 0 and (index - 1) in value and (index - 2) in value and (index - 1) in self.bis_queue:
                key_to_remove = self.bis_queue.index(index-1)
                self.bis_queue.pop(key_to_remove)
                i += 1
            if index + 2 < self.total_steps and (index + 1) in value and (index + 2) in value and (index + 1) in self.bis_queue:
                key_to_remove = self.bis_queue.index(index+1)
                self.bis_queue.pop(key_to_remove)
                i += 1

        return "\n\n".join(thought_step for thought_step in self.thought_steps), self.max_search_steps
