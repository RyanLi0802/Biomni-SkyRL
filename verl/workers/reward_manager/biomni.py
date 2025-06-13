from __future__ import annotations
from typing import Any, Callable
import importlib
import json
import numpy as np
import torch
from verl import DataProto
from verl.workers.agentic.biomni.task.screen_design import screen_design
from verl.workers.agentic.biomni.task.gwas_causal_gene import gwas_causal_gene

class BiomniRewardManager:
    def __init__(self, tokenizer, num_examine, config, compute_score=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.config = config

        top_k = config.get("top_k", 100)
        self.task_mapping = {
            "screen_design": screen_design(top_k=top_k),
            "gwas_causal_gene_pharmaprojects": gwas_causal_gene(path = '/dfs/project/bioagentos/biomni_data/benchmark/', dataset = 'pharmaprojects', num_samples = 100000),
        }

    def __call__(self, data: DataProto, *, return_dict: bool = False):
        """
        data.non_tensor_batch MUST contain
            - 'input'    (whatever your task expects as *input* to reward())
            - 'solution' (assistant's final answer)
        """
        inputs = data.non_tensor_batch["instance_id"]
        solutions = data.non_tensor_batch["solution"]
        task_names = data.non_tensor_batch["task_name"]
        messages = data.non_tensor_batch["messages"].tolist()

        rewards = []
        # gt reward
        for task_name, inp, out in zip(task_names, inputs, solutions):
            score = float(self.task_mapping[task_name].reward(inp, out))
            
            # instance_id is actually the index not screen_id, this is a temporary fix, uncomment the line above once the dataset has been updated to proper screen_id
            # ex = self.task_mapping[task_name].get_example(inp)
            # sid = ex["screen_id"]
            # score = float(self.task_mapping[task_name].reward(sid, out))
            
            rewards.append(score)
        
        gt_rewards = rewards.copy()
        ft_rewards = []
        
        # formatting reward
        for i, convo in enumerate(messages):
            valid_format = 1
            for msg in convo:
                if msg["role"] == "assistant":
                    content = msg["content"]
                    think_open = content.count("<think>")
                    think_close = content.count("</think>")
                    execute_open = content.count("<execute>")
                    execute_close = content.count("</execute>")
                    solution_open = content.count("<solution>")
                    solution_close = content.count("</solution>")
                    if (think_open != think_close or 
                        execute_open != execute_close or 
                        solution_open != solution_close):
                        valid_format = 0
                        break
                    if "<solution>" in content and "<execute>" in content:
                        # duplicated tags
                        valid_format = 0
                        break
            rewards[i] += valid_format
            ft_rewards.append(valid_format)

        rewards = torch.tensor(rewards, dtype=torch.float32,
                               device=data.batch["responses"].device)

        # Expand to token-level: reward only last token
        resp_len = data.batch["responses"].shape[-1]
        token_mask = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        valid_len  = data.batch["attention_mask"][:, -resp_len:].sum(-1)
        for i, L in enumerate(valid_len):
            token_mask[i, int(L.item()) - 1] = rewards[i]
        
        # Calculate reward means for both return paths
        gt_reward_mean = torch.tensor(gt_rewards, dtype=torch.float32, device=data.batch["responses"].device).mean().item()
        ft_reward_mean = torch.tensor(ft_rewards, dtype=torch.float32, device=data.batch["responses"].device).mean().item()
        
        if return_dict:
            return {
                "reward_tensor": token_mask,
                "reward_extra_info": {
                    "raw_score": rewards.cpu().tolist(),
                    "gt_reward_mean": gt_reward_mean,
                    "ft_reward_mean": ft_reward_mean
                }
            }

        reward_tensor_dict = {"task_reward": token_mask,
                              "all": token_mask}
        reward_metrics = {
            "task_reward_mean": rewards.mean().item(), 
            "gt_reward_mean": gt_reward_mean, 
            "ft_reward_mean": ft_reward_mean,
        }
        
        print("\nReward metrics:", reward_metrics)
        first_tokens = [reward_tensor_dict["all"][i][valid_len[i] - 1] for i in range(len(valid_len))]
        print("Reward first tokens:", first_tokens)
        reward_details = []
        
        for task_name, inp, out, convo, gt_reward, ft_reward, first_token in zip(task_names, inputs, solutions, messages, gt_rewards, ft_rewards, first_tokens):
            reward_details.append({
                "screen_id": inp,
                "gt_reward": gt_reward,
                "ft_reward": ft_reward,
                "first_token": first_token.item(),
                "prompt": convo[1]["content"],
                "output": out,
            })
        
        print("Reward details:")
        print(json.dumps(reward_details, indent=2))
        

        return reward_tensor_dict, reward_metrics
