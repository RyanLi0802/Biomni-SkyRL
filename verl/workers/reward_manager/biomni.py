from __future__ import annotations
from typing import Any, Callable
import importlib
import numpy as np
import torch
from verl import DataProto
from verl.workers.agentic.biomni.task.screen_design import screen_design

class BiomniRewardManager:
    def __init__(self, tokenizer, num_examine, config, compute_score=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.config = config

        self.task_mapping = {
            "screen_design": screen_design(),
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

        rewards = []
        for task_name, inp, out in zip(task_names, inputs, solutions):
            score = float(self.task_mapping[task_name].reward(inp, out))
            rewards.append(score)

        rewards = torch.tensor(rewards, dtype=torch.float32,
                               device=data.batch["responses"].device)

        # Expand to token-level: reward only last token
        resp_len = data.batch["responses"].shape[-1]
        token_mask = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        valid_len  = data.batch["attention_mask"][:, -resp_len:].sum(-1)
        for i, L in enumerate(valid_len):
            token_mask[i, int(L.item()) - 1] = rewards[i]
        
        if return_dict:
            return {
                "reward_tensor": token_mask,
                "reward_extra_info": {"raw_score": rewards.cpu().tolist()}
            }

        reward_tensor_dict = {"task_reward": token_mask,
                              "all": token_mask}
        reward_metrics = {"task_reward_mean": rewards.mean().item()}

        return reward_tensor_dict, reward_metrics
