from __future__ import annotations
from typing import Any, Callable
import importlib
import json
import re
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
        
        # ------------------------------------------------------------------
        # Formatting reward.
        #   Every assistant message must follow one of the two templates:
        #       1. "<think>...</think> ... <execute>...</execute>"
        #       2. "<think>...</think> ... <solution>...</solution>"
        #   * <solution></solution> must appear exactly once and only in the
        #     final assistant block.
        #   * <solution> and <execute> must never co-exist in the same block.
        #   * Tags must not interleave (e.g. <think><execute></execute></think>).
        # ------------------------------------------------------------------

        # Pre-compiled regex to extract XML-like tags of interest
        tag_pattern = re.compile(r"</?(think|execute|solution)>", re.IGNORECASE)

        def _valid_block(content: str, *, is_last: bool) -> bool:
            """Check whether *content* of an assistant message satisfies the
            formatting rules described above.  The *is_last* flag indicates if
            this is the final assistant turn in the conversation (required for
            <solution>)."""

            # Quick reject if mixed execute & solution appear at all
            if "<execute>" in content and "<solution>" in content:
                return False

            # Gather the tag sequence in order of appearance
            tags = [m.group(0) for m in tag_pattern.finditer(content)]

            # We expect exactly four tags: opening/closing <think>, followed by
            #   opening/closing <execute> *or* <solution>.
            if len(tags) != 4:
                return False

            # Validate first pair is <think> ... </think>
            if tags[0].lower() != "<think>" or tags[1].lower() != "</think>":
                return False

            # Second pair must be execute or solution consistently
            second_open, second_close = tags[2], tags[3]
            if second_open.lower() not in ("<execute>", "<solution>"):
                return False

            expected_close = second_open.replace("<", "</")
            if second_close.lower() != expected_close.lower():
                return False

            # If it's a solution tag, ensure this message is the last
            if second_open.lower() == "<solution>" and not is_last:
                return False
            elif second_open.lower() != "<solution>" and is_last:
                return False

            # Ensure no execute/solution tags appear inside the <think> block
            think_block = content.split(tags[0], 1)[1].split(tags[1], 1)[0]
            if "<execute>" in think_block or "<solution>" in think_block:
                return False

            # Ensure no think tags after the first closing </think>
            after_think = content.split(tags[1], 1)[1]
            if "<think>" in after_think or "</think>" in after_think:
                return False

            return True

        for i, convo in enumerate(messages):
            valid_format = 1

            # gather indices of assistant messages
            assistant_indices = [idx for idx, m in enumerate(convo) if m["role"] == "assistant"]
            if not assistant_indices:
                valid_format = 0
                ft_rewards.append(valid_format)
                rewards[i] += valid_format
                continue

            last_assistant_idx = assistant_indices[-1]

            for idx in assistant_indices:
                content = convo[idx]["content"]

                is_last_msg = idx == last_assistant_idx
                if not _valid_block(content, is_last=is_last_msg):
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
        
        # ------------------------------------------------------------------
        # Compute mean gt reward per task
        # ------------------------------------------------------------------
        task_to_gt: dict[str, list[float]] = {}
        for tname, gt in zip(task_names, gt_rewards):
            task_to_gt.setdefault(tname, []).append(gt)
        gt_reward_mean_per_task = {k: float(np.mean(v)) for k, v in task_to_gt.items()}
        
        # Flatten keys for WandB compatibility (no nested dicts)
        flat_gt_reward_mean_per_task = {f"gt_reward_mean_per_task/{k}": v for k, v in gt_reward_mean_per_task.items()}
        
        if return_dict:
            return {
                "reward_tensor": token_mask,
                "reward_extra_info": {
                    "raw_score": rewards.cpu().tolist(),
                    "gt_reward_mean": gt_reward_mean,
                    "ft_reward_mean": ft_reward_mean,
                    **flat_gt_reward_mean_per_task
                }
            }

        reward_tensor_dict = {"task_reward": token_mask,
                              "all": token_mask}
        reward_metrics = {
            "task_reward_mean": rewards.mean().item(), 
            "gt_reward_mean": gt_reward_mean, 
            "ft_reward_mean": ft_reward_mean,
            **flat_gt_reward_mean_per_task,
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
