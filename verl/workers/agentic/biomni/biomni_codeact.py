import asyncio, re, json, uuid, logging, aiohttp, time, os, tempfile
import sys
sys.path.append("/afs/cs.stanford.edu/u/lansong/SkyRL/")
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple

import torch
from tensordict import TensorDict
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.workers.agentic.biomni.prompt_manager import PromptManager
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)



# -- Qwen-3 chat templates ------------------------------------------------
# includes everything
chat_template = (
    "{% for message in messages %}"
    "{% if (message['role'] != 'assistant') %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% elif (message['role'] == 'assistant')%}"
    "{{'<|im_start|>' + message['role'] + '\\n'}}"
    "{% generation %}"
    "{{message['content'] + '<|im_end|>'}}"
    "{% endgeneration %}"
    "{{'\\n'}}"
    "{% endif %}"
    "{% endfor %}"
)

# drops previous <think> blocks
chat_template_qwen3_thinking = (
    "{% for message in messages %}"
    "{% if (message['role'] != 'assistant') %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% elif (message['role'] == 'assistant')%}"
    "{{'<|im_start|>' + message['role'] + '\\n'}}"
    "{% generation %}"
    "{% set full_content = message['content'] %}"
    "{% set mycontent = message['content'] %}"
    "{% set is_last_message = loop.last and messages[-1]['role'] == 'assistant' %}"
    "{% if '</think>' in full_content and not is_last_message %}"
    "{% set mycontent = full_content.split('</think>')[-1].lstrip('\\n') %}"
    "{% endif %}"
    "{{mycontent + '<|im_end|>'}}"
    "{% endgeneration %}"
    "{{'\\n'}}"
    "{% endif %}"
    "{% endfor %}"
)



class BiomniRuntimeClient:
    """Thin async wrapper around server.py endpoints."""
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base = base_url.rstrip("/")
        self.session_id: Optional[str] = None
        self._client: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._client = aiohttp.ClientSession()      # one connection pool per runtime
        self.session_id = await self._start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session_id:
            try:
                await self._client.post(f"{self.base}/delete_runtime",
                                        json={"session_id": self.session_id})
            finally:
                await self._client.close()

    # low-level helpers -------------------------------------------------
    async def _start(self) -> str:
        async with self._client.post(f"{self.base}/start_runtime") as r:
            r.raise_for_status()
            return (await r.json())["session_id"]

    async def execute(self, code: str, timeout: int = 600) -> str:
        """Run *code* inside the persistent namespace of this session."""
        payload = {"session_id": self.session_id,
                   "code": code,
                   "timeout_seconds": timeout}
        async with self._client.post(f"{self.base}/execute", json=payload,
                                     timeout=timeout+5) as r:
            r.raise_for_status()
            return (await r.json())["output"]


# ----------------------------------------------------------------------
# Helper utils
# ----------------------------------------------------------------------
_TAG_RGX = {
    "solution": re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE),
    "execute":  re.compile(r"<execute>(.*?)</execute>",  re.DOTALL | re.IGNORECASE),
    "think":    re.compile(r"<think>(.*?)</think>",      re.DOTALL | re.IGNORECASE),
}

def _parse_first(match_type: str, text: str) -> Optional[str]:
    m = _TAG_RGX[match_type].search(text)
    return m.group(1).strip() if m else None


# ----------------------------------------------------------------------
# One Agent
# ----------------------------------------------------------------------
class BiomniCodeActAgent:
    """
    Rollout loop for a single problem instance.

    Parameters
    ----------
    prompt: str
        The initial user prompt / task description.
    runtime: BiomniRuntimeClient
        A *connected* runtime. The agent does NOT own it - caller decides lifespan.
    infer_engine, tokenizer, sampling_params
        Passed straight to sglang.
    max_iterations : int
        Hard limit to avoid infinite loops.
    """
    def __init__(
        self,
        prompt: str,
        runtime: BiomniRuntimeClient,
        infer_engine,
        tokenizer: PreTrainedTokenizerBase,
        sampling_params: Dict[str, Any],
        max_prompt_len: int = 31744,
        max_iterations: int = 32,
        qwen3_enable_thinking: bool = True,
    ):
        self.runtime = runtime
        self.engine = infer_engine
        self.tok = tokenizer
        self.sampling_params = sampling_params
        self.max_prompt_len = max_prompt_len
        self.max_iterations = max_iterations
        self.qwen3_enable_thinking = qwen3_enable_thinking
        self.prompt_manager = PromptManager(tool_path="verl/workers/agentic/biomni/tool")

        # -- conversation memory ------------------------------------------------
        self.messages = self.prompt_manager.get_initial_messages(prompt)
        self.log: List[Dict[str, str]] = []   # optional external logging

    # ------------------------------------------------------------------
    # generation & routing
    # ------------------------------------------------------------------
    async def _llm_generate(self) -> str:
        """Call sglang engine asynchronously and return *raw* assistant string."""
        input_ids = self.tok.apply_chat_template(
            self.messages, add_generation_prompt=True,
            tokenize=True, chat_template=chat_template, enable_thinking=self.qwen3_enable_thinking
        )
        if len(input_ids) >= self.max_prompt_len:
            return "The context is too long. Exit now."

        res = await self.engine.async_generate(
            input_ids=input_ids,
            sampling_params=self.sampling_params,
        )
        return res["text"]

    async def run(self) -> Dict[str, Any]:
        """
        Execute the interaction loop.

        Returns
        -------
        dict
            {
              "messages": <full conversation>,
              "solution": str | None,
              "iterations": int,
            }
        """
        solution: Optional[str] = None

        for step in range(1, self.max_iterations + 1):
            assistant_reply = await self._llm_generate()
            if assistant_reply == "The context is too long. Exit now.":
                self.messages.append({"role": "user", "content": "The context is too long. Exit now."})
                self.log.append({"role": "user", "content": "The context is too long. Exit now."})
                break
            self.messages.append({"role": "assistant", "content": assistant_reply})
            self.log.append({"role": "assistant", "content": assistant_reply})

            # -- parse ----------------------------------------------------------
            sol = _parse_first("solution", assistant_reply)
            if sol:
                solution = sol
                break
            
            code = _parse_first("execute", assistant_reply)
            if code is not None:
                try:
                    out = await self.runtime.execute(code)
                except Exception as e:
                    out = f"[runtime-error] {e}"
                # feed runtime output back as user message
                self.messages.append({"role": "user", "content": f"<observation>{out}</observation>"})
                self.log.append({"role": "user", "content": f"<observation>{out}</observation>"})
                continue

            # optional <think> branch – do nothing but continue
            if _parse_first("think", assistant_reply) is not None:
                continue

            # Malformed – corrective feedback
            error_count = sum(
                1
                for m in self.messages
                if m["role"] == "user" and "There are no tags" in m["content"]
            )
            if error_count >= 2:
                self.messages.append(
                    {"role": "user",
                     "content": "Execution terminated due to repeated parsing errors."}
                )
                break
            self.messages.append(
                {"role": "user",
                 "content": "There are no tags (e.g. <execute><solution>). "
                            "Please follow the instruction, fix and update."}
            )

        return {
            "messages": self.messages,
            "solution": solution,
            "iterations": step,
        }


def _left_pad(input_ids: torch.Tensor,
              mask: torch.Tensor,
              pad_token_id: int,
              tgt_len: int,
              device) -> Tuple[torch.Tensor, torch.Tensor]:
    bs, cur = input_ids.shape
    out_ids = torch.full((bs, tgt_len), pad_token_id,
                         dtype=torch.long, device=device)
    out_mask = torch.zeros((bs, tgt_len),
                           dtype=torch.long, device=device)
    for i in range(bs):
        seq_len = int(mask[i].sum())
        seq_len = min(seq_len, tgt_len)
        offset = tgt_len - seq_len
        out_ids[i, offset:] = input_ids[i, :seq_len]
        out_mask[i, offset:] = 1
    return out_ids, out_mask


def _pad_right(t: torch.Tensor, length: int, val: int) -> torch.Tensor:
    if t.shape[1] >= length:
        return t[:, :length]
    pad = (0, length - t.shape[1])
    return torch.nn.functional.pad(t, pad, value=val)


class BiomniCodeActAgentGroup:
    """
    Generate num_trajectories rollouts for each DataProto item in batch.

    Each item in batch must carry `non_tensor_batch['raw_prompt']`
    (string with the initial task prompt).  Nothing else is required.
    """
    def __init__(
        self,
        batch: DataProto,
        num_trajectories: int,
        infer_engine,
        tokenizer: PreTrainedTokenizerBase,
        sampling_params: Dict[str, Any],
        *,
        runtime_url: str = "http://localhost:8000",
        device: str | torch.device = "cpu",
        max_prompt_length: int = 31744,
        max_response_length: int = 3072,
        max_starting_message_length: int = 12000,
        max_iterations: int = 32,
        max_parallel_agents: int = 32,
        qwen3_enable_thinking: bool = True,
        remove_think_tokens: bool = False,
    ):
        self.orig_batch = batch
        self.nt = num_trajectories
        self.engine = infer_engine
        self.tok = tokenizer
        self.sampling_params = sampling_params
        self.runtime_url = runtime_url
        self.device = device
        self.max_prompt_len = max_prompt_length
        self.max_resp_len = max_response_length
        self.total_len = self.max_prompt_len + self.max_resp_len
        self.max_iters = max_iterations
        self.max_parallel = max_parallel_agents
        self.qwen_think = qwen3_enable_thinking
        self.remove_think_tokens = remove_think_tokens
        self.max_starting_message_length = max_starting_message_length
    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self) -> DataProto:
        res = asyncio.run(self._async_run_group())
        return res

    # ------------------------------------------------------------------
    # asyncio helpers
    # ------------------------------------------------------------------
    async def _async_run_group(self) -> DataProto:
        sem = asyncio.Semaphore(self.max_parallel)
        tasks = []
        results: List[Dict[str, Any]] = []

        async def _run_single(prompt: str):
            async with sem, BiomniRuntimeClient(self.runtime_url) as rt:
                agent = BiomniCodeActAgent(
                    prompt=prompt,
                    runtime=rt,
                    infer_engine=self.engine,
                    tokenizer=self.tok,
                    sampling_params=self.sampling_params,
                    max_prompt_len=self.max_prompt_len,
                    max_iterations=self.max_iters,
                    qwen3_enable_thinking=self.qwen_think,
                )
                out = await agent.run()
                results.append(out)

        # schedule
        for data_item in self.orig_batch:
            instr = data_item.non_tensor_batch["raw_prompt"]
            for _ in range(self.nt):
                tasks.append(asyncio.create_task(_run_single(instr)))

        await asyncio.gather(*tasks)
        return self._pack_results(results)

    # ------------------------------------------------------------------
    # packing into DataProto (same tensor layout as CodeAct) ------------
    # ------------------------------------------------------------------
    def _pack_results(self, outputs: List[Dict[str, Any]]) -> DataProto:
        # 1) split conversation into PROMPT (up to just before first assistant)
        #    and RESPONSE (from first assistant on)
        prompts, responses = [], []
        for convo in outputs:
            msgs = convo["messages"]
            split_at = next(
                (i for i, m in enumerate(msgs) if m["role"] == "assistant"), len(msgs)
            )
            prompts.append(msgs[:split_at])
            responses.append(msgs[split_at:])

        # 2) tokenizer encode with chat template
        prompt_enc = self.tok.apply_chat_template(
            prompts, add_generation_prompt=False, return_dict=True, padding=True
        )
        resp_enc = self.tok.apply_chat_template(
            responses, add_generation_prompt=False, return_dict=True,
            chat_template=chat_template_qwen3_thinking if self.remove_think_tokens else chat_template,
            return_assistant_tokens_mask=True, padding=True
        )

        pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        # left-pad prompt to fixed len
        pr_ids = torch.tensor(prompt_enc["input_ids"], device=self.device)
        pr_mask = torch.tensor(prompt_enc["attention_mask"], device=self.device)
        pr_ids, pr_mask = _left_pad(
            pr_ids, pr_mask, pad_id, self.max_starting_message_length, self.device
        )

        # right-pad response to fixed len
        rsp_ids = torch.tensor(resp_enc["input_ids"], device=self.device)
        rsp_mask = torch.tensor(resp_enc["attention_mask"], device=self.device)
        rsp_ass_mask = torch.tensor(resp_enc["assistant_masks"], device=self.device)
        
        # this is okay, the padding will be removed in training
        rsp_ids = _pad_right(rsp_ids, self.total_len, pad_id)
        rsp_mask = _pad_right(rsp_mask, self.total_len, 0)
        rsp_ass_mask = _pad_right(rsp_ass_mask, self.total_len, 0)

        # concat
        input_ids = torch.cat([pr_ids, rsp_ids], dim=1)              # (B, total_len)
        attn_mask = torch.cat([pr_mask, rsp_mask], dim=1)
        pos_ids = compute_position_id_with_mask(attn_mask)

        batch = TensorDict(
            {
                "input_ids": input_ids,
                "responses": rsp_ids,
                "attention_mask": attn_mask,
                "position_ids": pos_ids,
                "loss_mask": rsp_ass_mask,        # train only on assistant tokens
            },
            batch_size=input_ids.size(0),
        )

        # non-tensor info (solution string, iterations, raw messages)
        non_tensor = {
            "solution": [o["solution"] for o in outputs],
            "iterations": [o["iterations"] for o in outputs],
            "messages": [o["messages"] for o in outputs],
            "instance_id": self.orig_batch.non_tensor_batch["instance_id"].tolist(),
            "task_name": self.orig_batch.non_tensor_batch["task_name"].tolist(),
        }

        return DataProto.from_dict(tensors=batch, non_tensors=non_tensor)


# ----------------------------------------------------------------------
# convenience CLI ------------------------------------------------------
if __name__ == "__main__":
    import argparse, json
    from sglang import Engine
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--instance_id", type=int, required=True)
    args = parser.parse_args()

    # -- quick demo -----------------------------------------------------
    # eng = Engine(args.model_path, max_num_seqs=8, dtype="float16", device="cuda")
    eng = Engine(
                model_path=args.model_path,
                port=40000,
                dtype="float16",
                max_total_tokens=60*32768,
                max_prefill_tokens=2*32768,
                enable_memory_saver=True,
                mem_fraction_static=0.4,
                tp_size=4,
                log_level="INFO",
                # enable_metrics=True,
            )
    tok = AutoTokenizer.from_pretrained(args.model_path)
    sampling = {"max_new_tokens": 4096, "temperature": 1.0}

    dummy = DataProto.from_dict(
        tensors={"input_ids": torch.zeros(1, 32768, dtype=torch.long)},
        non_tensors={"raw_prompt": [args.prompt], "instance_id": [args.instance_id], "task_name": ["screen_design"]},
    )
    grp = BiomniCodeActAgentGroup(
        batch=dummy,
        num_trajectories=1,
        infer_engine=eng,
        tokenizer=tok,
        sampling_params=sampling,
        runtime_url="http://172.24.75.232:8000",
    )
    dp = grp.run()
    msg0 = dp.non_tensor_batch["messages"][0].tolist()
    print(json.dumps(msg0, indent=2))
    
    print("="*20)
    
    from verl.workers.reward_manager.biomni import BiomniRewardManager
    reward_manager = BiomniRewardManager(tokenizer=tok, num_examine=1, config={})
    result_tensor, result_metrics = reward_manager(dp)
    response_ids = dp.batch['responses']
    response_length = response_ids.shape[-1]
    valid_response_length = dp.batch['attention_mask'][:, -response_length:].sum(-1)
    print(result_metrics)
    print(result_tensor['all'][0][valid_response_length[0] - 1])
