#!/bin/bash

# NCCL Configuration
export NCCL_TIMEOUT=28800  # Increase timeout to 8 hours (from default 2 hours)
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1  # per-rank error report
export TORCH_NCCL_TRACE_BUFFER_SIZE=10000
export PYTHONFAULTHANDLER=1

# Original environment variables
export UV_CACHE_DIR=/dfs/scratch1/lansong/uv_cache
export XDG_CACHE_HOME=$UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT=/dfs/scratch1/lansong/venvs/skyrl
export HOME=/dfs/scratch1/lansong
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PROJECT_NAME='biomni-training-qwen3-32b-grpo'
EXPERIMENT_NAME='biomni-training-qwen3-32b-32bsz-temp0.6-clip-0.28-64turn-rope-drgrpo'
# DATA_PATH="/dfs/scratch1/lansong/BioAgentOS/biomni_env_screen/data/rl_data/all"
TRAIN_FILE='/dfs/scratch1/lansong/BioAgentOS/biomni_env_screen/data/rl_data/all/train_updated.parquet'
VAL_FILE='/dfs/scratch1/lansong/BioAgentOS/biomni_env_screen/data/rl_data/all/val.parquet'
SFT_MODEL_PATH='/dfs/scratch1/lansong/models/qwen/qwen3-32b-sft-full-v1/global_step_208' 
CKPT_PATH='/dfs/scratch1/lansong/models/qwen'
# RUNTIME_URL='http://172.24.75.232:8000'   # ampere7
RUNTIME_URL='http://172.24.75.90:8000'    # ampere9
TASK_TYPE='biomni'

BATCH_SIZE=64
MAX_NUM_ITERS=48
NUM_TRAJ=8
MAX_PARALLEL_AGENTS=256
SAVE_FREQ=2

# turn off KL loss for DAPO and DRGRPO
USE_KL_LOSS=False
KL_LOSS_COEF=0.001
KL_LOSS_TYPE=low_var_kl

# entropy coeff = 0 since we've already turned off KL loss
ENTROPY_COEFF=0.0
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28

# 2xA100: tp size -> 4, sequence parallel size -> 4, nnodes -> 2
GPU_MEM_UTIL=0.8
TP_SIZE=4
NNODES=2
SP_SIZE=4


TEMPERATURE=0.6
TOP_P=0.95


PYTHONUNBUFFERED=1 HOME=/dfs/scratch1/lansong uv run --env-file /dfs/scratch1/lansong/SkyRL/.env.biomni -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=["$TRAIN_FILE"] \
    data.val_files=["$VAL_FILE"] \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=49152 \
    data.max_response_length=4096 \
    data.truncation='error' \
    data.shuffle=True \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.rope_scaling.rope_type=yarn \
    +actor_rollout_ref.model.override_config.rope_scaling.factor=2.0 \
    +actor_rollout_ref.model.override_config.rope_scaling.original_max_position_embeddings=32768 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=async \
    actor_rollout_ref.rollout.task_type=$TASK_TYPE \
    +actor_rollout_ref.rollout.runtime_url=$RUNTIME_URL \
    '+actor_rollout_ref.rollout.json_override_args="{\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2.0,\"original_max_position_embeddings\":32768},\"max_position_embeddings\":53248}"' \
    +actor_rollout_ref.filter_overlong_invalid_trajectories=True \
    +actor_rollout_ref.overlong_trajectory_threshold=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_trajectories=$NUM_TRAJ \
    actor_rollout_ref.rollout.sampling_params.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.sampling_params.top_p=$TOP_P \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.max_parallel_agents=$MAX_PARALLEL_AGENTS \
    actor_rollout_ref.rollout.max_iterations=$MAX_NUM_ITERS \
    actor_rollout_ref.rollout.enable_memory_saver=True \
    +actor_rollout_ref.rollout.max_starting_message_length=12000 \
    actor_rollout_ref.rollout.remove_think_tokens=False \
    actor_rollout_ref.actor.masking=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=$TASK_TYPE \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.max_actor_ckpt_to_keep=128 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.exchange_size=500000000 \
    trainer.test_freq=-1 \
    trainer.total_epochs=2 $@