#! /bin/bash

export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PROJECT_NAME='biomni-training-qwen3-8b-ppo'
EXPERIMENT_NAME='biomni-training-qwen3-8b-32bsz-temp0.6-clip-0.28-32turn'
DATA_PATH="/afs/cs.stanford.edu/u/lansong/BioAgentOS/biomni_env_screen/data/screen_design_rl"
SFT_MODEL_PATH='/dfs/scratch0/lansong/models/qwen/qwen3-8b-sft-v1/global_step_66' 
CRITIC_MODEL_PATH=$SFT_MODEL_PATH
CKPT_PATH='/dfs/scratch0/lansong/models/qwen'
RUNTIME_URL='http://172.24.75.232:8000'
TASK_TYPE='biomni'

BATCH_SIZE=32
MAX_NUM_ITERS=32
NUM_TRAJ=8
MAX_PARALLEL_AGENTS=128
SAVE_FREQ=8

USE_KL_LOSS=True
KL_LOSS_COEF=0.001
KL_LOSS_TYPE=low_var_kl
ENTROPY_COEFF=0.001
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28

# Assumes a h200 node
# For 2xH100: change tp size -> 2, sequence parallel size -> 2, nnodes -> 2
GPU_MEM_UTIL=0.8
TP_SIZE=2
NNODES=1
SP_SIZE=2


TEMPERATURE=0.6
TOP_P=0.95

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=["$DATA_PATH/train.parquet"] \
    data.val_files=["$DATA_PATH/test.parquet"] \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=31744 \
    data.max_response_length=3072 \
    data.truncation='error' \
    data.shuffle=True \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
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
    critic.optim.lr=1e-5 \
    critic.ulysses_sequence_parallel_size=$SP_SIZE \
    critic.model.use_remove_padding=True \
    critic.model.path=$CRITIC_MODEL_PATH \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=async \
    actor_rollout_ref.rollout.task_type=$TASK_TYPE \
    +actor_rollout_ref.rollout.runtime_url=$RUNTIME_URL \
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
    actor_rollout_ref.rollout.remove_think_tokens=True \
    actor_rollout_ref.actor.masking=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=$TASK_TYPE \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.exchange_size=500000000 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 $@