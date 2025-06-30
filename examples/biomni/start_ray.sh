export UV_CACHE_DIR=/dfs/scratch1/lansong/uv_cache
export XDG_CACHE_HOME=$UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT=/dfs/scratch1/lansong/venvs/skyrl
export RAY_RUNTIME_ENV_WORKING_DIR_CACHE_SIZE_GB=64
export HOME=/dfs/scratch1/lansong

ray stop -f
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
uv run ray start --address 172.24.75.90:6379 --num-gpus 8 --num-cpus 128