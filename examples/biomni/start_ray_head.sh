ray stop -f
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7          # 8 GPUs
ray start --head --port=6379 --dashboard-host=0.0.0.0 \
          --num-gpus 8 --num-cpus 128