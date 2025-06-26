ray stop -f
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray start --address 172.24.75.90:6379 --num-gpus 8 --num-cpus 128