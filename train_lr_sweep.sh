#!/bin/bash

seed=42

for fold in 0 1; do
# for base_lr in 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001; do
for base_lr in 0.01 0.005 0.001; do
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_lseg_zs_proxy.py --dataset pascal --data_path /mnt/server14_hard1/msson/datasets/zs3_datasets --batch_size 2 --exp_name lseg_pascal_zs_proxy5_param_seed${seed}_lr${base_lr} --base_lr ${base_lr} --weight_decay 1e-4 --no-scaleinv --max_epochs 10 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384 --gpus 4 --no_resume --fold ${fold} --version ${fold} --seed ${seed}  --logpath /mnt/server14_hard1/msson/lang-seg/logs/${seed}/${fold}/${base_lr} --ckpt_save_path /mnt/server14_hard1/msson/lang-seg/ --use_proxy
done
done
