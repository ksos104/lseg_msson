#!/bin/bash

seed=42
fold=0
base_lr=0.1
batch_size=2
version=8

## target others
# CUDA_VISIBLE_DEVICES=4,5,6,7 python train_lseg_zs_proxy.py --dataset pascal --data_path /mnt/server14_hard1/msson/datasets/zs3_datasets --batch_size 2 --exp_name lseg_pascal_zs_compare_tar_oth --base_lr ${base_lr} --weight_decay 1e-4 --no-scaleinv --max_epochs 5 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384 --gpus 4 --no_resume --fold ${fold} --version ${fold} --seed ${seed}  --logpath /mnt/server14_hard0/msson/lang-seg/logs/compare_lr${base_lr}/tar_oth_fold${fold}_sofa --ckpt_save_path /mnt/server14_hard0/msson/lang-seg/

## target others proxy
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_lseg_zs_proxy.py --dataset pascal --data_path /mnt/server14_hard1/msson/datasets/zs3_datasets --batch_size 2 --exp_name lseg_pascal_zs_compare_tar_oth_pro --base_lr ${base_lr} --weight_decay 1e-4 --no-scaleinv --max_epochs 5 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384 --gpus 4 --no_resume --fold ${fold} --version ${fold} --seed ${seed}  --logpath /mnt/server14_hard0/msson/lang-seg/logs/compare_lr${base_lr}/tar_oth_pro_fold${fold} --ckpt_save_path /mnt/server14_hard0/msson/lang-seg/ --use_proxy

## target others prototype
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_lseg_zs_proxy.py --dataset pascal --data_path /mnt/server14_hard1/msson/datasets/zs3_datasets --batch_size 2 --exp_name lseg_pascal_zs_compare_tar_oth_proto --base_lr ${base_lr} --weight_decay 1e-4 --no-scaleinv --max_epochs 5 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384 --gpus 4 --no_resume --fold ${fold} --version ${fold} --seed ${seed}  --logpath /mnt/server14_hard0/msson/lang-seg/logs/compare_lr${base_lr}/tar_oth_fold${fold}_proto --ckpt_save_path /mnt/server14_hard0/msson/lang-seg/

## target others MASK
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_lseg_zs_mask.py --logpath /mnt/server14_hard0/msson/lang-seg/logs/compare_lr${base_lr}/mask_v${version}_fold${fold} --exp_name lseg_pascal_zs_mask_v${version} --ckpt_save_path /mnt/server14_hard0/msson/lang-seg/ --dataset pascal --data_path /mnt/server14_hard1/msson/datasets/zs3_datasets --batch_size ${batch_size} --base_lr ${base_lr} --weight_decay 1e-4 --no-scaleinv --max_epochs 5 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384 --gpus 4 --no_resume --fold ${fold} --version ${fold} --seed ${seed}

