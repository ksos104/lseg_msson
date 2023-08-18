#!/bin/bash
#SBATCH -J zss
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o slurm_logs/stdout_%j.txt
#SBATCH -e slurm_logs/stderr_%j.txt
#SBATCH --gres=gpu:4


#python -u train_lseg.py --dataset ade20k --data_path ../datasets --batch_size 4 --exp_name lseg_ade20k_l16 \
#--base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 240 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_lseg_zs.py --dataset pascal --data_path /mnt/server14_hard1/msson/datasets/zs3_datasets --batch_size 2 --exp_name lseg_pascal_zs --base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 3 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384 --gpus 4 --no_resume --fold 0 --version 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_lseg_zs_try.py --dataset pascal --data_path /mnt/server14_hard1/msson/datasets/zs3_datasets --batch_size 2 --exp_name lseg_pascal_zs_proxy5_kl --base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 3 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384 --gpus 4 --no_resume --fold 0 --version 0