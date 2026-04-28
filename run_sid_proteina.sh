#!/bin/bash

# Run the code below in command window to set CUDA visible devices and run specific script
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Decrease --batch-gpu to reduce memory consumption

# Command to run torch with specific parameters
# Add the option below to load a checkpoint:
# Many options are optional, such as --data_stat, which will be computed inside the code if not provided
# --resume 'protein_experiment/sid-train-runs/proteina_multistep/00000-Proteina_Uncond???/network-snapshot???.pkl'
torchrun --standalone --nproc_per_node=8 sid_train.py \
--alpha 1.0 \
--t_init 30 \
--t 400 \
--tmax 0.98 \
--batch 4096 \
--batch-gpu 4 \
--eval_batch 2 \
--outdir 'protein_experiment/sid-train-runs/proteina_multistep' \
--data 'pdb_raw/cath_label_mapping.pt' \
--arch proteina \
--precond proteina \
--metrics sctm \
--tick 5 \
--snap 2 \
--dump 12 \
--lr 1e-4 \
--glr 5e-5 \
--g_beta1 0.9 \
--fp16 1 \
--ls 1 \
--lsg 100 \
--duration 100 \
--use_sida false \
--config_path 'proteina/configs/experiment_config/' \
--config_name 'broteina_distillation' \
--noise_scale 1.0 \
--nstep 1 \
--min_n_res 30 \
--max_n_res 270 \

# --resume_pkl 'protein_experiment/sid-train-runs/proteina_multistep/00044-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/network-snapshot-1.000000-000692.pkl' \
# --resume 'protein_experiment/sid-train-runs/proteina_multistep/00044-uncond-proteina-glr5e-05-lr0.0001-initsigma2.5-gpus8-alpha1.0-batch4096-tmax0.98-fp16-nstep8/training-state-000692.pt' \