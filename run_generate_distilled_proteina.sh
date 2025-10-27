#!/bin/bash

if [ -z "$1" ]; then
  GPU_ID=0
  echo "No GPU specified. Defaulting to GPU 0."
else
  GPU_ID=$1
  echo "Using GPU $GPU_ID."
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

python generate_distilled_proteina.py \
--model_path "checkpoints/16-step-checkpoint.pt" \
--out_dir protein_out \
--lengths 50,100,150,200,250 \
--num_batch 10 \
--batch_size 10 \
--nstep 16 \
--noise_scale 0.45 \
--seed 5 \
# --conditional # Uncomment this line to perform conditional generation