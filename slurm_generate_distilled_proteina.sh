#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:volta:1
#SBATCH --array 0-7
#SBATCH --output=gen_job_%A_%a.out

source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/etc/profile.d/conda.sh
conda activate ../sid_protein_env/

echo "Running task ${SLURM_ARRAY_TASK_ID} on node ${SLURMD_NODENAME} with GPU ${CUDA_VISIBLE_DEVICES}..."

python generate_distilled_proteina.py \
--model_path "checkpoints/8-step-checkpoint.pt" \
--out_dir protein_out \
--lengths 50,100,150,200,250 \
--num_batch 5000 \
--batch_size 20 \
--nstep 8 \
--noise_scale 0.45 \
--seed ${SLURM_ARRAY_TASK_ID} \
# --conditional # Uncomment this line to perform conditional generation
