#!/bin/bash

#SBATCH -p sched_mit_sloan_gpu_r8
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:a100:1
#SBATCH --array 0-3
#SBATCH --output=logs/gen_job_%a.out
#SBATCH --mem 64G

conda init
conda activate sid_protein_env

echo "Running task ${SLURM_ARRAY_TASK_ID} on node ${SLURMD_NODENAME} with GPU ${CUDA_VISIBLE_DEVICES}..."

python generate_distilled_proteina_designability.py \
--model_path "checkpoints/network-snapshot-1.000000-001019.pkl" \
--out_dir protein_out \
--lengths 50,100,150,200,250 \
--num_batch 1 \
--batch_size 25 \
--nstep 1 \
--noise_scale 1 \
--seed ${SLURM_ARRAY_TASK_ID} \
# --conditional # Uncomment this line to perform conditional generation
