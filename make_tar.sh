#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 40
#SBATCH --array 0-7
#SBATCH --output=make_tar_%A_%a.out

tar -cvf - ./protein_out/${SLURM_ARRAY_TASK_ID} | pigz -p 40 -k > dataset_${SLURM_ARRAY_TASK_ID}.tar.gz
