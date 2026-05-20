#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"

LABEL="${1:-}"
if [[ -z "$LABEL" ]]; then
  echo "Usage: $0 <B16|B16_0.8|B8|B8_0.8|B5|B5_0.8|B1|B1_0.7>"
  exit 1
fi

OUT_ROOT="/homes/kasram/broteina/SiD_Protein/evaluation/distributional/neurips_selfd/${LABEL}"
DATA_DIR="${OUT_ROOT}/samples_fid_sharded"
LOG_DIR="/homes/kasram/broteina/SiD_Protein/logs/paper_distributional/${LABEL}"
METRIC_LOG="${LOG_DIR}/metrics.log"

mkdir -p "$LOG_DIR"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing sample directory: $DATA_DIR"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sid_protein_env
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/training/proteina:${PYTHONPATH:-}"

python training/proteina/script_utils/inference_fid.py \
  --data_dir "$DATA_DIR" \
  --ca_only \
  --batch_size 12 \
  --num_workers 32 | tee "$METRIC_LOG"
