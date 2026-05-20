#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LABEL="${1:-}"
if [[ -z "$LABEL" ]]; then
  echo "Usage: $0 <B16|B16_0.8|B8|B8_0.8|B5|B5_0.8|B1|B1_0.7>"
  exit 1
fi

case "$LABEL" in
  B16) MODEL_PATH="/homes/kasram/broteina/PaperModels/broteina_16.pkl"; NSTEP=16; NOISE_SCALE=1.0 ;;
  B16_0.8) MODEL_PATH="/homes/kasram/broteina/PaperModels/broteina_16.pkl"; NSTEP=16; NOISE_SCALE=0.8 ;;
  B8) MODEL_PATH="/homes/kasram/broteina/PaperModels/broteina_8.pkl"; NSTEP=8; NOISE_SCALE=1.0 ;;
  B8_0.8) MODEL_PATH="/homes/kasram/broteina/PaperModels/broteina_8.pkl"; NSTEP=8; NOISE_SCALE=0.8 ;;
  B5) MODEL_PATH="/homes/kasram/broteina/PaperModels/broteina_5.pkl"; NSTEP=5; NOISE_SCALE=1.0 ;;
  B5_0.8) MODEL_PATH="/homes/kasram/broteina/PaperModels/broteina_5.pkl"; NSTEP=5; NOISE_SCALE=0.8 ;;
  B1) MODEL_PATH="/homes/kasram/broteina/PaperModels/broteina_1.pkl"; NSTEP=1; NOISE_SCALE=1.0 ;;
  B1_0.7) MODEL_PATH="/homes/kasram/broteina/PaperModels/broteina_1.pkl"; NSTEP=1; NOISE_SCALE=0.7 ;;
  *)
    echo "Unknown LABEL=$LABEL"
    exit 1
    ;;
esac

SESSION_PREFIX="${SESSION_PREFIX:-paper_dist_${LABEL}}"
GPUS_STR="${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_ARR <<< "$GPUS_STR"
NUM_SPLITS="${NUM_SPLITS:-8}"
if [[ "${#GPU_ARR[@]}" -ne "$NUM_SPLITS" ]]; then
  echo "Expected ${NUM_SPLITS} GPUs, got ${#GPU_ARR[@]} from GPUS='$GPUS_STR'"
  exit 1
fi

OUT_ROOT="${OUT_ROOT:-/homes/kasram/broteina/SiD_Protein/evaluation/distributional/neurips_selfd/${LABEL}}"
LOG_DIR="${LOG_DIR:-/homes/kasram/broteina/SiD_Protein/logs/paper_distributional/${LABEL}}"
SEED="${SEED:-5}"
SEED_STRIDE="${SEED_STRIDE:-1000003}"

mkdir -p "$LOG_DIR"
mkdir -p "$OUT_ROOT"

for split_id in $(seq 0 $((NUM_SPLITS - 1))); do
  gpu="${GPU_ARR[$split_id]}"
  session="${SESSION_PREFIX}_g${gpu}"
  log_path="${LOG_DIR}/split$(printf '%02d' "$split_id").log"

  tmux kill-session -t "$session" >/dev/null 2>&1 || true
  tmux new-session -d -s "$session" \
    "cd /homes/kasram/broteina/SiD_Protein && \
     source \"\$(conda info --base)/etc/profile.d/conda.sh\" && \
     conda activate sid_protein_env && \
     export CUDA_VISIBLE_DEVICES=${gpu} && \
     PYTHONUNBUFFERED=1 python generate_distilled_proteina_distributional.py \
       --model_path '${MODEL_PATH}' \
       --out_dir '${OUT_ROOT}' \
       --nstep ${NSTEP} \
       --noise_scale ${NOISE_SCALE} \
       --seed ${SEED} \
       --split_id ${split_id} \
       --num_splits ${NUM_SPLITS} \
       --seed_stride ${SEED_STRIDE} \
       > '${log_path}' 2>&1"
done

echo "Launched ${NUM_SPLITS} tmux sessions for ${LABEL}"
echo "Output: ${OUT_ROOT}"
echo "Logs:   ${LOG_DIR}"
