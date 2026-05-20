#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -n "${LABELS_OVERRIDE:-}" ]]; then
  read -r -a LABELS <<< "${LABELS_OVERRIDE}"
else
  LABELS=(
    B16
    B16_0.8
    B8
    B8_0.8
    B5
    B5_0.8
    B1
    B1_0.7
  )
fi

GPUS_STR="${GPUS:-0}"
read -r -a GPU_ARR <<< "$GPUS_STR"
SUMMARY_DIR="/homes/kasram/broteina/SiD_Protein/evaluation/distributional/neurips_selfd"
SUMMARY_CSV="${SUMMARY_DIR}/paper_distributional_metrics_summary.csv"

mkdir -p "$SUMMARY_DIR"

run_one() {
  local label="$1"
  local gpu="$2"
  echo "Running ${label} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" bash run_paper_distributional_metrics.sh "${label}"
}

is_complete() {
  local label="$1"
  local log_path="/homes/kasram/broteina/SiD_Protein/logs/paper_distributional/${label}/metrics.log"
  [[ -f "$log_path" ]] && grep -q 'Paper-style distributional metrics:' "$log_path"
}

idx=0
while [[ "$idx" -lt "${#LABELS[@]}" ]]; do
  pids=()
  batch_labels=()
  for gpu in "${GPU_ARR[@]}"; do
    if [[ "$idx" -ge "${#LABELS[@]}" ]]; then
      break
    fi
    label="${LABELS[$idx]}"
    if is_complete "$label"; then
      echo "Skipping completed ${label}"
      idx=$((idx + 1))
      continue
    fi
    run_one "$label" "$gpu" &
    pids+=("$!")
    batch_labels+=("$label")
    idx=$((idx + 1))
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  if [[ "${#batch_labels[@]}" -gt 0 ]]; then
    printf 'Finished batch: %s\n' "${batch_labels[*]}"
  fi
done

python - <<'PY'
from pathlib import Path
import ast
import csv

labels = ["B16","B16_0.8","B8","B8_0.8","B5","B5_0.8","B1","B1_0.7"]
summary_csv = Path("/homes/kasram/broteina/SiD_Protein/evaluation/distributional/neurips_selfd/paper_distributional_metrics_summary.csv")
base_logs = Path("/homes/kasram/broteina/SiD_Protein/logs/paper_distributional")

fieldnames = [
    "label",
    "FPSD_vs_PDB",
    "FPSD_vs_AFDB",
    "fS_C",
    "fS_A",
    "fS_T",
    "fJSD_vs_PDB",
    "fJSD_vs_AFDB",
    "fJSD_C_vs_PDB",
    "fJSD_A_vs_PDB",
    "fJSD_T_vs_PDB",
    "fJSD_C_vs_AFDB",
    "fJSD_A_vs_AFDB",
    "fJSD_T_vs_AFDB",
]

rows = []
for label in labels:
    log_path = base_logs / label / "metrics.log"
    text = log_path.read_text()
    marker = "Paper-style distributional metrics:"
    idx = text.rfind(marker)
    if idx < 0:
        raise RuntimeError(f"Missing paper-style metrics block in {log_path}")
    payload = text[idx + len(marker):].strip()
    metrics = ast.literal_eval(payload)
    row = {"label": label}
    for key in fieldnames[1:]:
        row[key] = metrics.get(key)
    rows.append(row)

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(summary_csv)
PY

echo "Finished all distributional metric evaluations."
echo "Summary: ${SUMMARY_CSV}"
