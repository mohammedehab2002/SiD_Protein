#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

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

EXPECTED_COUNT=5000
POLL_SECONDS="${POLL_SECONDS:-30}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
NUM_SPLITS="${NUM_SPLITS:-8}"

wait_for_label() {
  local label="$1"
  local session_prefix="${SESSION_PREFIX_BASE:-paper_dist}_${label}"
  local out_root="/homes/kasram/broteina/SiD_Protein/evaluation/distributional/neurips_selfd/${label}"
  local sample_dir="${out_root}/samples_fid_sharded"

  while true; do
    local tmux_alive
    tmux_alive="$(tmux ls 2>/dev/null | grep -c "^${session_prefix}_g" || true)"
    local pdb_count=0
    if [[ -d "$sample_dir" ]]; then
      pdb_count="$(find "$sample_dir" -maxdepth 1 -type f -name '*.pdb' | wc -l)"
    fi

    printf '[%s] tmux=%s pdbs=%s/%s\n' "$label" "$tmux_alive" "$pdb_count" "$EXPECTED_COUNT"

    if [[ "$tmux_alive" -eq 0 ]]; then
      if [[ "$pdb_count" -eq "$EXPECTED_COUNT" ]]; then
        return 0
      fi
      echo "Generation for ${label} stopped early: found ${pdb_count}/${EXPECTED_COUNT} pdbs"
      return 1
    fi

    sleep "$POLL_SECONDS"
  done
}

for label in "${LABELS[@]}"; do
  echo "Launching ${label}"
  GPUS="$GPUS" NUM_SPLITS="$NUM_SPLITS" SESSION_PREFIX="${SESSION_PREFIX_BASE:-paper_dist}_${label}" \
    bash run_paper_distributional_generation.sh "$label"
  wait_for_label "$label"
done

echo "All distributional generations finished."
