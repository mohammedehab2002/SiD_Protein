#!/usr/bin/env bash
# Run SiD_Protein/compute_stats.py for each paper checkpoint.
# Outputs per-checkpoint logs to ./paper_stats_logs/<label>.log and prints a summary.

set -u
cd "$(dirname "$0")"

LOG_DIR="paper_stats_logs"
mkdir -p "$LOG_DIR"

# label : ckpt_path (relative to design_eval_neurips/)
declare -a RUNS=(
    "B1=neurips_selfd/1/network-snapshot-1.000000-005561.pkl"
    "B1_0.7=neurips_selfd/1/network-snapshot-1.000000-005561.pkl_sc_0.7"
    "B5=neurips_selfd/5/5step_network-snapshot-1.000000-000626.pkl"
    "B5_0.8=neurips_selfd/5/5step_network-snapshot-1.000000-000626.pkl_sc_0.8"
    "B8=neurips_selfd/8/network-snapshot-1.000000-000462.pkl"
    "B8_0.8=neurips_selfd/8/network-snapshot-1.000000-000462.pkl_sc_0.8"
    "B16=neurips_selfd/16/network-snapshot-1.000000-000069.pkl"
    "B16_0.8=neurips_selfd/16/network-snapshot-1.000000-000069.pkl_sc_0.8"
)

for entry in "${RUNS[@]}"; do
    label="${entry%%=*}"
    ckpt="${entry#*=}"
    log="$LOG_DIR/$label.log"

    echo "=========================================================="
    echo "[$label] ckpt=$ckpt"
    echo "        log=$log"
    echo "=========================================================="

    if [ ! -d "design_eval_neurips/$ckpt/pdbs/designable" ]; then
        echo "[$label] MISSING samples at design_eval_neurips/$ckpt/pdbs/designable -- skipping" | tee "$log"
        continue
    fi

    # Stream to console and tee to log; don't abort the whole script if one fails.
    if python compute_stats.py -c "$ckpt" 2>&1 | tee "$log"; then
        echo "[$label] done"
    else
        echo "[$label] FAILED (see $log)"
    fi
done

echo
echo "=========================================================="
echo "Summary"
echo "=========================================================="
for entry in "${RUNS[@]}"; do
    label="${entry%%=*}"
    log="$LOG_DIR/$label.log"
    echo "--- $label ---"
    if [ -f "$log" ]; then
        grep -E "^(Designability|Secondary Structure Content|TMScore by Length|Average TMScore|Diversity|PDB Novelty|AFDB Novelty):" "$log" || echo "(no metrics parsed)"
    else
        echo "(no log)"
    fi
done
