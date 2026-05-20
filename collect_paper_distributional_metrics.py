#!/usr/bin/env python3

from pathlib import Path
import ast
import csv


LABELS = ["B16", "B16_0.8", "B8", "B8_0.8", "B5", "B5_0.8", "B1", "B1_0.7"]
BASE_LOGS = Path("/homes/kasram/broteina/SiD_Protein/logs/paper_distributional")
SUMMARY_CSV = Path(
    "/homes/kasram/broteina/SiD_Protein/evaluation/distributional/neurips_selfd/"
    "paper_distributional_metrics_summary.csv"
)

FIELDNAMES = [
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


def parse_metrics_payload(payload: str):
    payload = payload.strip()
    if payload.startswith("OrderedDict(") and payload.endswith(")"):
        inner = payload[len("OrderedDict("):-1].strip()
        return dict(ast.literal_eval(inner))
    return ast.literal_eval(payload)


def main():
    rows = []
    missing = []
    for label in LABELS:
        log_path = BASE_LOGS / label / "metrics.log"
        text = log_path.read_text() if log_path.exists() else ""
        marker = "Paper-style distributional metrics:"
        idx = text.rfind(marker)
        if idx < 0:
            missing.append(label)
            continue
        payload = text[idx + len(marker):].strip()
        metrics = parse_metrics_payload(payload)
        row = {"label": label}
        for key in FIELDNAMES[1:]:
            row[key] = metrics.get(key)
        rows.append(row)

    if missing:
        raise RuntimeError(
            "Missing paper-style metrics block for labels: " + ", ".join(missing)
        )

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
