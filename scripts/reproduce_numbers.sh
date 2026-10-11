#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DIRS="scores/finetuned_A scores/finetuned_B scores/finetuned_K scores/zero_shot scores/public"
mkdir -p results
python3 scripts/analyze_det.py $DIRS results/analysis_det.json
python3 scripts/dev_replication.py scores/finetuned_A scores/finetuned_B scores/finetuned_K scores/zero_shot
python3 scripts/check_claims.py
