#!/usr/bin/env bash
# Example: bash scripts/rerun_4run.sh qwen /path/to/adapters/ffinal_qwen
# Re-running requires 12 weight files + adapter_config.json in each model folder.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: bash scripts/rerun_4run.sh {qwen|llama3|mistral|llama2} ROOT" >&2; exit 2; }
model=$1; root=$2
case "$model" in qwen|llama3|mistral|llama2) ;; *) echo "Invalid model" >&2; exit 2 ;; esac
[[ -d "$root" ]] || { echo "Adapter root not found" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
out="recomputed/${model}_4run.json"
mkdir -p recomputed
python3 signcv_recheck_safe.py --root "$root" --seeds 42 43 --device cpu --within-agg l2mean --hash-inputs --out "$out"
python3 verify_release.py --allow-absolute-paths "$out"
