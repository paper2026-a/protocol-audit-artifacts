# SignCV — protocol-aware structural compatibility audit

Artifacts for anonymous review of coordinate-level sign-consensus measurements across multiple LoRA objectives.

## Contents

- `scripts/signcv_recheck_safe.py`: structural measurement and reference-normalization pipeline.
- `scripts/signcv_selftest.py`: numerical checks and synthetic tests.
- `scripts/verify_release.py`: consistency checks for the released result summaries.
- `scripts/rerun_4run.sh`: CPU rerun command using separately obtained adapter weights.
- `results/*_4run.json`: per-model results, including 27-state sign histograms and input manifests.
- `results/table_4run.csv`: summary of the four-run structural measurements.

## Check results

Requires Python 3.9+:

```bash
python3 scripts/verify_release.py
```

Optional numerical tests:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/signcv_selftest.py --recheck scripts/signcv_recheck_safe.py
```

## Rerun the structural audit

For each model, place 12 LoRA adapter folders under a common root: three objectives (`race`, `gender`, `ses`), two learning rates (`0.0001`, `0.0002`), and seeds `42` and `43`. Each folder contains `adapter_model.safetensors` and `adapter_config.json`.

```bash
bash scripts/rerun_4run.sh qwen /path/to/ffinal_qwen
bash scripts/rerun_4run.sh llama3 /path/to/ffinal_llama
bash scripts/rerun_4run.sh mistral /path/to/ffinal
```

The four-run within-objective rule is 3-of-4. Cross-objective results are reported for 2-of-3 and 3-of-3 rules, each with its own analytic independence reference. The input-weight SHA256 records for Qwen2.5, Llama-3, and Mistral are included in their JSON manifests.

## Provenance and scope

The result summaries and per-tensor histograms are unchanged apart from anonymizing machine-specific input paths. The release supports independent verification of the four-run **structural calculations**; raw adapter weights, eight-run inputs/results, training pipelines, and downstream behavioral evaluation are not included. The historical Llama-2 four-run recheck used fallback LoRA configuration values because the original adapter configuration files were unavailable; its raw input identity has not been independently established.
