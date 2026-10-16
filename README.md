# SignCV — Protocol-Aware Structural Compatibility Audit

Anonymous research artifacts for the four-run structural analysis of coordinate-level sign consensus across multiple LoRA objectives. All released files are located at the **repository root**.

## Files

- `signcv_recheck_safe.py` — sign-consensus measurements, protocol-matched reference values, and 27-state histograms.
- `signcv_selftest.py` — numerical checks and synthetic tests of the counting rules.
- `verify_release.py` — consistency checks for the four released result files.
- `rerun_4run.sh` — CPU rerun wrapper (requires separately obtained adapter weights).
- `llama2_4run.json`, `qwen_4run.json`, `llama3_4run.json`, `mistral_4run.json` — per-model four-run results and input manifests.
- `table_4run.csv` — summary table.
- `FILE_CHECKSUMS.sha256` — SHA256 checksums for the files in this repository.

## Verify the released results

With Python 3.9 or later, from the repository root:

```bash
python3 verify_release.py llama2_4run.json qwen_4run.json llama3_4run.json mistral_4run.json
```

Check the released file hashes (on Linux/macOS):

```bash
sha256sum -c FILE_CHECKSUMS.sha256
```

Optional numerical unit and synthetic-data tests:

```bash
python3 -m pip install -r requirements.txt
python3 signcv_selftest.py --recheck ./signcv_recheck_safe.py
```

## Rerun the four-run structural analysis

Obtain the original LoRA adapter weights separately. Each model requires 12 adapter directories: `race`, `gender`, and `ses`, each at learning rates `0.0001` and `0.0002`, and seeds `42` and `43`. Every directory must contain `adapter_model.safetensors` and `adapter_config.json`.

```bash
bash rerun_4run.sh qwen /path/to/ffinal_qwen
bash rerun_4run.sh llama3 /path/to/ffinal_llama
bash rerun_4run.sh mistral /path/to/ffinal
```

The four-run within-objective voting rule is 3-of-4. Cross-objective results are reported for 2-of-3 majority and 3-of-3 unanimity with corresponding, protocol-matched independence references. The JSON input manifests include weight SHA256 records for Qwen2.5, Llama-3, and Mistral.

## Reproducibility scope

The released JSON files include structural counts and per-tensor histograms for verifying the reported **four-run structural calculations**. Adapter weights, eight-run inputs/results, training pipelines, and downstream behavioral evaluation are not part of this release. The historical Llama-2 four-run recheck used fallback LoRA configuration values because the original adapter configuration files were unavailable; original weight identity for that run has not been independently authenticated.
