#!/usr/bin/env python
"""
SignCV structural audit: recalculate sign-consensus statistics from LoRA adapters.

Expected layout: ROOT/<objective>_lr<learning_rate>_seed<seed>/
    adapter_model.safetensors
    adapter_config.json

For each objective, count within-objective sign votes across repeated runs
and aggregate dense LoRA deltas (optionally normalized per tensor). The
cross-objective stage reports both unanimity (3-of-3) and majority (2-of-3),
plus protocol-matched analytic independence references and 27-state histograms.
Both eps=0 and eps=1e-10 counting conventions are recorded separately.
The quorum is an integer count (3-of-4 or 6-of-8), not a floating threshold.

Example (4 runs per objective):
    python3 scripts/signcv_recheck_safe.py --root PATH --seeds 42 43 \
      --device cpu --within-agg l2mean --out recomputed/model_4run.json

Resource note: tensors are processed sequentially but peak memory usage is
not bounded. On a shared server, request CPU access and limit BLAS threads.
Dependencies: torch and safetensors.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
from safetensors import safe_open

SUFFIX_A = ".lora_A.weight"
SUFFIX_B = ".lora_B.weight"
DEFAULT_Q_WITHIN = {4: 3, 8: 6}  # Protocol quorums: 3-of-4, 6-of-8


def load_cfg(run_dir):
    with open(os.path.join(run_dir, "adapter_config.json"), encoding="utf-8") as f:
        return json.load(f)


def lora_scale(cfg):
    r, alpha = cfg["r"], cfg["lora_alpha"]
    if cfg.get("use_rslora"):
        return alpha / math.sqrt(r)
    return alpha / r


def module_prefixes(handle):
    return sorted(k[: -len(SUFFIX_A)] for k in handle.keys() if k.endswith(SUFFIX_A))


def pct(x):
    return f"{100.0 * x:8.4f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Model adapter root directory")
    ap.add_argument("--axes", nargs="+", default=["race", "gender", "ses"])
    ap.add_argument("--lrs", nargs="+", default=["0.0001", "0.0002"], help="Values exactly as written in adapter folder names")
    ap.add_argument("--seeds", nargs="+", default=["42", "43"])
    ap.add_argument("--q-within", type=int, default=None, help="Defaults: n=4 -> 3, n=8 -> 6")
    ap.add_argument("--device", default="cpu", help="Compute device (default: CPU to limit shared GPU use)")
    ap.add_argument("--hash-inputs", action="store_true", help="Store SHA256 hashes of input safetensors (extra I/O)")
    ap.add_argument("--reported-scross", type=float, default=None,
                    help="Historical reported S_cross (percent); compare under both rules")
    ap.add_argument("--within-agg", choices=["l2mean", "rawmean"], default="l2mean",
                    help="Within aggregation: l2mean = mean of per-tensor L2-normalized updates; "
                         "rawmean = mean without prior per-tensor normalization. "
                         "Within masks match; aggregate signs may differ slightly")
    ap.add_argument("--out", default=None, help="Path to output JSON summary")
    ap.add_argument("--limit-tensors", type=int, default=0,
                    help="Smoke test: process first N tensors (0=all). Do NOT cite partial-tensor ratios")
    args = ap.parse_args()

    runs = [(lr, s) for lr in args.lrs for s in args.seeds]
    n = len(runs)
    m = len(args.axes)
    if m != 3:
        sys.exit("The 2-of-3 reference assumes exactly three objectives (--axes)")
    q_within = args.q_within if args.q_within is not None else DEFAULT_Q_WITHIN.get(n)
    if q_within is None:
        sys.exit(f"No default q_within for n={n}; specify --q-within")
    q_unan = m           # 3-of-3
    q_major = m // 2 + 1  # 2-of-3

    print(f"[protocol] n={n} runs/objective (lrs={args.lrs}, seeds={args.seeds}), "
          f"within rule = {q_within}-of-{n}, cross rules = {q_unan}-of-{m} and {q_major}-of-{m}")

    dev = torch.device(args.device)
    handles, scales = {}, {}
    manifest = []
    expected_cfg = None
    for ax in args.axes:
        for lr, s in runs:
            run_dir = os.path.join(args.root, f"{ax}_lr{lr}_seed{s}")
            path = os.path.join(run_dir, "adapter_model.safetensors")
            if not os.path.exists(path):
                sys.exit(f"missing: {path}")
            cfg = load_cfg(run_dir)
            fields = ("base_model_name_or_path", "revision", "r", "lora_alpha",
                      "use_rslora", "peft_type", "target_modules")
            signature = {key: cfg.get(key) for key in fields}
            if isinstance(signature.get("target_modules"), list):
                signature["target_modules"] = sorted(signature["target_modules"])
            if expected_cfg is None:
                expected_cfg = signature
            elif signature != expected_cfg:
                sys.exit(f"Adapter config mismatch in {run_dir}: {signature} != {expected_cfg}")
            record = {"axis": ax, "learning_rate": lr, "seed": s,
                      "file": os.path.abspath(path), "size_bytes": os.path.getsize(path),
                      "config": signature}
            if args.hash_inputs:
                sha = hashlib.sha256()
                with open(path, "rb") as file:
                    for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                        sha.update(block)
                record["sha256"] = sha.hexdigest()
            manifest.append(record)
            handles[(ax, lr, s)] = safe_open(path, framework="pt", device="cpu")
            scales[(ax, lr, s)] = lora_scale(cfg)
    base = expected_cfg.get("base_model_name_or_path")
    print(f"[adapters] base={base} r={cfg['r']} alpha={cfg['lora_alpha']} "
          f"scale={lora_scale(cfg):.4f} targets={cfg.get('target_modules')}")

    prefixes = module_prefixes(next(iter(handles.values())))
    for key, h in handles.items():
        if module_prefixes(h) != prefixes:
            sys.exit(f"adapter {key} targets different modules than the others")
    print(f"[adapters] {len(handles)} adapters, {len(prefixes)} LoRA-targeted tensors each")
    if args.limit_tensors > 0:
        prefixes = prefixes[: args.limit_tensors]
        print(f"[smoke test] processing only the first {len(prefixes)} tensors — ratios below are NOT paper numbers")

    EPS_COUNT = 1e-10          # survivor_log / protocol_audit magnitude floor
    d_total = 0
    within_mask = {ax: 0 for ax in args.axes}      # coords passing the within vote
    within_nnz = {ax: 0 for ax in args.axes}       # coords with |aggregate| > 1e-10 (survivor_log convention)
    surv_unan = surv_major = joint = 0             # cross mask counts, votes on sign(agg) with eps = 0
    surv_unan_e = surv_major_e = 0                 # cross mask counts, votes with |agg| > 1e-10 (protocol_audit)
    nnz_unan_norm = nnz_major_norm = 0             # |Delta*| > 1e-10, Delta* = keep * mean of L2-normalised axes (bias_vector)
    nnz_unan_raw = nnz_major_raw = 0               # |Delta*| > 1e-10, Delta* = keep * raw mean of axes (historical ffinal/signcv.py)
    hist27 = torch.zeros(27, dtype=torch.int64)   # primary eps=0 histogram, matches cross mask
    hist27_e = torch.zeros(27, dtype=torch.int64) # supplementary eps=1e-10 histogram
    hist27_per_tensor = {}
    hist27_per_tensor_e = {}
    eps = 1e-12
    t0 = time.time()

    with torch.no_grad():
        for i, pre in enumerate(prefixes):
            aggs = []
            for ax in args.axes:
                acc = cpos = cneg = None
                for lr, s in runs:
                    h = handles[(ax, lr, s)]
                    A = h.get_tensor(pre + SUFFIX_A).to(dev, torch.float32)
                    B = h.get_tensor(pre + SUFFIX_B).to(dev, torch.float32)
                    dW = scales[(ax, lr, s)] * (B @ A)                 # dense update for this tensor
                    pos = (dW > 0).to(torch.int8)
                    neg = (dW < 0).to(torch.int8)
                    cpos = pos if cpos is None else cpos + pos
                    cneg = neg if cneg is None else cneg + neg
                    if args.within_agg == "l2mean":
                        dWn = dW / torch.clamp(dW.norm(), min=eps)     # positive L2 normalization (per tensor)
                    else:
                        dWn = dW                                        # raw mean, as in lowmem / protocol_audit
                    acc = dWn if acc is None else acc + dWn
                    del A, B, dW, pos, neg, dWn
                mask = torch.maximum(cpos, cneg) >= q_within            # integer vote, no threshold arithmetic
                agg = (acc / n) * mask
                within_mask[ax] += int(mask.sum().item())
                within_nnz[ax] += int((agg.abs() > EPS_COUNT).sum().item())
                aggs.append(agg)
                del acc, cpos, cneg, mask

            T = torch.stack(aggs)                                       # (m, out, in) axis aggregates
            S0 = torch.sign(T).to(torch.int8)                           # eps = 0 votes (bias_vector / historical)
            Se = ((T > EPS_COUNT).to(torch.int8) - (T < -EPS_COUNT).to(torch.int8))  # eps = 1e-10 votes (protocol_audit)

            mx0 = torch.maximum((S0 > 0).sum(0), (S0 < 0).sum(0))
            keep_unan = mx0 >= q_unan
            keep_major = mx0 >= q_major
            surv_unan += int(keep_unan.sum().item())
            surv_major += int(keep_major.sum().item())
            joint += int((S0 != 0).all(0).sum().item())

            mxe = torch.maximum((Se > 0).sum(0), (Se < 0).sum(0))
            surv_unan_e += int((mxe >= q_unan).sum().item())
            surv_major_e += int((mxe >= q_major).sum().item())

            norms = T.reshape(m, -1).norm(dim=1).clamp_min(eps).view(m, *([1] * (T.ndim - 1)))
            mean_norm = (T / norms).mean(0)
            mean_raw = T.mean(0)
            nnz_unan_norm += int(((mean_norm.abs() > EPS_COUNT) & keep_unan).sum().item())
            nnz_major_norm += int(((mean_norm.abs() > EPS_COUNT) & keep_major).sum().item())
            nnz_unan_raw += int(((mean_raw.abs() > EPS_COUNT) & keep_unan).sum().item())
            nnz_major_raw += int(((mean_raw.abs() > EPS_COUNT) & keep_major).sum().item())

            idx0 = ((S0[0].to(torch.int64) + 1) * 9 + (S0[1].to(torch.int64) + 1) * 3 + (S0[2].to(torch.int64) + 1)).reshape(-1)
            idxe = ((Se[0].to(torch.int64) + 1) * 9 + (Se[1].to(torch.int64) + 1) * 3 + (Se[2].to(torch.int64) + 1)).reshape(-1)
            hb0 = torch.bincount(idx0, minlength=27).cpu()
            hbe = torch.bincount(idxe, minlength=27).cpu()
            hist27 += hb0
            hist27_e += hbe
            hist27_per_tensor[pre] = hb0.tolist()
            hist27_per_tensor_e[pre] = hbe.tolist()

            d_total += T[0].numel()
            del T, S0, Se, mx0, mxe, keep_unan, keep_major, norms, mean_norm, mean_raw, idx0, idxe, hb0, hbe, aggs

            if (i + 1) % 28 == 0 or i + 1 == len(prefixes):
                print(f"  {i + 1:4d}/{len(prefixes)} tensors, {d_total / 1e9:.3f}B coords, {time.time() - t0:6.1f}s")

    within = within_mask
    s = [within[ax] / d_total for ax in args.axes]
    prod = math.prod(s)
    s1, s2, s3 = s
    P33 = prod / 4.0
    P23 = prod + 0.5 * (s1 * s2 * (1 - s3) + s1 * s3 * (1 - s2) + s2 * s3 * (1 - s1))
    S33 = surv_unan / d_total
    S23 = surv_major / d_total
    J = joint / d_total
    U = S33 / J if J > 0 else float("nan")
    R33 = S33 / P33
    R23 = S23 / P23
    R_support = J / prod
    R_sign = U / 0.25

    print("\n================ SignCV re-check ================")
    print(f"coordinates analysed d = {d_total:,}")
    for ax, sa in zip(args.axes, s):
        print(f"S_within[{ax:>6}] = {pct(sa)}")
    print(f"mean S_within       = {pct(sum(s) / m)}   (independent-sign ref for {q_within}-of-{n}: "
          f"{pct(2 * sum(math.comb(n, k) for k in range(q_within, n + 1)) / 2 ** n)})")
    print("--- cross-objective, same learned updates, same within masks ---")
    print(f"S_cross({q_unan}-of-{m})     = {pct(S33)}   P_ref = {pct(P33)}   R_cross = {R33:.3f}")
    print(f"S_cross({q_major}-of-{m})     = {pct(S23)}   P_ref = {pct(P23)}   R_cross = {R23:.3f}")
    print(f"J (joint support)   = {pct(J)}   R_support = {R_support:.3f}")
    print(f"U (unanimity | J)   = {pct(U)}   R_sign    = {R_sign:.3f}   (R_support*R_sign = {R_support * R_sign:.3f})")
    print("--- counting-convention check (same tensors, same masks) ---")
    for ax in args.axes:
        print(f"within[{ax:>6}]  mask count = {within_mask[ax]:>14,}   |agg|>1e-10 count = {within_nnz[ax]:>14,}   "
              f"diff = {within_mask[ax] - within_nnz[ax]:,}")
    print(f"cross {q_unan}-of-{m}: mask(eps=0) = {surv_unan:,}  mask(eps=1e-10) = {surv_unan_e:,}  "
          f"nnz(Delta*, normalised mean) = {nnz_unan_norm:,}  nnz(Delta*, raw mean) = {nnz_unan_raw:,}")
    print(f"cross {q_major}-of-{m}: mask(eps=0) = {surv_major:,}  mask(eps=1e-10) = {surv_major_e:,}  "
          f"nnz(Delta*, normalised mean) = {nnz_major_norm:,}  nnz(Delta*, raw mean) = {nnz_major_raw:,}")
    # Recount voting masks from the 27-state histogram using the same (eps=0) state definition.
    def recount(hist, q):
        total = 0
        for k, count in enumerate(hist):
            states = (k // 9 - 1, (k // 3) % 3 - 1, k % 3 - 1)
            if max(sum(v > 0 for v in states), sum(v < 0 for v in states)) >= q:
                total += int(count)
        return total

    if (recount(hist27, q_unan) != surv_unan or recount(hist27, q_major) != surv_major
            or recount(hist27_e, q_unan) != surv_unan_e or recount(hist27_e, q_major) != surv_major_e):
        sys.exit("ERROR: 27-state histogram does not match cross mask counts; aborting")
    print("[validated] both eps=0 and eps=1e-10 histograms exactly reproduce their cross mask counts")
    print("--- 27-state sign histogram (race, gender, ses), eps = 0 (primary mask) ---")
    labels = {-1: "-", 0: "0", 1: "+"}
    for k in range(27):
        a, b, c = k // 9 - 1, (k // 3) % 3 - 1, k % 3 - 1
        if hist27[k] > 0:
            print(f"  {labels[a]}{labels[b]}{labels[c]}  {int(hist27[k]):>14,}  ({100.0 * int(hist27[k]) / d_total:7.4f}%)")

    verdict = None
    if args.reported_scross is not None:
        rep = args.reported_scross / 100.0
        print("--- comparison with the reported value ---")
        print(f"reported S_cross    = {pct(rep)}")
        print(f"reported / P_ref({q_unan}-of-{m}) = {rep / P33:.3f}    reported / P_ref({q_major}-of-{m}) = {rep / P23:.3f}")
        print(f"reported / recomputed S_cross({q_unan}-of-{m}) = {rep / S33:.3f}    "
              f"reported / recomputed S_cross({q_major}-of-{m}) = {rep / S23:.3f}")
        diff_unan_pp = abs(rep - S33) * 100.0
        diff_major_pp = abs(rep - S23) * 100.0
        closest = (f"{q_major}-of-{m} majority" if diff_major_pp < diff_unan_pp
                   else f"{q_unan}-of-{m} unanimity")
        verdict = (f"Closest reconstructed measurement: {closest}; "
                   f"distance to unanimity={diff_unan_pp:.6f} pp, "
                   f"distance to majority={diff_major_pp:.6f} pp. "
                   "Closeness is NOT proof of historical provenance or reproduction.")
        print(f"COMPARISON: {verdict}")

    summary = {
        "root": args.root, "base_model": base, "n_runs": n, "lrs": args.lrs, "seeds": args.seeds,
        "q_within": q_within, "within_agg": args.within_agg, "d": d_total,
        "input_manifest": manifest,
        "S_within": dict(zip(args.axes, s)),
        "S_cross_unanimity": S33, "P_ref_unanimity": P33, "R_cross_unanimity": R33,
        "S_cross_majority": S23, "P_ref_majority": P23, "R_cross_majority": R23,
        "J": J, "U": U, "R_support": R_support, "R_sign": R_sign,
        "counts": {
            "within_mask": within_mask, "within_nnz_1e-10": within_nnz,
            "cross_mask_eps0": {"unanimity": surv_unan, "majority": surv_major},
            "cross_mask_eps1e-10": {"unanimity": surv_unan_e, "majority": surv_major_e},
            "cross_nnz_deltastar_normalised_mean": {"unanimity": nnz_unan_norm, "majority": nnz_major_norm},
            "cross_nnz_deltastar_raw_mean": {"unanimity": nnz_unan_raw, "majority": nnz_major_raw},
            "joint_nonzero": joint,
        },
        "hist27_total": hist27.tolist(),  # primary eps=0 (matches S_cross mask counts)
        "hist27_eps1e-10_total": hist27_e.tolist(),
        "hist27_index": "idx=(s_race+1)*9+(s_gender+1)*3+(s_ses+1), states in {-1,0,+1}",
        "hist27_primary_epsilon": 0.0,
        "hist27_per_tensor": hist27_per_tensor,
        "hist27_eps1e-10_per_tensor": hist27_per_tensor_e,
        "reported_scross": args.reported_scross, "verdict": verdict,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
