#!/usr/bin/env python3
"""
signcv_selftest.py — unit tests for the SignCV counting rules and analytic references.

Tests 1–3 need only numpy and run in seconds anywhere.
Test 4 runs end to end if torch + safetensors are installed: it writes tiny synthetic LoRA
adapters, runs signcv_recheck_safe.py on them, and compares every reported count with an
independent brute-force computation on the same dense updates.

    python signcv_selftest.py                      # tests 1-3 (+4 if torch is available)
    python signcv_selftest.py --recheck ./signcv_recheck_safe.py --skip-e2e

What is verified
  1. closed-form references  P_ref(3/3) = s1 s2 s3 / 4  and  P_ref(2/3) = ... equal an exact
     enumeration over support and sign states (same enumeration style as protocol_audit.py)
  2. on iid symmetric data the empirical mask survivor rates match the references
     (within 3-of-4 -> 0.625; cross 2-of-3 and 3-of-3 -> P_ref), so R ~ 1 when the rule
     and the reference are matched, and the mismatched ratio S_2/3 / P_3/3 is large
  3. counting semantics: mask count vs. nonzero count differ exactly on cancellation
     coordinates under majority rules (3-of-4 within, 2-of-3 cross) and never under 3-of-3
  4. (e2e) signcv_recheck_safe.py reproduces brute-force counts on synthetic adapters
"""
import argparse, itertools, json, math, os, shutil, subprocess, sys, tempfile
import numpy as np


# ----------------------------------------------------------------------------- references
def ref_within(n, q):
    return 2.0 * sum(math.comb(n, k) for k in range(q, n + 1)) / 2.0 ** n


def ref_cross_enum(s, q):
    """Exact reference by enumeration over support bits and iid symmetric signs."""
    m = len(s)
    p = 0.0
    for bits in itertools.product((0, 1), repeat=m):
        ps = 1.0
        active = []
        for i, b in enumerate(bits):
            ps *= s[i] if b else (1.0 - s[i])
            if b:
                active.append(i)
        if ps == 0 or len(active) < q:
            continue
        for signs in itertools.product((-1, 1), repeat=len(active)):
            pos = sum(x > 0 for x in signs)
            if max(pos, len(signs) - pos) >= q:
                p += ps * 0.5 ** len(active)
    return p


def P33(s):
    return s[0] * s[1] * s[2] / 4.0


def P23(s):
    a, b, c = s
    return a * b * c + 0.5 * (a * b * (1 - c) + a * c * (1 - b) + b * c * (1 - a))


# ----------------------------------------------------------------------------- counting (numpy)
EPS_COUNT = 1e-10


def within_stage(runs, q_within, l2mean=True):
    """runs: array (n, d). Returns mask, aggregate (masked), mask count, nonzero count."""
    n = runs.shape[0]
    cpos = (runs > 0).sum(0)
    cneg = (runs < 0).sum(0)
    mask = np.maximum(cpos, cneg) >= q_within
    if l2mean:
        norms = np.maximum(np.linalg.norm(runs, axis=1, keepdims=True), 1e-12)
        agg = (runs / norms).mean(0)
    else:
        agg = runs.mean(0)
    agg = agg * mask
    return mask, agg, int(mask.sum()), int((np.abs(agg) > EPS_COUNT).sum())


def cross_stage(aggs, q):
    """aggs: array (m, d) of axis aggregates. Returns mask count (eps=0 votes), nonzero count
    of Delta* under the normalised-mean and raw-mean conventions, and joint nonzero count."""
    S = np.sign(aggs)
    mx = np.maximum((S > 0).sum(0), (S < 0).sum(0))
    keep = mx >= q
    norms = np.maximum(np.linalg.norm(aggs, axis=1, keepdims=True), 1e-12)
    d_norm = (aggs / norms).mean(0)
    d_raw = aggs.mean(0)
    return (int(keep.sum()),
            int(((np.abs(d_norm) > EPS_COUNT) & keep).sum()),
            int(((np.abs(d_raw) > EPS_COUNT) & keep).sum()),
            int((S != 0).all(0).sum()))


# ----------------------------------------------------------------------------- tests
def test1_references():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        s = rng.uniform(0.05, 0.95, size=3)
        worst = max(worst, abs(ref_cross_enum(s, 3) - P33(s)), abs(ref_cross_enum(s, 2) - P23(s)))
    assert worst < 1e-12, worst
    assert abs(ref_within(4, 3) - 0.625) < 1e-12 and abs(ref_within(8, 6) - 0.2890625) < 1e-12
    print(f"[1] closed-form references == exact enumeration (max |diff| = {worst:.1e}); "
          f"P_within(4,3) = 0.625, P_within(8,6) = 0.2890625  OK")


def test2_iid_matched_vs_mismatched(d=400_000, n=4, q_within=3, seed=1):
    rng = np.random.default_rng(seed)
    aggs, s = [], []
    for _ in range(3):
        runs = rng.standard_normal((n, d))
        _, agg, mc, _ = within_stage(runs, q_within)
        aggs.append(agg)
        s.append(mc / d)
    aggs = np.stack(aggs)
    S3 = cross_stage(aggs, 3)[0] / d
    S2 = cross_stage(aggs, 2)[0] / d
    r_w = [x / 0.625 for x in s]
    R33, R23 = S3 / P33(s), S2 / P23(s)
    mism = S2 / P33(s)
    assert all(abs(x - 1) < 0.02 for x in r_w), r_w
    assert abs(R33 - 1) < 0.05 and abs(R23 - 1) < 0.02, (R33, R23)
    assert mism > 5, mism
    print(f"[2] iid data, d={d:,}: within/0.625 = {min(r_w):.3f}-{max(r_w):.3f}; "
          f"R(3-of-3) = {R33:.3f}, R(2-of-3) = {R23:.3f}; mismatched S_2/3 / P_3/3 = {mism:.2f}  OK")


def test3_cancellation_semantics():
    # within 3-of-4: (1,1,1,-3) passes the vote, raw mean is exactly 0 -> mask 1, nonzero 0
    runs = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [-3.0, 1.0, -1.0]])
    mask, agg, mc, nz = within_stage(runs, 3, l2mean=False)
    assert mask.tolist() == [True, True, True] and (mc, nz) == (3, 2), (mask, mc, nz)
    # cross 2-of-3: (1,1,-2) passes the vote, raw mean is 0 -> mask 1, nonzero(raw) 0;
    # the same coordinate fails 3-of-3, so 3-of-3 shows no mask/nonzero gap
    aggs = np.array([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [-2.0, 1.0, 0.5]])
    m2, nzn2, nzr2, J = cross_stage(aggs, 2)
    m3, nzn3, nzr3, _ = cross_stage(aggs, 3)
    assert (m2, nzr2) == (3, 2) and nzn2 == 3 and (m3, nzn3, nzr3) == (2, 2, 2) and J == 3, (m2, nzn2, nzr2, m3, nzn3, nzr3, J)
    print("[3] cancellation: 3-of-4 within (1,1,1,-3) -> mask 1 / nonzero 0; "
          "2-of-3 cross (1,1,-2) -> mask 1 / raw-mean nonzero 0; 3-of-3 -> no gap  OK")


def test4_end_to_end(recheck_path):
    try:
        import torch  # noqa: F401
        from safetensors.torch import save_file
    except Exception as e:  # pragma: no cover
        print(f"[4] skipped (torch/safetensors not importable: {e})")
        return
    import torch
    root = tempfile.mkdtemp(prefix="signcv_selftest_")
    try:
        axes, lrs, seeds = ["race", "gender", "ses"], ["0.0001", "0.0002"], ["42", "43"]
        mods = [f"base_model.model.model.layers.{l}.self_attn.{p}" for l in range(2) for p in ("q_proj", "v_proj")]
        r, alpha, dout, din = 2, 4, 12, 10
        dense = {}  # (axis, lr, seed, mod) -> numpy dense delta
        g = torch.Generator().manual_seed(123)
        for ax in axes:
            for lr in lrs:
                for sd in seeds:
                    st = {}
                    for mo in mods:
                        A = torch.randn(r, din, generator=g)
                        B = torch.randn(dout, r, generator=g)
                        # make exact cancellations possible: quantise to a coarse grid
                        A = torch.round(A * 2) / 2
                        B = torch.round(B * 2) / 2
                        st[mo + ".lora_A.weight"] = A.contiguous()
                        st[mo + ".lora_B.weight"] = B.contiguous()
                        dense[(ax, lr, sd, mo)] = ((alpha / r) * (B @ A)).numpy()
                    dd = os.path.join(root, f"{ax}_lr{lr}_seed{sd}")
                    os.makedirs(dd)
                    save_file(st, os.path.join(dd, "adapter_model.safetensors"))
                    json.dump({"r": r, "lora_alpha": alpha, "base_model_name_or_path": "synthetic",
                               "target_modules": ["q_proj", "v_proj"]}, open(os.path.join(dd, "adapter_config.json"), "w"))
        out = os.path.join(root, "out.json")
        # raw-mean aggregation keeps the arithmetic exact on the quantised grid, so torch and numpy
        # agree bit-for-bit on every sign (l2 normalisation would introduce rounding at cancellations)
        cmd = [sys.executable, recheck_path, "--root", root, "--seeds", *seeds, "--device", "cpu",
               "--within-agg", "rawmean", "--out", out]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-2000:], res.stderr[-2000:])
            raise SystemExit("[4] signcv_recheck_safe.py failed")
        got = json.load(open(out))
        # brute force with the same conventions, tensor by tensor
        within_mask = {a: 0 for a in axes}
        within_nnz = {a: 0 for a in axes}
        unan = major = joint = 0
        d = 0
        for mo in mods:
            aggs = []
            for ax in axes:
                runs = np.stack([dense[(ax, lr, sd, mo)].reshape(-1) for lr in lrs for sd in seeds])
                mask, agg, mc, nz = within_stage(runs, 3, l2mean=False)
                within_mask[ax] += mc
                within_nnz[ax] += nz
                aggs.append(agg)
            aggs = np.stack(aggs)
            unan += cross_stage(aggs, 3)[0]
            major += cross_stage(aggs, 2)[0]
            joint += cross_stage(aggs, 3)[3]
            d += aggs.shape[1]
        c = got["counts"]
        assert got["d"] == d, (got["d"], d)
        assert c["within_mask"] == within_mask, (c["within_mask"], within_mask)
        assert c["within_nnz_1e-10"] == within_nnz, (c["within_nnz_1e-10"], within_nnz)
        assert c["cross_mask_eps0"] == {"unanimity": unan, "majority": major}, (c["cross_mask_eps0"], unan, major)
        assert c["joint_nonzero"] == joint, (c["joint_nonzero"], joint)
        assert sum(got["hist27_total"]) == d
        print(f"[4] end-to-end on synthetic adapters: d={d}, within mask {within_mask}, "
              f"cross 3-of-3 = {unan}, 2-of-3 = {major}, J = {joint} — signcv_recheck_safe.py matches brute force  OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recheck", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "signcv_recheck_safe.py"))
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()
    test1_references()
    test2_iid_matched_vs_mismatched()
    test3_cancellation_semantics()
    if not args.skip_e2e:
        test4_end_to_end(args.recheck)
    print("all tests passed")


if __name__ == "__main__":
    main()
