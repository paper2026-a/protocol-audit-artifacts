#!/usr/bin/env python3
"""Independently verify numerical consistency of released 4-run JSON summaries.

Does not authenticate raw adapter tensors (they are NOT in this repository).
"""
import argparse, json, math, sys
from pathlib import Path

AXES=("race","gender","ses")

def recount(hist,q):
    assert isinstance(hist,list) and len(hist)==27, "27 bins required"
    total=0
    for k,v in enumerate(hist):
        assert isinstance(v,int) and v>=0
        s=(k//9-1,(k//3)%3-1,k%3-1)
        if max(s.count(1),s.count(-1))>=q:
            total+=v
    return total

def nearly(x,y,eps=1e-11):
    assert math.isfinite(x) and math.isfinite(y) and abs(x-y)<eps, (x,y)

def verify(path, allow_absolute_paths=False):
    d=json.loads(Path(path).read_text(encoding="utf-8"))
    D=d["d"]
    assert isinstance(D,int) and D>0
    assert d["n_runs"]==4 and d["q_within"]==3
    assert list(d["S_within"])==list(AXES)
    assert d["within_agg"]=="l2mean"
    assert len(d["input_manifest"])==12
    assert len({(m["axis"],m["learning_rate"],m["seed"]) for m in d["input_manifest"]})==12
    for m in d["input_manifest"]:
        p=Path(m["file"])
        if not allow_absolute_paths:
            assert not p.is_absolute() and ".." not in p.parts, "released input paths must be safe and relative"
    for ax in AXES:
        nearly(d["S_within"][ax],d["counts"]["within_mask"][ax]/D)
    primary=d["hist27_total"]
    assert sum(primary)==D
    secondary=d["hist27_eps1e-10_total"]
    assert sum(secondary)==D
    assert len(d["hist27_per_tensor"])>0
    assert len(d["hist27_eps1e-10_per_tensor"])==len(d["hist27_per_tensor"])
    per=[0]*27
    for tensor,bins in d["hist27_per_tensor"].items():
        assert len(bins)==27
        assert sum(bins)>0
        for k,c in enumerate(bins): per[k]+=c
    assert per==primary, "per tensor histogram != total histogram"
    for hist,kind in ((primary,"cross_mask_eps0"),(secondary,"cross_mask_eps1e-10")):
        assert recount(hist,3)==d["counts"][kind]["unanimity"]
        assert recount(hist,2)==d["counts"][kind]["majority"]
    s1,s2,s3=(d["S_within"][ax] for ax in AXES)
    pref3=s1*s2*s3/4
    pref2=s1*s2*s3+0.5*(s1*s2*(1-s3)+s1*s3*(1-s2)+s2*s3*(1-s1))
    nearly(pref3,d["P_ref_unanimity"])
    nearly(pref2,d["P_ref_majority"])
    for label,pref in (("unanimity",pref3),("majority",pref2)):
        rate=d["counts"]["cross_mask_eps0"][label]/D
        nearly(rate,d[f"S_cross_{label}"])
        nearly(rate/pref,d[f"R_cross_{label}"])
    nearly(d["J"],d["counts"]["joint_nonzero"]/D)
    nearly(d["J"]*d["U"],d["S_cross_unanimity"])
    nearly(d["R_sign"],4*d["U"])
    nearly(d["R_support"],d["J"]/(s1*s2*s3))
    nearly(d["R_support"]*d["R_sign"],d["R_cross_unanimity"])
    print(f"OK {Path(path).name}: d={D:,}, S3={100*d['S_cross_unanimity']:.4f}%, R3={d['R_cross_unanimity']:.4f}, SHA256 records={sum(bool(m.get('sha256')) for m in d['input_manifest'])}/12")

if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-absolute-paths", action="store_true",
                    help="For LOCAL reruns only: skip released-JSON path anonymization requirement")
    ap.add_argument("files", nargs="*", help="Result JSON files to validate")
    args=ap.parse_args()
    files=[Path(p) for p in args.files] if args.files else sorted((Path(__file__).resolve().parents[1]/"results").glob("*_4run.json"))
    if not files: sys.exit("No input JSON found")
    for f in files: verify(f, allow_absolute_paths=args.allow_absolute_paths)
