#!/usr/bin/env python3
"""Merge a single-task fusion result (e.g. MCMR) into the 36-task result -> 37 tasks.

The leaderboard's Image-Overall column is the unweighted mean over 37 image datasets;
this script only concatenates per-task fusion output so the two runs stay independent.

Usage:
  python3 merge_fuse.py out/fuse_36.json out/fuse_mcmr.json out/fuse_37.json
"""
import json
import sys


def main():
    base_p, mcmr_p, out_p = sys.argv[1:4]
    base = json.load(open(base_p))
    mcmr = json.load(open(mcmr_p))
    if base["models"] != mcmr["models"]:
        print("WARN: extra task used a different model subset:", [m[0] for m in mcmr["models"]])
    assert base["norm"] == mcmr["norm"], "normalisation mode differs"
    per = dict(base["per_dataset"])
    per.update(mcmr["per_dataset"])
    out = {"models": base["models"], "norm": base["norm"], "per_dataset": per,
           "mcmr_models": [m[0] for m in mcmr["models"]]}
    json.dump(out, open(out_p, "w"), indent=2, ensure_ascii=False)

    vals = per
    n36 = [d for d in vals if d != "MCMR"]
    print("items:", len(per), "(+MCMR)" if "MCMR" in per else "")
    print("36-item fused mean = %.4f" % (sum(vals[d]["fused"] for d in n36) / len(n36) * 100))
    if "MCMR" in per:
        print("MCMR fused = %.4f" % (per["MCMR"]["fused"] * 100))
        print("37-item (leaderboard Image-Overall) = %.4f"
              % (sum(vals[d]["fused"] for d in vals) / len(vals) * 100))


if __name__ == "__main__":
    main()
