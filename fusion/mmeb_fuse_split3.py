#!/usr/bin/env python3
"""Score-level late fusion of N embedding models, scored honestly.

For every sub-task the queries are split in half (odd/even indices by default):
  tune the simplex weights on half A by grid search (hit@1), report on half B, then swap.
Every query is therefore scored by weights that never saw it.

For speed, the candidate list of each query is first narrowed to the union of the models'
top-k, intersected with the dataset's allowed candidate set when the task is `local`.

Inputs: an MMEB dataset config (yaml) and one directory per model produced by
`eval.py --encode_output_path`, containing `<dataset>_qry`, `<dataset>_tgt` and
`<dataset>_info.jsonl`.

Usage:
  python mmeb_fuse_split3.py --config image.yaml --model q8b=out/qwen3vl-embedding-8b \
      --model wemm=out/wemm-embedding-9b --jobs 12 --json out/fuse.json
"""
import argparse
import io
import itertools
import json
import os
import pickle

import numpy as np
import yaml

CATEGORIES = {
    "CLS": ["ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365",
            "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"],
    "VQA": ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W",
            "ScienceQA", "VizWiz", "GQA", "TextVQA"],
    "RET": ["MSCOCO_i2t", "VisualNews_i2t", "VisDial", "MSCOCO_t2i", "VisualNews_t2i",
            "WebQA", "EDIS", "Wiki-SS-NQ", "CIRR", "NIGHTS", "OVEN", "FashionIQ"],
    "VG": ["MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"],
}


def simplex(n, step=10):
    out = []

    def rec(prefix, left, slots):
        if slots == 1:
            out.append(tuple(prefix + [left / step]))
            return
        for v in range(left + 1):
            rec(prefix + [v / step], left - v, slots - 1)

    rec([], step, n)
    return [np.array(w) for w in out]


def load_dir(path, ds):
    q, t, i = (os.path.join(path, "%s_%s" % (ds, s)) for s in ("qry", "tgt", "info.jsonl"))
    if not all(os.path.exists(p) for p in (q, t, i)):
        return None
    qry = pickle.load(open(q, "rb"))
    tgt = pickle.load(open(t, "rb"))
    infos = [json.loads(l) for l in open(i, encoding="utf-8")]
    return {"qry": qry.astype(np.float32), "tgt": tgt, "infos": infos, "keys": list(tgt.keys())}


def align(loaded):
    """Models may have encoded slightly different candidate sets; intersect and re-index."""
    common = list(loaded[0]["keys"])
    for r in loaded[1:]:
        s = set(r["keys"])
        common = [k for k in common if k in s]
    Ss = []
    for r in loaded:
        mat = np.stack([np.asarray(r["tgt"][k], dtype=np.float32) for k in common])
        Ss.append(np.dot(r["qry"], mat.T))
    return Ss, common, loaded[0]["infos"]


def norm_rows(S, mode):
    if mode == "none":
        return S
    if mode == "z":
        return (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-6)
    return (S - S.min(1, keepdims=True)) / np.maximum(S.max(1, keepdims=True) - S.min(1, keepdims=True), 1e-6)


def make_candidates(Ss, infos, keys, eval_type, topk=100):
    """Candidate columns per query: union of each model's top-k (local: intersect with cand_names)."""
    pos = {k: i for i, k in enumerate(keys)}
    cands = []
    for qi, info in enumerate(infos):
        cols = set()
        for S in Ss:
            cols.update(np.argsort(-S[qi])[:topk].tolist())
        if eval_type == "local" and info.get("cand_names"):
            allowed = [pos[c] for c in info["cand_names"] if c in pos]
            inter = [c for c in allowed if c in cols]
            cols = inter if inter else allowed
        cands.append(np.array(sorted(cols), dtype=np.int64))
    return cands


def hits_with(S, keys, infos, cands, qidx):
    out = []
    for qi in qidx:
        cols = cands[qi]
        best = cols[int(np.argmax(S[qi, cols]))]
        labels = infos[qi]["label_name"]
        labels = labels if isinstance(labels, list) else [labels]
        out.append(int(keys[best] in labels))
    return np.array(out)


def _fuse_one(payload):
    """Tune + score one sub-task on two disjoint halves (safe to run under a process pool)."""
    ds, eval_type, models, norm, step, topk, seed = payload
    loaded = [load_dir(p, ds) for _, p in models]
    if any(r is None for r in loaded):
        return ds, None
    Ss_raw, keys, infos = align(loaded)
    Ss = [norm_rows(S, norm) for S in Ss_raw]
    cands = make_candidates(Ss, infos, keys, eval_type, topk=topk)
    n = len(infos)
    if seed is None:
        halves = [np.arange(0, n, 2), np.arange(1, n, 2)]
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        halves = [np.sort(perm[: n // 2]), np.sort(perm[n // 2:])]
    grid = simplex(len(models), step)

    fused = np.zeros(n)
    used = []
    for i in (0, 1):
        tune, ev = halves[i], halves[1 - i]
        best_w, best_h = None, -1.0
        for w in grid:
            S = sum(wi * Si for wi, Si in zip(w, Ss))
            h = hits_with(S, keys, infos, cands, tune).mean()
            if h > best_h:
                best_w, best_h = w, h
        S = sum(wi * Si for wi, Si in zip(best_w, Ss))
        fused[ev] = hits_with(S, keys, infos, cands, ev)
        used.append([round(float(x), 2) for x in best_w])
    base = hits_with(Ss[0], keys, infos, cands, np.arange(n))
    row = {"n": n, "w_used": used, "base": float(base.mean()), "fused": float(fused.mean())}
    return ds, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--norm", default="z", choices=["z", "minmax", "none"])
    ap.add_argument("--step", type=int, default=10, help="grid steps for the weight simplex (10 = step 0.1)")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (one task per worker)")
    ap.add_argument("--topk", type=int, default=100, help="top-k per model entering the candidate shortlist")
    ap.add_argument("--seed", type=int, default=None, help="use a random half split (default: odd/even)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    models = [tuple(s.split("=", 1)) for s in args.model]
    names = [m[0] for m in models]
    grid = simplex(len(models), args.step)
    print("models: %s, %d weight candidates" % (" / ".join(names), len(grid)))

    cfg = yaml.safe_load(io.open(args.config, encoding="utf-8"))
    per_ds = {}
    jobs = [(ds, str(cfg[ds].get("eval_type", "global")).strip().lower()) for ds in cfg]
    if args.jobs and args.jobs > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        payload = [(ds, et, models, args.norm, args.step, args.topk, args.seed) for ds, et in jobs]
        with ctx.Pool(min(args.jobs, len(payload))) as pool:
            results = pool.map(_fuse_one, payload)
    else:
        results = [_fuse_one((ds, et, models, args.norm, args.step, args.topk, args.seed)) for ds, et in jobs]
    for name, row in results:
        if row is None:
            continue
        per_ds[name] = row
        print("%-22s w=%s  %s=%.4f fused=%.4f %+.4f"
              % (name, "/".join(str(u) for u in row["w_used"]), names[0], row["base"], row["fused"],
                 row["fused"] - row["base"]))

    print("\n== summary (weights tuned on the other half, scored on this one) ==")
    for tag, key in ((names[0], "base"), ("fused", "fused")):
        ov = float(np.mean([r[key] for r in per_ds.values()]))
        cats = []
        for cat, members in CATEGORIES.items():
            v = [per_ds[d][key] for d in members if d in per_ds]
            cats.append("%s=%s" % (cat, "%.2f" % (100 * np.mean(v)) if v else "NA"))
        print("%-6s overall=%.2f  %s" % (tag, 100 * ov, "  ".join(cats)))
    d = np.mean([r["fused"] - r["base"] for r in per_ds.values()])
    print("fusion %+.2f over %d tasks (norm=%s)" % (100 * d, len(per_ds), args.norm))
    if args.json:
        json.dump({"models": models, "norm": args.norm, "per_dataset": per_ds},
                  open(args.json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
