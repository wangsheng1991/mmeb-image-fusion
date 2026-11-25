#!/usr/bin/env python3
"""Turn our fused per-task scores into the exact JSON the MMEB Leaderboard accepts.

Usage:
    python3 build_lb_submission.py out/fuse5_fixed.json submission.json \
        --name "MODEL_NAME" --size 35 --url "https://huggingface.co/ORG/REPO" --contact you@example.com

The output shape is what https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard
reads from scores/*.json (see SUBMIT_INTRODUCTION + report_score_v3.py):

    {"metadata": {...}, "metrics": {"image": {"<dataset>": {"hit@1": 0.949, ...}, ...}}}
"""
import argparse
import json
from datetime import datetime

# Exactly the image datasets the leaderboard scores (MMEB-V1/V2 image set).
IMAGE_DATASETS = [
    "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365",
    "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211",
    "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W",
    "ScienceQA", "VizWiz", "GQA", "TextVQA",
    "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i",
    "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS",
    "MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fused_json")
    ap.add_argument("out_json")
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", default="35")
    ap.add_argument("--url", default="")
    ap.add_argument("--contact", default="")
    ap.add_argument("--backbone", default="")
    ap.add_argument("--release-date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--data-source", default="Self-Reported")
    args = ap.parse_args()

    fused = json.load(open(args.fused_json))["per_dataset"]
    scores = {}
    missing = []
    for ds in IMAGE_DATASETS:
        if ds in fused:
            scores[ds] = {"hit@1": round(float(fused[ds]["fused"]), 6)}
        else:
            missing.append(ds)
    if missing:
        raise SystemExit(f"missing datasets in fused json: {missing}")
    if "MCMR" in fused:  # V3-only 37th image dataset
        scores["MCMR"] = {"hit@1": round(float(fused["MCMR"]["fused"]), 6)}

    payload = {
        "metadata": {
            "model_name": args.name,
            "model_size": args.size,
            "model_backbone": args.backbone,
            "data_source": args.data_source,
            "model_release_date": args.release_date,
            "url": args.url,
            "contact": args.contact,
            "report_generated_date": datetime.now().isoformat(),
        },
        "metrics": {"image": scores},
    }
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=4)
    mean = sum(v["hit@1"] for v in scores.values()) / len(scores) * 100
    print(f"wrote {args.out_json}: {len(scores)} datasets")
    if "MCMR" in scores:
        print(f"  Image-Overall ({len(scores)} datasets, leaderboard column) = {mean:.2f}")
    else:
        print(f"  Image mean = {mean:.2f}; with MCMR missing (leaderboard Image-Overall) = "
              f"{mean * len(scores) / (len(scores) + 1):.2f}")


if __name__ == "__main__":
    main()
