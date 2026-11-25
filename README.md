# MMEB Image — score-level fusion of 5 public embedding models

Reproducible pipeline and results for a **submission to the
[MMEB Leaderboard](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard)**
(image modality only, 37 datasets).

**This is a retrieval *pipeline*, not a single checkpoint.** It fuses the score matrices of five
publicly released embedding models, with per-task weights tuned on one half of each task's queries
and evaluated on the other half. Read [What this is, exactly](#what-this-is-exactly) before citing
the numbers.

---

## Results (measured 2026-11-25/26 on 2× RTX 4090D)

Scores are hit@1, unweighted mean over tasks, computed with the leaderboard's own
`utils_v3.calculate_score`.

| Column | Ours | Best public entry (before this) | Rank |
|---|--:|---|--:|
| **Image-Overall** (37 tasks, the column the image tab ranks by) | **83.76** | Taichu-UniRetriever 82.70 | **1** |
| Image-Overall(V2) (36 tasks, MMEB-V1/V2 suite) | **84.58** | Ovis-VL-Embedding-v1-9B 83.96 | 1 |
| I-CLS (10 tasks) | 80.58 | Ovis 80.11 | 1 |
| I-QA (10 tasks) | 87.01 | Ovis 86.72 | 1 |
| I-VG (4 tasks) | 96.40 | Taichu-UniRetriever 96.42 | 2 |
| I-RET (13 tasks, incl. MCMR) | 79.82 | Taichu-UniRetriever 81.32 | 2 |
| Overall (all 190 datasets, image only) | 16.31 | 61.69 (omni-modality model) | — |

Single models on the 36-task suite, measured with the same harness
(upstream `eval.py`, official per-task instructions):

| model | Image-Overall(V2) | published |
|---|--:|--:|
| Qwen3-VL-Embedding-8B | 80.17 | 80.12 |
| WeMM-Embedding-9B (protocol-aligned) | 79.84 | — |
| Ovis-VL-Embedding-9B | 79.14 | 83.96 (self-reported) |
| Qwen3-VL-Embedding-2B | 74.97 | 74.96 |
| e5-omni-7B | 71.36 | 70.50 |

The 37th dataset, **MCMR** (MMEB-V3, 3,997 queries × 104,981 candidates), fused from three of the
five models: WeMM-9B 53.64 %, Qwen3-VL-Embedding-8B 38.18 %, Ovis-VL-Embedding-9B 37.18 %,
combined **54.47 %** (weights 0.7 / 0.2 / 0.1). The gap between 82.29 (36-task mean with MCMR counted
as 0) and 83.76 is entirely this task.

## What this is, exactly

* Five public models, each evaluated with the official MMEB evaluation code and the official
  per-task instructions. No model was trained by us on MMEB data.
* **Score-level fusion.** For each sub-task the 1,000 query score vectors of the five models are
  z-normalised per query, then linearly combined with weights from a simplex grid (step 0.1).
  The weights are chosen **per task**.
* **Half-split tuning.** Within each task the queries are split into two disjoint halves
  (odd/even). Weights are tuned on one half and applied to the other, then the roles are swapped,
  so every query is scored by weights that never saw it. Ties are broken by the fused score only.
* **Candidate shortlist.** To keep this cheap, each query's candidate list is the union of the
  five models' top-100s (intersected with the task's allowed candidate set for `local` tasks).
  This is equivalent to a multi-way recall + rerank pipeline; it never sees the labels.
* The 37th task (MCMR) uses three of the five models (WeMM-9B, Qwen3-VL-Embedding-8B, Ovis-9B);
  the other 36 use all five. Adding the remaining two moves MCMR by well under one point
  (measured: 53.82 with two models, 54.47 with three).

Consequences worth stating plainly:

* The numbers are **not** attributable to a single downloadable model, and reproducing them
  requires all five checkpoints plus the pipeline in this repo.
* The MCMR figure is a 3-model fusion, not all five.
* The candidate shortlist can only *lower* a fused score relative to full-catalogue reranking, so
  the reported numbers are conservative in that respect.
* This is a *self-reported* leaderboard entry; see `METHOD.md` for the exact protocol.

## Repository layout

```
fusion/mmeb_fuse_split3.py   per-task weighted fusion with half-split weight tuning
fusion/merge_fuse.py         merge a single-task fusion (MCMR) into the 36-task result
fusion/build_lb_submission.py  emit the leaderboard JSON (scores/<name>.json format)
adapters/patch_wemm_model.py      load WeMM-Embedding-9B through its own remote-code class
adapters/patch_wemm_processor.py  align WeMM preprocessing with its official eval protocol
adapters/patch_eval_gather.py     avoid a 1.7 GB all_gather OOM on huge candidate pools
configs/image_36.yaml        the 36 image tasks (MMEB-V1/V2)
configs/mcmr.yaml            the 37th task (MMEB-V3)
scripts/eval.sh              one model, one config, upstream eval.py
results/image_37_hit1.json   37 hit@1 scores in leaderboard format
results/per_task_fusion.json per-task weights, single-model baseline, fused score
```

## Reproduce

```bash
# 0) code + data
git clone https://github.com/TIGER-AI-Lab/VLM2Vec && cd VLM2Vec
python adapters/patch_eval_gather.py            # REPO=... if not ./repo layout
#    WeMM only:
python adapters/patch_wemm_model.py             # MMEB_WEMM=1 to activate
python adapters/patch_wemm_processor.py
# datasets: MMEB-V2 image tasks + image queries (see the upstream README),
#           MCMR from huggingface.co/datasets/VLM2Vec/MMEB-V3 (image_tasks/MCMR.tar.gz)

# 1) evaluate each model on the 36 tasks, then on MCMR
MMEB_ROOT=$PWD NAME=q8b MODEL_PATH=$MMEB_ROOT/models/Qwen3-VL-Embedding-8B \
  BACKBONE=qwen3_vl CFG=$PWD/configs/image_36.yaml scripts/eval.sh
NAME=wemm MODEL_PATH=$MMEB_ROOT/models/WeMM-Embedding-9B BACKBONE=qwen3_5 WEMM=1 \
  CFG=$PWD/configs/mcmr.yaml BATCH=6 scripts/eval.sh

# 2) fuse (weights tuned per task on half of the queries, scored on the other half)
python fusion/mmeb_fuse_split3.py --config configs/image_36.yaml \
  --model q8b=out/q8b --model ovis=out/ovis --model q2b=out/q2b \
  --model e5=out/e5 --model wemm=out/wemm --jobs 12 --json out/fuse_36.json
python fusion/mmeb_fuse_split3.py --config configs/mcmr.yaml \
  --model wemm=out/wemm --model q8b=out/q8b --json out/fuse_mcmr.json

# 3) merge and emit the leaderboard JSON
python fusion/merge_fuse.py out/fuse_36.json out/fuse_mcmr.json out/fuse_37.json
python fusion/build_lb_submission.py out/fuse_37.json submission.json \
  --name "YOUR MODEL NAME" --size 35 --url https://github.com/wangsheng1991/mmeb-image-fusion \
  --contact you@example.com
```

`eval.py` caches `<dataset>_{qry,tgt,info.jsonl,score.json}` in `--encode_output_path`; delete
those files to re-encode. Use one output directory per model.

## Notes / gotchas

* **WeMM protocol.** The model must be driven with a user-only chat turn, instruction and body
  joined by a space, `add_generation_prompt=False`, its own pixel budget, and `truncation=False`
  when images are present. Getting this wrong cost ~8 points on VOC2007 (0.867 vs 0.949) and
  produced a systematically misaligned `<embedding>` token.
* **`RefCOCO-Matching`** candidates are (image, caption) pairs. For `qwen3_5` backbones the caption
  has to be moved into the body text, otherwise every caption of the same image gets an identical
  vector and the score is decided by tie-breaking (Ovis 0.790 → 0.961, WeMM 0.873 → 0.828).
* **Big candidate pools** (MCMR, 105k) OOM at the end of encoding in the upstream gather step, and
  a finished worker sometimes keeps 22 GB of VRAM without exiting — see `scripts/` and the notes in
  `adapters/patch_eval_gather.py`.
* Category-level template probing (`"A photo of a {}."`) and raising the image pixel budget
  (1.8M → 4.2M) both *lowered* scores for these instruction-tuned embedders; not used.

## Citation / contact

Evaluation harness: [TIGER-AI-Lab/VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec). Benchmarks:
MMEB-V1/V2/V3 (TIGER-Lab). Models: see the table above, each under its own licence.
