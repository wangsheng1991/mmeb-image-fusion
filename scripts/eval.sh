#!/usr/bin/env bash
# Evaluate one model on an MMEB image config with the upstream VLM2Vec eval.py.
#
# Required:
#   NAME          short name, used for the output directory
#   MODEL_PATH    local path (or hub id) of the model / processor
#   CFG           dataset config yaml (see configs/ for the image and MCMR examples)
# Optional:
#   BACKBONE      model_backbone flag (qwen3_vl | qwen3_5 | e5_omni | ...)   [qwen3_vl]
#   POOLING       pooling flag                                               [last]
#   OUT           output dir for <dataset>_{qry,tgt,info.jsonl,score.json}     [$MMEB_ROOT/out/$NAME]
#   NPROC         torchrun --nproc_per_node                                  [1]
#   BATCH         per-device eval batch size                                 [8]
#   WORKERS       dataloader workers (0 = single-threaded preprocessing)      [8]
#   GPU           CUDA_VISIBLE_DEVICES (empty = all visible)                  []
#   WEMM=1        enable the WeMM-Embedding processor/model adapters
#   MMEB_ROOT     root holding repo/, MMEB-V2-eval/ and out/                  [.]
#
# Example (MCMR, 105k candidates, one 24 GB card):
#   NAME=wemm-embedding-9b MODEL_PATH=$MMEB_ROOT/models/WeMM-Embedding-9B \
#   BACKBONE=qwen3_5 CFG=configs/mcmr.yaml BATCH=6 WEMM=1 scripts/eval.sh
set -euo pipefail

NAME=${NAME:?}
MODEL_PATH=${MODEL_PATH:?}
CFG=${CFG:?}
MMEB_ROOT=${MMEB_ROOT:-.}
REPO=${REPO:-$MMEB_ROOT/repo}
DATA=${DATA:-$MMEB_ROOT/MMEB-V2-eval}
BACKBONE=${BACKBONE:-qwen3_vl}
POOLING=${POOLING:-last}
OUT=${OUT:-$MMEB_ROOT/out/$NAME}
NPROC=${NPROC:-1}
BATCH=${BATCH:-8}
WORKERS=${WORKERS:-8}

cd "$REPO"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "${WEMM:-}" ]; then export MMEB_WEMM=1; fi
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi

mkdir -p "$OUT"
echo "==> $(date) $NAME backbone=$BACKBONE nproc=$NPROC batch=$BATCH out=$OUT"
python -m torch.distributed.run --nproc_per_node "$NPROC" \
  --master_port $((20000 + RANDOM % 5000)) eval.py \
  --pooling "$POOLING" --normalize true \
  --per_device_eval_batch_size "$BATCH" --model_backbone "$BACKBONE" \
  --model_name "$MODEL_PATH" --processor_name "$MODEL_PATH" \
  --dataset_config "$CFG" --encode_output_path "$OUT" --data_basedir "$DATA" \
  --resize_min_pixels 4096 --resize_max_pixels 1843200 \
  --dataloader_num_workers "$WORKERS"
echo "==> $(date) done $NAME"
