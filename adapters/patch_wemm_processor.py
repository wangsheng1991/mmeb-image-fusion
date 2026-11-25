"""Align the preprocessing of WeMM-Embedding-9B with its official evaluation protocol.

Differences between the upstream Qwen3-VL processor path in VLM2Vec and the reference
`Wemm_Embedding_process_fn` shipped with the model:
  1) a single user turn, no system turn (sentence-transformers training convention);
  2) instruction and body joined by a space inside that user turn (not split into
     system / user messages);
  3) image pixel budget 64*32*32 .. 8192*32*32 (the repo default of 4096 .. 1.8M is
     too small and costs ~8 points on VOC2007 alone);
  4) `add_generation_prompt=False`; with visual input `truncation=False`, otherwise the
     trailing `<embedding>` token is cut off;
  5) candidates carry no instruction by default (matches the official eval).

This patch appends that function to `src/model/processor.py` and dispatches to it from
`Qwen3_VL_Embedding_process_fn` when `MMEB_WEMM=1`. Nothing changes for other backbones.

Usage:
    REPO=/path/to/VLM2Vec python patch_wemm_processor.py

Diagnostics that this fixes (all observed): VOC2007 0.867 -> 0.949 for the 9B model;
with `add_generation_prompt=True` the `<embedding>` token lands *after*
`<|im_start|>assistant` and every embedding is off.
"""
import io
import os
import py_compile

REPO = os.environ.get("REPO", os.environ.get("MMEB_ROOT", ".") + "/repo")
P = os.path.join(REPO, "src", "model", "processor.py")
t = io.open(P, encoding="utf-8").read()

if "def Wemm_Embedding_process_fn" not in t:
    NEW_FN = '''
WEMM_IMAGE_KWARGS = {
    "min_pixels": 64 * 32 * 32,
    "max_pixels": 8192 * 32 * 32,
}


def Wemm_Embedding_process_fn(model_inputs: dict, processor, max_length=None):
    """Official preprocessing for WeMM-Embedding-9B (mirrors mmeb_v3_eval).

    - single user turn, add_generation_prompt=False, trailing <embedding> added by the
      tokenizer post-processor
    - instruction and body joined by a space inside the same user turn; candidates carry
      no instruction
    - image pixel budget from WEMM_IMAGE_KWARGS (override with MMEB_WEMM_{MIN,MAX}_PIXELS)
    - truncation=False when visual input is present, so the trailing <embedding> survives
    """
    texts = model_inputs.get("text", []) or []
    images_in = model_inputs.get("images", None)
    videos_in = model_inputs.get("videos", None)
    audios = model_inputs.get("audios", None)

    if audios is not None and any(a is not None for a in audios):
        raise ValueError("WeMM-Embedding does not support audio inputs in this eval pipeline.")

    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info

    def _strip_special_tokens(text: str) -> str:
        if text is None:
            return ""
        stripped = text
        all_mm_tokens = set(VLM_IMAGE_TOKENS.values()) | set(VLM_VIDEO_TOKENS.values())
        for tok in all_mm_tokens:
            if tok:
                stripped = stripped.replace(tok, "")
        return stripped.strip()

    def _normalize_video_frames(video_item):
        if video_item is None:
            return None
        if not isinstance(video_item, list):
            return video_item
        flat = []
        for frame in video_item:
            if frame is None:
                continue
            if isinstance(frame, list):
                flat.extend([x for x in frame if x is not None])
            else:
                flat.append(frame)
        return flat if flat else None

    img_min = int(os.environ.get("MMEB_WEMM_MIN_PIXELS", WEMM_IMAGE_KWARGS["min_pixels"]))
    img_max = int(os.environ.get("MMEB_WEMM_MAX_PIXELS", WEMM_IMAGE_KWARGS["max_pixels"]))
    keep_inst_on_cand = os.environ.get("MMEB_INST_ON_CAND", "0") == "1"
    video_token = VLM_VIDEO_TOKENS.get(QWEN3_5, "")

    conversations = []
    for idx, raw in enumerate(texts):
        raw = raw if isinstance(raw, str) else str(raw)
        instruction, sep, body = raw.partition(QWEN3_VL_INST_SEP)
        if not sep:
            instruction, body = "", raw
        instruction = _strip_special_tokens(instruction)
        body = _strip_special_tokens(body)
        if model_inputs.get("encode_side", "qry") == "cand" and not keep_inst_on_cand:
            instruction = ""
        if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
            instruction += "."
        prompt = (instruction + " " + body).strip() if instruction else body

        content = []
        video_item = videos_in[idx] if videos_in is not None and idx < len(videos_in) else None
        image_item = images_in[idx] if images_in is not None and idx < len(images_in) else None
        if video_item is None and isinstance(image_item, list) and video_token and video_token in raw:
            video_item, image_item = _normalize_video_frames(image_item), None
        if video_item is not None:
            content.append({"type": "video", "video": video_item})
        if image_item is not None:
            for image in (image_item if isinstance(image_item, list) else [image_item]):
                if image is not None:
                    content.append({"type": "image", "image": image,
                                    "min_pixels": img_min, "max_pixels": img_max})
        if prompt:
            content.append({"type": "text", "text": prompt})
        elif not content:
            content.append({"type": "text", "text": "NULL"})
        conversations.append([{"role": "user", "content": content}])

    chat_texts = []
    for conv in conversations:
        text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        if isinstance(text, list):
            text = "".join([x if isinstance(x, str) else str(x) for x in text])
        chat_texts.append(str(text))

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversations, image_patch_size=QWEN3_VL_PATCH_SIZE,
            return_video_kwargs=True, return_video_metadata=True)
    except TypeError:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversations, return_video_kwargs=True)

    video_metadata = None
    if video_inputs and isinstance(video_inputs[0], tuple):
        video_inputs, video_metadata = (list(x) for x in zip(*video_inputs))

    def _has_visual(x):
        if x is None:
            return False
        if not isinstance(x, list):
            return True
        return any(item is not None for item in x)

    effective_max_length = 8192 if max_length is None else int(max_length)
    processor_kwargs = dict(
        text=chat_texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        do_resize=False,
        return_tensors="pt",
    )
    if video_metadata is not None:
        processor_kwargs["video_metadata"] = video_metadata
    processor_kwargs.update(video_kwargs or {})
    if _has_visual(image_inputs) or _has_visual(video_inputs):
        processor_kwargs["truncation"] = False
    else:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = effective_max_length
    try:
        outputs = processor(**processor_kwargs)
    except RuntimeError as e:
        if "shape" in str(e) and "invalid for input of size" in str(e):
            logger.warning("WeMM preprocess failed with raw size; retrying with do_resize=True.")
            processor_kwargs["do_resize"] = True
            outputs = processor(**processor_kwargs)
        else:
            raise
    return outputs


'''
    anchor = "def Qwen3_VL_Embedding_process_fn(model_inputs: dict, processor, max_length=None):"
    assert t.count(anchor) == 1, f"anchor found {t.count(anchor)} times (applied twice?)"
    t = t.replace(anchor, NEW_FN.lstrip("\n") + anchor, 1)

# dispatch from the Qwen3-VL entry point
DELEGATE_OLD = (
    '    texts = model_inputs.get("text", []) or []\n'
    '    # Only the query side carries an instruction.'
)
DELEGATE_NEW = (
    '    if os.environ.get("MMEB_WEMM", "0") == "1":\n'
    '        return Wemm_Embedding_process_fn(model_inputs, processor, max_length=max_length)\n'
    '    texts = model_inputs.get("text", []) or []\n'
    '    # Only the query side carries an instruction.'
)
if DELEGATE_NEW not in t:
    assert t.count(DELEGATE_OLD) == 1, t.count(DELEGATE_OLD)
    t = t.replace(DELEGATE_OLD, DELEGATE_NEW, 1)

io.open(P, "w", encoding="utf-8").write(t)
py_compile.compile(P, doraise=True)
print(f"patched {P} (WeMM processor), syntax ok")
