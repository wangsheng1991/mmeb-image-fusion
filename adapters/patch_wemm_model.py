"""Load WeMM-Embedding-9B through its own remote-code model class.

Why this is needed
------------------
`src/model/model.py` in the upstream VLM2Vec repo builds every `qwen3_5` backbone as
`Qwen3VLForConditionalGeneration`. WeMM-Embedding-9B ships its own implementation
(`modeling_wemm_embedding.WeMMEmbedding`, declared in `config.auto_map`), whose
`embedding()` clears the cached `rope_deltas` between multimodal forwards and whose
tokenizer appends the trailing `<embedding>` token itself. Feeding it through the
Qwen3-VL class produces vectors that are close but not identical to the official
evaluation protocol.

What this patch does (string surgery on model.py, two anchored edits)
--------------------------------------------------------------------
1) inside the `qwen3_5` loading branch: when `MMEB_WEMM=1`, build the base model with
   `AutoModel.from_pretrained(..., trust_remote_code=True, attn_implementation="sdpa")`
   and force left padding; otherwise keep the original Qwen3-VL loading code
   (the original branch body is re-indented under `else:`);
2) in the text-only encode path, clear `self.encoder.model.rope_deltas` when
   `MMEB_WEMM=1` (mirrors the reference implementation).

Usage
-----
    REPO=/path/to/VLM2Vec python patch_wemm_model.py
    # a pristine copy of model.py is kept next to it as model.py.orig (created on first run)

Notes
-----
* Only active when the environment variable `MMEB_WEMM=1`; every other backbone keeps
  the original behaviour, so the same tree can still evaluate the other models.
* Requires the sibling patch `patch_wemm_processor.py`, which fixes the chat template
  (user-only turn, instruction and body in one string, no generation prompt).
"""
import io
import os
import py_compile
import shutil

REPO = os.environ.get("REPO", os.environ.get("MMEB_ROOT", ".") + "/repo")
MODEL_PY = os.path.join(REPO, "src", "model", "model.py")
BAK = MODEL_PY + ".orig"

if not os.path.exists(BAK):
    shutil.copyfile(MODEL_PY, BAK)
    print(f"kept pristine copy at {BAK}")

t = io.open(BAK, encoding="utf-8").read()

if "\nimport os\n" not in t:
    t = t.replace("from typing import Dict\n", "from typing import Dict\nimport os\n", 1)

HEAD = "        elif model_args.model_backbone in (QWEN3_VL, QWEN3_5):\n"
TAIL = "        elif model_args.model_backbone == NVOMNIEMBED:\n"
s = t.index(HEAD)
e = t.index(TAIL)
seg = t[s:e]

# original branch body, minus the `elif` line and the `config = ...` line
lines = seg.split("\n")
assert lines[0] == "        elif model_args.model_backbone in (QWEN3_VL, QWEN3_5):"
assert "AutoConfig.from_pretrained" in lines[1]
body = lines[2:]
while body and body[-1].strip() == "":
    body.pop()
indented = "\n".join(("    " + ln) if ln.strip() else ln for ln in body)

new_seg = (
    HEAD
    + "            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)\n"
    + "            if model_args.model_backbone == QWEN3_5 and os.environ.get(\"MMEB_WEMM\", \"0\") == \"1\":\n"
    + "                # WeMM-Embedding-9B: repo ships modeling_wemm_embedding.WeMMEmbedding\n"
    + "                # (config.auto_map.AutoModel -> trust_remote_code)\n"
    + "                try:\n"
    + "                    config._attn_implementation = \"sdpa\"\n"
    + "                    if hasattr(config, \"vision_config\"):\n"
    + "                        config.vision_config._attn_implementation = \"sdpa\"\n"
    + "                except Exception as e:\n"
    + "                    print_master(f\"Warning: Could not set sdpa for WeMM: {e}\")\n"
    + "                base_model = AutoModel.from_pretrained(\n"
    + "                    model_args.model_name,\n"
    + "                    config=config,\n"
    + "                    trust_remote_code=True,\n"
    + "                    torch_dtype=torch.bfloat16,\n"
    + "                    low_cpu_mem_usage=True,\n"
    + "                    attn_implementation=\"sdpa\",\n"
    + "                )\n"
    + "                for _m in (base_model, getattr(base_model, \"model\", None)):\n"
    + "                    if _m is None:\n"
    + "                        continue\n"
    + "                    if hasattr(_m, \"padding_side\"):\n"
    + "                        _m.padding_side = \"left\"\n"
    + "                    if hasattr(_m, \"config\") and hasattr(_m.config, \"padding_side\"):\n"
    + "                        _m.config.padding_side = \"left\"\n"
    + "            else:\n"
    + indented
    + "\n"
)

t = t[:s] + new_seg + t[e:]

# text-only encode path: clear rope_deltas, as the reference implementation does
old2 = (
    "                base_input = {k: v for k, v in input.items()\n"
    "                              if v is not None and k not in {\"texts\", \"images\", \"audios\"}}\n"
    "                outputs = self.encoder.model(**base_input, return_dict=True, use_cache=False)"
)
new2 = (
    "                base_input = {k: v for k, v in input.items()\n"
    "                              if v is not None and k not in {\"texts\", \"images\", \"audios\"}}\n"
    "                if os.environ.get(\"MMEB_WEMM\", \"0\") == \"1\" and hasattr(self.encoder, \"model\"):\n"
    "                    # WeMM embedding() clears the rope_deltas cached by the last multimodal forward\n"
    "                    try:\n"
    "                        self.encoder.model.rope_deltas = None\n"
    "                    except Exception:\n"
    "                        pass\n"
    "                outputs = self.encoder.model(**base_input, return_dict=True, use_cache=False)"
)
assert t.count(old2) == 1, f"anchor found {t.count(old2)} times"
t = t.replace(old2, new2)

io.open(MODEL_PY, "w", encoding="utf-8").write(t)
py_compile.compile(MODEL_PY, doraise=True)
print(f"patched {MODEL_PY} (WeMM support), syntax ok")
