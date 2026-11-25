"""Avoid the end-of-run OOM when the candidate pool is huge (e.g. MCMR: 104,981 candidates).

Symptom (upstream VLM2Vec `eval.py`, 2 GPUs, ~105k candidates):
once candidate encoding reaches 100%, the collect step allocates *another* copy of the
whole embedding matrix on the GPU:

    final_embeddings = torch.cat(slices, dim=0)   # gathered_embeds_tensor is 1.72 GiB

and raises `torch.OutOfMemoryError` after ~3.5 h of encoding.

Fix (does not change any number):
  1) with a single process (world_size == 1) skip all_gather entirely and use the CPU branch;
  2) with more ranks, drop the local/padded tensors and empty the CUDA cache before gathering.

Usage:
  REPO=/path/to/VLM2Vec python patch_eval_gather.py
"""
import io
import os
import py_compile

REPO = os.environ.get("REPO", os.environ.get("MMEB_ROOT", ".") + "/repo")
P = os.path.join(REPO, "eval.py")

OLD = (
    "    if dist.is_initialized() and full_dataset.num_rows >= world_size:\n"
    "        print_master(f\"Gathering {encode_side} embeddings across all ranks...\")\n"
)
NEW = (
    "    if dist.is_initialized() and world_size > 1 and full_dataset.num_rows >= world_size:\n"
    "        print_master(f\"Gathering {encode_side} embeddings across all ranks...\")\n"
    "        try:\n"
    "            del local_embeds, padded_embeds\n"
    "        except NameError:\n"
    "            pass\n"
    "        torch.cuda.empty_cache()\n"
)

t = io.open(P, encoding="utf-8").read()
if "world_size > 1 and full_dataset.num_rows" not in t:
    assert t.count(OLD) == 1, f"anchor found {t.count(OLD)} times in {P}"
    io.open(P, "w", encoding="utf-8").write(t.replace(OLD, NEW, 1))
    print(f"patched {P}")
else:
    print(f"already patched {P}")

py_compile.compile(P, doraise=True)
print("syntax ok")
