#!/usr/bin/env python
"""COSC 6385 - Experiment A, Part 3 (GPU side), run LOCALLY on an NVIDIA GPU.

The required Colab T4 crashes (native segfault) on the jax-CUDA path while
training Gemma 3 1B, so we reproduce the *same* Keras 3 / KerasHub LoRA workload
on a local RTX 5070 Ti using the PyTorch backend (reliable on Blackwell), with
identical config to the TPU run so the GPU-vs-TPU comparison stays valid:
  preset gemma3_1b, SEQLEN=256, EPOCHS=3, BATCH=1, LORA_RANK=4, dolly-15k x400.
Writes one row to gemma_lora_results_gpu.csv with the same columns as the TPU CSV.
"""
import os, time, datetime, platform, json

os.environ["KERAS_BACKEND"] = "torch"
# Kaggle creds: read KAGGLE_API_TOKEN from env, or from ~/.kaggle/kaggle_api_token
if not os.environ.get("KAGGLE_API_TOKEN"):
    p = os.path.expanduser("~/.kaggle/kaggle_api_token")
    if os.path.exists(p):
        os.environ["KAGGLE_API_TOKEN"] = open(p).read().strip()

import numpy as np, pandas as pd
import torch
import keras, keras_hub

# Fair comparison with the XLA-compiled TPU run: compute in bf16 (engage the
# GPU tensor cores, as the TPU MXU does) and compile the training step.
keras.mixed_precision.set_global_policy("mixed_bfloat16")

assert torch.cuda.is_available(), "CUDA GPU not visible to torch"
DEV = torch.cuda.get_device_name(0)
RUN_ID = f"{DEV.replace(' ', '_')}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
ENV = dict(
    run_id=RUN_ID, platform="gpu", device_kind=DEV, n_devices=1,
    jax=f"torch-{torch.__version__}",          # backend column (this run uses torch, not jax)
    keras=keras.__version__, keras_hub=keras_hub.__version__,
    python=platform.python_version(),
    colab_runtime="local (not Colab); Keras3 torch backend; mixed_bfloat16 + jit_compile",
)
for k, v in ENV.items():
    print(f"{k:>14}: {v}")

SEQLEN, EPOCHS, BATCH, LORA_RANK = 256, 3, 1, 4

# ---- 1. load Gemma 3 1B (same preset as the TPU run) ----
t0 = time.perf_counter()
gemma = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_1b")
gemma.preprocessor.sequence_length = SEQLEN
load_s = time.perf_counter() - t0
print(f"loaded gemma3_1b in {load_s:.1f}s")

# ---- 2. dataset: 400 context-free Dolly examples, Gemma3 prompts/responses dict ----
import urllib.request
TPL = "Instruction:\n{instruction}\n\nResponse:\n"
prompts, responses = [], []
try:
    url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
    raw = urllib.request.urlopen(url, timeout=30).read().decode()
    for line in raw.splitlines():
        r = json.loads(line)
        if r.get("context"):
            continue
        prompts.append(TPL.format(instruction=r["instruction"]))
        responses.append(r["response"])
        if len(prompts) >= 400:
            break
    src = "dolly-15k (first 400 context-free)"
except Exception as e:
    print("download failed, tiny set:", type(e).__name__, e)
    pairs = [("What is the capital of France?", "The capital of France is Paris."),
             ("Name three primary colors.", "Red, blue, and yellow."),
             ("What is 2+2?", "It is 4.")] * 40
    prompts = [TPL.format(instruction=i) for i, _ in pairs]
    responses = [r for _, r in pairs]
    src = "built-in tiny set"
data = {"prompts": prompts, "responses": responses}
print(len(prompts), "examples from", src)

# ---- 3. before samples (torch backend errors are catchable) ----
PROMPTS = [
    "Instruction:\nExplain what a TPU is in one sentence.\n\nResponse:\n",
    "Instruction:\nWhat is speculative execution?\n\nResponse:\n",
    "Instruction:\nGive me a haiku about GPUs.\n\nResponse:\n",
]
def sample(tag):
    outs = []
    for p in PROMPTS:
        try:
            o = gemma.generate(p, max_length=96)
        except Exception as e:
            o = f"[generate failed: {type(e).__name__}: {e}]"
        outs.append(o)
        print(f"--- [{tag}] {p!r}\n{o}\n")
    return outs
before = sample("before")

# ---- 4. enable LoRA and fine-tune ----
gemma.backbone.enable_lora(rank=LORA_RANK)
gemma.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    jit_compile=True,   # compile the training step (torch.compile) -> GPU analogue of XLA
)

class EpochTimer(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.t = time.perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        dt = time.perf_counter() - self.t
        self.times = getattr(self, "times", []) + [dt]
        print(f"  epoch {epoch}: {dt:.1f}s  loss={logs.get('loss'):.4f}")

torch.cuda.reset_peak_memory_stats()
timer = EpochTimer()
t0 = time.perf_counter()
gemma.fit(data, epochs=EPOCHS, batch_size=BATCH, callbacks=[timer])
train_s = time.perf_counter() - t0

peak_mb = torch.cuda.max_memory_allocated() / 1e6
steady = timer.times[1:] if len(timer.times) > 1 else timer.times
epoch_s = float(np.median(steady))
print(f"median steady epoch: {epoch_s:.1f}s   peak_mem: {peak_mb:.1f} MB")

# ---- 5. after samples ----
after = sample("after")

# ---- 6. record ----
row = dict(**ENV, seqlen=SEQLEN, epochs=EPOCHS, batch=BATCH, lora_rank=LORA_RANK,
           load_s=round(load_s, 1), train_total_s=round(train_s, 1),
           epoch_s=round(epoch_s, 2), peak_mem_mb=round(peak_mb, 1),
           epoch_times=json.dumps([round(t, 1) for t in timer.times]))
out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma_lora_results_gpu.csv")
pd.DataFrame([row]).to_csv(out_csv, index=False)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"gemma_samples_{RUN_ID}.txt"), "w") as f:
    for p, b, a in zip(PROMPTS, before, after):
        f.write(f"PROMPT: {p}\n--- BEFORE ---\n{b}\n--- AFTER ---\n{a}\n\n{'='*60}\n")
print("WROTE", out_csv)
print(pd.DataFrame([row]).to_string(index=False))
