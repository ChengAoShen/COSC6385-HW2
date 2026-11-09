# Experiment A — Comparing GPUs and TPUs on Google Colab

COSC 6385 November homework. Benchmark a GPU against a **v5e-1 TPU** on three
workloads, using the same code on each so the differences come from the hardware
and the compiler, not the code.

**Submitted to Teams** as the PDF report + merged CSVs; mirrored here for the record.

➡️ **Full write-up: [`report/Experiment-A-report.pdf`](report/Experiment-A-report.pdf)** (source: [`report/REPORT.md`](report/REPORT.md)).

## Parts and headline results

| Part | Workload | Headline result |
|---|---|---|
| 1 | Matrix-multiply microbenchmark (`n ∈ {1024,2048,4096,8192}`, fp32-highest / fp32-default / bf16) | TPU bf16 → **177.7 TFLOP/s @ n=8192 (90 % of peak)**; T4 stuck ≈ 5 TFLOP/s (no native bf16) |
| 2 | CNN training on CIFAR-10 (Flax NNX), fp32/bf16 × batch {128,512} | **batch 128: T4 faster** (43k vs 32k img/s); **batch 512: TPU wins** (110k vs 44k img/s) — a roofline/arithmetic-intensity story |
| 3 | LoRA fine-tuning of **Gemma 3 1B** (KerasHub) | TPU **6.1 s/epoch** vs RTX 5070 Ti **43 s** (compiled) / 210 s (eager); ~equal peak memory |

Required accelerators: **T4 GPU** and **v5e-1 TPU** (both free tier).

## ⚠️ Part 3 GPU note

The required **Colab T4 crashes the kernel with a native fault** the moment KerasHub
runs `generate()`/`fit()` on Gemma 3 1B via the JAX-CUDA path (the model *loads*
fine; lowering the sequence length does not help; an L4 behaves the same). This is
documented as a result in the report's *Problems* section. To still produce a real
GPU-vs-TPU **training** comparison, Part 3's GPU side was reproduced on a local
**RTX 5070 Ti** with the *same* Keras 3 / KerasHub model and the identical LoRA
config, on the PyTorch backend (stable on Blackwell) — see
[`notebooks/run_gpu_local.py`](notebooks/run_gpu_local.py).

## Contents

```
colab-gpu-tpu/
├── notebooks/
│   ├── colab_gpu_tpu_benchmark_FIXED.ipynb   Parts 1 & 2 (patched for the Nov-2026 Colab image)
│   ├── gemma3_lora_finetune.ipynb            Part 3 (TPU / T4)
│   ├── run_gpu_local.py                      Part 3 GPU number, one command on any CUDA GPU
│   └── colab_gpu_tpu_benchmark.ipynb         original starter notebook (reference)
├── report/   Experiment-A-report.pdf  (+ REPORT.md source, figures/)
└── results/  merged CSVs + Gemma before/after sample transcripts
```

## How to run

1. Open each notebook in Google Colab, `Runtime → Change runtime type`, pick the
   accelerator, run top-to-bottom, download the CSVs. See
   [`RUN_INSTRUCTIONS.md`](RUN_INSTRUCTIONS.md) for the exact per-accelerator steps
   and the environment fixes baked into the notebooks.
2. For the Part-3 GPU number on a local NVIDIA GPU: `python notebooks/run_gpu_local.py`
   (needs `KAGGLE_API_TOKEN` for the Gemma download; Keras 3 + KerasHub + PyTorch).
