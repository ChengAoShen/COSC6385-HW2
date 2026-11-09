# How to run Experiment A on Colab

The notebooks are pre-patched for the current Colab image (Nov 2026): the first
code cell pins `flax==0.10.6` and re-adds a removed JAX symbol, and CIFAR-10 is
loaded without TensorFlow. So each accelerator is just **Run all**, no debugging.

Local paths of the notebooks:
```
notebooks/colab_gpu_tpu_benchmark_FIXED.ipynb   # Parts 1 & 2
notebooks/gemma3_lora_finetune.ipynb            # Part 3 (Gemma LoRA)
```

## Parts 1 & 2 — matmul + CNN  (run on **T4 GPU**)

> The **v5e-1 TPU** run of Parts 1 & 2 is already done (see `results/*_TPU_v5e.csv`).
> You only need the **T4 GPU** run to complete the GPU-vs-TPU comparison.

1. https://colab.research.google.com → **File → Upload notebook** → pick
   `colab_gpu_tpu_benchmark_FIXED.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. **Runtime → Run all.** (First cell installs flax 0.10.6, ~1 min; the whole
   run is a few minutes.)
4. At the end it downloads `results_matmul.csv` and `results_cnn.csv`. If Chrome
   asks, allow multiple downloads. (Or use the Files panel on the left → download.)
5. Leave both files in your Downloads folder and tell me — I'll pick them up.
   (Optional: rename them `*_T4.csv` so they don't overwrite the TPU ones.)

## Part 3 — Gemma LoRA  (run on **v5e-1 TPU**, then **T4 GPU**)

**One-time:** in Colab, click the **🔑 key icon** (left sidebar) → **Add new secret**
→ name it `KAGGLE_API_TOKEN`, paste the Kaggle token you created, and toggle
**Notebook access** on. (Your Kaggle account has already accepted the Gemma 3
license, so this is all that's needed.)

1. **Upload** `gemma3_lora_finetune.ipynb`.
2. **Runtime → v5e-1 TPU → Run all.** Downloads Gemma 3 1B (~2 GB), runs LoRA
   fine-tuning, prints sample generations before/after, then downloads
   `gemma_lora_results.csv` and a samples `.txt`. (~10–15 min.)
3. **Runtime → Change runtime type → T4 GPU → Run all** again. This appends a T4
   row to the same CSV so the two devices can be compared.
4. Tell me when the files are in your Downloads folder.

## What I do with the results

I merge your CSVs with the TPU data already collected, make the figures, and
write the PDF report (Environment / Method / Problems / Results / Analysis).
Then everything goes into the `COSC6385-HW2` GitHub repo and the report + CSVs
are submitted to Teams.
