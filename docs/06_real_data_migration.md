# 6. Migrating to Real BERT and Mistral

The default v1 configuration uses a toy encoder (TF-IDF + SVD) and a toy LLM (bigram model). This is intentional — it lets the full pipeline run in <20 minutes on a laptop. For the published results, you will want real BERT and Mistral. This document describes that migration.

## What changes

### Config flag

In `src/config.py`:

```python
USE_REAL_MODELS = True  # was False
EMBED_DIM = 768          # BERT's hidden size
```

### Additional dependencies

Uncomment in `requirements.txt`:

```
torch>=2.0,<3.0
transformers>=4.40,<5.0
accelerate>=0.30,<1.0
bitsandbytes>=0.43,<1.0  # for 4-bit Mistral quantization
```

Then `pip install -r requirements.txt` again.

### Hardware

You need:

- **A GPU** (the encoder and LLM both run on GPU). Tested specs:
  - **For BERT only**: any modern GPU with ≥6 GB VRAM (RTX 3060 works).
  - **For Mistral-7B with 4-bit quantization**: ≥6 GB VRAM (RTX 3090, A4000, or better).
  - **For Mistral-7B in fp16**: ≥16 GB VRAM (A5000, A100).
- **Disk**: ~25 GB for cached BERT + Mistral checkpoints.

## What does NOT change

The code interface. Both `BertEncoder` and `MistralLLM` expose the same `fit` / `transform` / `generate` methods as their toy counterparts. The orchestrator in `pipeline.py` does not need to be modified.

## Step-by-step

### 1. Verify CUDA is available

```python
import torch
print(torch.cuda.is_available())   # should be True
print(torch.cuda.device_count())   # number of GPUs
print(torch.cuda.get_device_name(0))
```

If False, debug your CUDA install before continuing.

### 2. Pre-download the checkpoints

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# BERT
AutoTokenizer.from_pretrained("bert-base-uncased")
AutoModel.from_pretrained("bert-base-uncased")

# Mistral (this downloads ~14 GB)
AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
```

You will need a Hugging Face token if you haven't accepted Mistral's license. Run `huggingface-cli login` first.

### 3. Re-tune filter thresholds

The toy encoder produces embeddings on a different scale than BERT. Re-tune:

```python
# In src/config.py — these are starting points for BERT [CLS]:
TAU_LABEL = 0.7         # was 0.65 — BERT classifier confidence is higher
PROBE_VARIANCE_TAU = 0.03   # tighter, since BERT probes agree more
DIVERSITY_DELTA = 2.0   # higher, since BERT covariance log-det is larger
```

Calibrate these by running `scripts/run_demo.py` with `USE_REAL_MODELS=True` and inspecting the filter attrition counts in the log output. The goal is roughly:

- After Filter 1: ≥70% survive (label fidelity is usually high with Mistral).
- After Filter 2: ≥40% survive (probe calibration eliminates worst ~half).
- After Filter 3: ≥80% survive (diversity rarely fails on real LLM output).

### 4. Re-tune `V_disc` size

With Mistral's 32K-token vocabulary, expanding `TOP_K_DISC_VOCAB = 80` discriminative words to subword tokens can yield 200-500 banned token ids. This is fine for hard banning, but check generation speed — banning 1000+ tokens can slow generation 2-3×.

If generation is too slow:

```python
# In src/config.py
TOP_K_DISC_VOCAB = 40  # was 80
```

### 5. Pilot run

```bash
# Run with very few seeds first to check end-to-end works
python scripts/run_experiment.py --n-seeds 1
```

Inspect generated text samples:

```python
import json
with open("results/experiment_summary.json") as f:
    s = json.load(f)
print(s.get("demoaug_log_first_seed", {}).get("disc_vocab_examples"))
```

Verify that `V_disc` contains words from both groups (G=0 markers and G=1 markers).

### 6. Full run

```bash
python scripts/run_experiment.py
```

Estimated runtime per (regime, seed):

| Component | Toy time | Real time (single A100) |
|---|---|---|
| BERT encoding (3700 posts) | <1s | ~30s |
| LLM generation (deficit cells, ~3000 samples × 3× oversample) | ~5s | ~30 min |
| Filter cascade | ~5s | ~10s |
| Method-3 (K=25) | ~5s | ~5s |
| **Per-(regime, seed) total** | **~50s** | **~32 min** |
| **Full experiment (10 × 2)** | **~17 min** | **~10 hours** |

For the full ablation matrix (5 seeds × 3 generation temperatures × 4 ablations × 2 datasets) the estimate is ~1500 A100-GPU-hours. On a shared university cluster expect 4-6 weeks wall-clock; on commercial GPU rental ~$2,500-$3,000.

## Switching the dataset

The default uses our synthetic Forum-Post-like generator. To use a real educational dataset:

### If you have access to Sha et al. 2023 Forum

Replace `make_realistic_forum(seed)` with:

```python
import pandas as pd
def load_real_forum():
    df = pd.read_csv("path/to/sha_forum.csv")
    # Rename columns to: text, G, Y
    df = df.rename(columns={"post_text": "text", "sex": "G", "content_relevant": "Y"})
    # Ensure binary 0/1 encoding for G and Y
    return df
```

Plug into `scripts/run_experiment.py`:

```python
from your_loader import load_real_forum
df = load_real_forum()
```

Note: Sha et al.'s Forum dataset is not publicly available at time of this v1 release. The most likely public alternatives:

- **OULAD** (Open University Learning Analytics) — has VLE forum logs, demographic info, course outcomes.
- **edX MOOC discussion forum corpora** — some are released for research; check Lopez et al. 2017.

### Generic interface

The pipeline works on any DataFrame with columns:

- `text` (string)
- `G` (binary 0/1 sensitive attribute)
- `Y` (binary 0/1 task label)

Larger datasets and multi-class extensions require code changes (see `docs/07_limitations.md`).

## Likely surprises in real data

Things to watch for in your first real-data run:

1. **`V_disc` quality**: real-data `V_disc` will contain richer signals than our toy `V_disc`. Print it and have a domain expert audit before the experiment runs at scale.
2. **Probe variance is much lower** with BERT than with TF-IDF — typical `PROBE_VARIANCE_TAU=0.03` works.
3. **Method-3 stability**: with K=25 the best $\Gamma_A^{\text{IH}}$ may vary 5-10% across runs. If so, increase to K=50.
4. **Length bias**: Mistral may produce longer or shorter outputs than real posts. Check `df.groupby("G")["length"].mean()` after augmentation matches real-data ratios.
5. **Cell-imbalance might be more extreme** than our synthetic data. If a cell has <30 real samples, the calibration filter will be unreliable. Consider falling back to neutrality-targeting (the legacy `keep_mask` method in `filter_probe.py`) for those cells.

## Reporting standards for the real-data run

When publishing real-data results, include:

- The exact Hugging Face checkpoint hashes for BERT and Mistral.
- The list of seeds used.
- A small qualitative example: 3-5 generated samples per cell, in an appendix.
- A human evaluation by 5-10 educators (R2's request) — see `docs/07_limitations.md` for protocol notes.
