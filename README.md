<div align="center">

```
██╗ ██████╗ ███╗   ██╗██╗████████╗███████╗
██║██╔════╝ ████╗  ██║██║╚══██╔══╝██╔════╝
██║██║  ███╗██╔██╗ ██║██║   ██║   █████╗  
██║██║   ██║██║╚██╗██║██║   ██║   ██╔══╝  
██║╚██████╔╝██║ ╚████║██║   ██║   ███████╗
╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝   ╚══════╝
     ██╗     ██╗     ███╗   ███╗
     ██║     ██║     ████╗ ████║
     ██║     ██║     ██╔████╔██║
     ██║     ██║     ██║╚██╔╝██║
     ███████╗███████╗██║ ╚═╝ ██║
     ╚══════╝╚══════╝╚═╝     ╚═╝
```

# Ignite-LLM

### A Large Language Model built from absolute zero — no shortcuts, no pretrained weights, no black boxes.
### Every token, every attention head, every weight — written and understood.
### Runs 100% locally on your own machine. Free. No cloud required.

<br>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-Decoder--Only_Transformer-6C3483?style=for-the-badge)
![Hardware](https://img.shields.io/badge/GPU-RTX_3060_8GB-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active_Build-00C851?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-F39C12?style=for-the-badge)

<br>

> *"I didn't want to use someone else's model and call it done. I wanted to know exactly what happens between a word going in and a word coming out. So I built the whole thing — on my own machine."*

</div>

---

## What Is This?

This repository is a complete, ground-up implementation of a **Large Language Model** — built the hard way on purpose. There is no Hugging Face Transformers. No pretrained checkpoints. No `AutoModel.from_pretrained()`.

Every single component — from the byte-pair encoding tokenizer, to the multi-head causal self-attention, to the training loop — is written from scratch in Python and PyTorch.

Everything runs **locally on your own PC**. No cloud account. No credit card. No internet connection needed after the first setup.

This exists for one reason: **to understand.** Not just to use.

---

## Table of Contents

- [Your Hardware](#your-hardware--what-it-can-do)
- [Architecture Overview](#architecture-overview)
- [Model Design](#model-design)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup-windows--linux)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Inference & Generation](#inference--generation)
- [Need More Power? Cheap Cloud Options](#need-more-power-cheap-cloud-options)
- [Roadmap](#roadmap)
- [Technical Deep Dives](#technical-deep-dives)

---

## My Hardware — What It Can Do

You have an **NVIDIA GeForce RTX 3060 8GB + 64GB RAM**. Here's exactly what that means for Ignite-LLM:

| Model Size | Params | Fits on Your GPU? | Training Time (est.) |
|------------|--------|-------------------|----------------------|
| Small (default) | ~10M | ✅ Yes, easily | ~1–2 hours on TinyShakespeare |
| Medium | ~85M | ✅ Yes, with AMP + grad checkpointing | ~6–12 hours |
| Large | ~350M | ⚠️ Tight — reduce batch size to 8 | ~2–3 days |
| 1B+ | 1B+ | ❌ Not enough VRAM | Needs cloud |

**The default config (Small, ~10M params) is tuned specifically for your RTX 3060.** You can train it, watch it learn, and generate text — all for free, right now.

Key optimisations already enabled for your GPU:
- **bfloat16 mixed precision** — halves VRAM usage, RTX 3060 supports it natively (Ampere arch)
- **Gradient checkpointing** — trades a bit of compute for ~40% less VRAM
- **Gradient accumulation** — simulates a batch of 256 using only 32 samples at a time
- **Memory-mapped datasets** — 64GB RAM means you can load huge corpora without issues

---

## Architecture Overview

The model is a **GPT-style decoder-only Transformer** — the same fundamental architecture used in GPT-2, GPT-3, LLaMA, and most modern LLMs.

```
Raw Text
   │
   ▼
┌─────────────────────────────────────────────┐
│               TOKENIZER (BPE)               │
│  "Hello world" → [15496, 995]               │
│  Vocabulary size: 32,000 tokens             │
└─────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────┐
│            TOKEN EMBEDDING TABLE            │
│  Each token ID → dense vector (d_model=256) │
└─────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────┐
│          POSITIONAL ENCODING (RoPE)         │
│  Injects position information into vectors  │
└─────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────┐  ─┐
│           TRANSFORMER BLOCK × N             │   │
│  ┌───────────────────────────────────────┐  │   │
│  │  LayerNorm (Pre-Norm)                 │  │   │
│  └───────────────────────────────────────┘  │   │
│  ┌───────────────────────────────────────┐  │   │ × 6 layers
│  │  Multi-Head Causal Self-Attention     │  │   │
│  │  ┌─────┐  ┌─────┐  ┌─────┐          │  │   │
│  │  │  Q  │  │  K  │  │  V  │          │  │   │
│  │  └──┬──┘  └──┬──┘  └──┬──┘          │  │   │
│  │     └────────┴─────────┘             │  │   │
│  │     Scaled Dot-Product + Causal Mask │  │   │
│  │     8 heads × 32 head_dim            │  │   │
│  └───────────────────────────────────────┘  │   │
│  + Residual Connection                      │   │
│  ┌───────────────────────────────────────┐  │   │
│  │  LayerNorm                            │  │   │
│  └───────────────────────────────────────┘  │   │
│  ┌───────────────────────────────────────┐  │   │
│  │  Feed-Forward MLP                     │  │   │
│  │  Linear(256→1024) → GELU → Linear    │  │   │
│  └───────────────────────────────────────┘  │   │
│  + Residual Connection                      │   │
└─────────────────────────────────────────────┘  ─┘
   │
   ▼
┌─────────────────────────────────────────────┐
│             FINAL LAYER NORM                │
└─────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────┐
│          LM HEAD (Linear Projection)        │
│  d_model(256) → vocab_size(32,000)          │
│  Output: logits over full vocabulary        │
└─────────────────────────────────────────────┘
   │
   ▼
Predicted Next Token (+ sampling strategy)
```

---

## Model Design

### Hyperparameter Configuration

All model dimensions live in `config.py`. Here's what the default small model looks like:

| Parameter | Value | Why |
|-----------|-------|-----|
| `vocab_size` | 32,000 | BPE vocabulary — balances coverage vs embedding table size |
| `d_model` | 256 | Embedding dimension — the width of the model |
| `n_layers` | 6 | Number of stacked transformer blocks — the depth |
| `n_heads` | 8 | Attention heads — each learns different relationships |
| `d_head` | 32 | Per-head dimension (`d_model / n_heads = 256/8`) |
| `d_ff` | 1024 | FFN hidden size — typically `4 × d_model` |
| `ctx_len` | 512 | Max context/sequence length (tokens) |
| `dropout` | 0.1 | Regularization during training |
| `activation` | GELU | Smoother than ReLU — standard in modern LLMs |
| `pos_encoding` | RoPE | Rotary Positional Embedding — handles long contexts better |

**Total parameters (small model): ~10M**

---

## Project Structure

```
Ignite-LLM/
│
├── tokenizer/
│   ├── bpe.py              # Byte-Pair Encoding — full implementation
│   └── __init__.py
│
├── data/
│   ├── dataset.py          # PyTorch Dataset — sliding window sequences
│   ├── preprocess.py       # Raw text → tokenized binary
│   └── __init__.py
│
├── model/
│   ├── embeddings.py       # Token + Positional (RoPE) embeddings
│   ├── attention.py        # Multi-head causal self-attention
│   ├── mlp.py              # Feed-forward network block
│   ├── block.py            # Full transformer block (attn + mlp + norms)
│   ├── gpt.py              # Full model — assembles all components
│   └── __init__.py
│
├── train/
│   ├── trainer.py          # Main training loop
│   ├── optimizer.py        # AdamW + LR scheduler setup
│   ├── checkpoint.py       # Save/load model weights
│   └── __init__.py
│
├── inference/
│   ├── generate.py         # Autoregressive generation
│   └── sampling.py         # Greedy, temperature, top-k, top-p
│
├── checkpoints/            # Saved model weights (gitignored)
├── data/raw/               # Raw text datasets (gitignored)
│
├── config.py               # Global config — single source of truth
├── utils.py                # Device setup, logging, seeding
├── train.py                # Entry point for training
├── generate.py             # Entry point for inference
│
├── requirements.txt
└── README.md
```

---

## Local Setup (Windows & Linux)

Everything runs locally. No cloud. No accounts. Just your PC.

### Step 1 — Install Python 3.11+

**Windows:** Download from [python.org](https://www.python.org/downloads/) and check "Add to PATH" during install.

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip -y
```

### Step 2 — Install CUDA Toolkit

Your RTX 3060 needs CUDA to train on GPU. Download **CUDA 12.x** from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

Check it installed correctly:
```bash
nvidia-smi
# Should show: NVIDIA GeForce RTX 3060, CUDA Version: 12.x
```

### Step 3 — Clone and set up the project

```bash
git clone https://github.com/yourusername/Ignite-LLM.git
cd Ignite-LLM

# Create a virtual environment (keeps things clean)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 4 — Install PyTorch with CUDA support

Go to [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and select your OS + CUDA version, or run:

```bash
# PyTorch with CUDA 12.1 (works on RTX 3060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 5 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 6 — Verify GPU is detected

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 3060
```

You're ready to train.

---

## Quick Start

### 1. Download and preprocess a dataset

```bash
# TinyShakespeare — ~1MB, trains in under 2 hours on your RTX 3060
python data/preprocess.py --dataset tinyshakespeare

# Or use your own text file
python data/preprocess.py --file /path/to/your/corpus.txt
```

### 2. Train the model

```bash
python train.py
```

That's it. Ignite-LLM will automatically detect your RTX 3060, enable bfloat16 mixed precision, and start training. You'll see live loss + perplexity + tokens/sec in your terminal.

### 3. Generate text

```bash
python generate.py \
  --checkpoint checkpoints/best.pt \
  --prompt "Once upon a time" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-p 0.9
```

---

## Configuration

All hyperparameters live in `config.py`. Change one file, everything updates.

```python
class ModelConfig:
    vocab_size:  int   = 32_000
    d_model:     int   = 256
    n_layers:    int   = 6
    n_heads:     int   = 8
    d_ff:        int   = 1_024
    ctx_len:     int   = 512
    dropout:     float = 0.1

class TrainConfig:
    learning_rate:     float = 3e-4
    weight_decay:      float = 0.1
    grad_clip:         float = 1.0
    warmup_steps:      int   = 200
    total_steps:       int   = 10_000

    # RTX 3060 8GB tuned settings
    batch_size:        int   = 32
    grad_accumulation: int   = 8     # Effective batch = 256
    use_amp:           bool  = True  # bfloat16 — saves ~50% VRAM
    gradient_checkpointing: bool = True  # saves another ~40% VRAM
```

**Scaling up is just changing numbers:**

```python
# Small  (~10M)  → d_model=256,  n_layers=6,  n_heads=8   ← default, use this
# Medium (~85M)  → d_model=512,  n_layers=12, n_heads=16  ← works on your GPU
# Large  (~350M) → d_model=1024, n_layers=24, n_heads=16  ← reduce batch_size=8
```

---

## Training

### What the training loop does

```
for each batch:
   1.  tokens       = batch[:, :-1]          # Input: all but last
   2.  targets      = batch[:, 1:]            # Target: all but first
   3.  logits       = model(tokens)           # Forward pass
   4.  loss         = cross_entropy(logits, targets)
   5.  loss.backward()                        # Compute gradients
   6.  clip_grad_norm(model, 1.0)            # Clip gradients
   7.  optimizer.step()                       # Update weights
   8.  scheduler.step()                       # Update LR
   9.  optimizer.zero_grad()
  10.  log(loss, perplexity, tokens_per_sec, VRAM_used)
```

### Understanding the loss

**Perplexity** = `exp(loss)` — the model's average uncertainty per token.

| Perplexity | What it means |
|------------|---------------|
| ~1000 | Random / untrained |
| ~100 | Learning patterns |
| ~50 | Decent language model |
| ~20 | Good model — coherent sentences |
| <10 | Very strong — GPT-2 level |

### Expected training speed on RTX 3060

| Model | Batch | Tokens/sec | Time for 10K steps |
|-------|-------|------------|-------------------|
| Small (10M) | 32 × 8 accum | ~80,000 | ~1.5 hours |
| Medium (85M) | 16 × 16 accum | ~25,000 | ~5 hours |

---

## Inference & Generation

```python
from model.gpt import IgniteLLM
from tokenizer.bpe import BPETokenizer
from inference.generate import generate

# Load
model = IgniteLLM.from_checkpoint("checkpoints/best.pt")
tokenizer = BPETokenizer.load("data/tokenizer.json")

# Generate
output = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="The universe began",
    max_tokens=300,
    temperature=0.8,     # 1.0 = no change, <1.0 = sharper, >1.0 = wilder
    top_p=0.9,           # Nucleus sampling
    top_k=50,
)

print(output)
```

---


## Roadmap

```
[✅] Phase 0  — Architecture design, project planning
[✅] Phase 1  — BPE Tokenizer
[✅] Phase 2  — Data pipeline & DataLoader
[✅] Phase 3  — Transformer model (embeddings → attention → full model)
[✅] Phase 4  — Training loop + optimizer
[✅] Phase 5  — Mixed precision (bfloat16) + gradient checkpointing
[🔨] Phase 6  — Inference & sampling strategies
[ ]  Phase 7  — KV Cache for fast inference
[ ]  Phase 8  — Flash Attention integration
[ ]  Phase 9  — Fine-tuning on domain-specific data
[ ]  Phase 10 — RLHF / instruction tuning (long-term)
```

---

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0      # optional — for loss curves
```

Install with:
```bash
pip install -r requirements.txt
```

---

## License

MIT — do whatever you want with this. If it helps you understand transformers, that's the whole point.

---

<div align="center">

**Built from scratch. Runs on your machine. Understood completely.**

*If you're reading this and want to learn — the code is the documentation.*
*Start from `tokenizer/bpe.py` and read forward.*

<br>

[![GitHub stars](https://img.shields.io/github/stars/yourusername/Ignite-LLM?style=social)](https://github.com/yourusername/Ignite-LLM)

</div>
