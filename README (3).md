<div align="center">

```
██╗     ██╗     ███╗   ███╗    ███████╗██████╗  ██████╗ ███╗   ███╗
██║     ██║     ████╗ ████║    ██╔════╝██╔══██╗██╔═══██╗████╗ ████║
██║     ██║     ██╔████╔██║    █████╗  ██████╔╝██║   ██║██╔████╔██║
██║     ██║     ██║╚██╔╝██║    ██╔══╝  ██╔══██╗██║   ██║██║╚██╔╝██║
███████╗███████╗██║ ╚═╝ ██║    ██║     ██║  ██║╚██████╔╝██║ ╚═╝ ██║
╚══════╝╚══════╝╚═╝     ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝
     ███████╗ ██████╗██████╗  █████╗ ████████╗ ██████╗██╗  ██╗
     ██╔════╝██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██║  ██║
     ███████╗██║     ██████╔╝███████║   ██║   ██║     ███████║
     ╚════██║██║     ██╔══██╗██╔══██║   ██║   ██║     ██╔══██║
     ███████║╚██████╗██║  ██║██║  ██║   ██║   ╚██████╗██║  ██║
     ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═╝    ╚═════╝╚═╝  ╚═╝
```

# LLM FROM SCRATCH

### A Large Language Model built from absolute zero — no shortcuts, no pretrained weights, no black boxes.
### Every token, every attention head, every weight — written and understood.

<br>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-Decoder--Only_Transformer-6C3483?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active_Build-00C851?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-F39C12?style=for-the-badge)

<br>

> *"I didn't want to use someone else's model and call it done. I wanted to know exactly what happens between a word going in and a word coming out. So I built the whole thing."*

</div>

---

## What Is This?

This repository is a complete, ground-up implementation of a **Large Language Model** — built the hard way on purpose. There is no Hugging Face Transformers. No pretrained checkpoints. No `AutoModel.from_pretrained()`.

Every single component — from the byte-pair encoding tokenizer, to the multi-head causal self-attention, to the training loop — is written from scratch in Python and PyTorch, with full explanations of the math and decisions behind each piece.

This exists for one reason: **to understand.** Not just to use.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Model Design](#model-design)
- [Project Structure](#project-structure)
- [Build Phases](#build-phases)
- [Hardware Requirements](#hardware-requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Inference & Generation](#inference--generation)
- [Roadmap](#roadmap)
- [Technical Deep Dives](#technical-deep-dives)

---

## Architecture Overview

The model is a **GPT-style decoder-only Transformer** — the same fundamental architecture used in GPT-2, GPT-3, LLaMA, and most modern LLMs. The key idea: given a sequence of tokens, predict the next one. Repeat. That's language modeling.

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

All model dimensions live in `config.py`. Here's what the default small model looks like and why each choice was made:

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

### Attention Mechanism — The Core

The entire model revolves around one equation:

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

In plain language:
- **Q (Query):** What this token is looking for
- **K (Key):** What each past token offers
- **V (Value):** What each past token actually contains
- **Causal Mask:** Tokens can only attend to themselves and tokens *before* them — no peeking at the future

**Multi-head** means we run this 8 times in parallel with different learned projections. Each head specializes — one might learn syntactic relationships, another semantic ones.

### Why Decoder-Only?

Encoder-decoder (like T5) is designed for translation and summarization where you have a full input sequence. Decoder-only (like GPT) is designed for **text generation** — it's simpler, scales better, and is the architecture behind every major modern LLM. We start here.

---

## Project Structure

```
llm-from-scratch/
│
├── tokenizer/
│   ├── bpe.py              # Byte-Pair Encoding — full implementation
│   ├── vocab.py            # Vocabulary builder from raw corpus
│   └── encode_decode.py    # Token ↔ text conversion utilities
│
├── data/
│   ├── dataset.py          # PyTorch Dataset — sliding window sequences
│   ├── dataloader.py       # Batching, padding, attention masks
│   └── preprocess.py       # Raw text → tokenized binary
│
├── model/
│   ├── config.py           # All hyperparameters in one place
│   ├── embeddings.py       # Token + Positional (RoPE) embeddings
│   ├── attention.py        # Multi-head causal self-attention
│   ├── mlp.py              # Feed-forward network block
│   ├── block.py            # Full transformer block (attn + mlp + norms)
│   └── gpt.py              # Full model — assembles all components
│
├── train/
│   ├── trainer.py          # Main training loop
│   ├── optimizer.py        # AdamW + LR scheduler setup
│   └── checkpoint.py       # Save/load model weights
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

## Build Phases

This project is built in deliberate phases. Each phase produces working, testable code before moving forward.

### Phase 1 — Tokenizer `[In Progress]`

**Goal:** Convert raw text into integer sequences the model can process.

We implement Byte-Pair Encoding (BPE) from scratch:

1. Start with a character-level vocabulary
2. Count all adjacent pair frequencies in the corpus
3. Merge the most frequent pair into a new token
4. Repeat 32,000 times → full vocabulary
5. Implement `encode(text) → List[int]` and `decode(tokens) → str`

```python
# What we're building
tokenizer = BPETokenizer()
tokenizer.train(corpus, vocab_size=32000)

tokens = tokenizer.encode("Hello, world!")
# → [15496, 11, 995, 0]

text = tokenizer.decode(tokens)
# → "Hello, world!"
```

---

### Phase 2 — Data Pipeline `[Planned]`

**Goal:** Feed the model clean, batched sequences efficiently.

- Load raw tokenized data as memory-mapped binary
- Sliding window over corpus: each sample = `[t₀...tₙ]` → target `[t₁...tₙ₊₁]`
- PyTorch `Dataset` + `DataLoader` with proper shuffling
- Attention masks for padded sequences

```python
# Input:  [The, cat, sat, on, the]
# Target: [cat, sat, on,  the, mat]
# The model learns: given these tokens, predict the next one
```

---

### Phase 3 — Transformer Architecture `[Planned]`

**Goal:** Build the model layer by layer.

**Step 3a — Embeddings**
```python
class TokenEmbedding(nn.Module):
    # Maps token ID → d_model dimensional vector
    # Lookup table: vocab_size × d_model

class RotaryPositionalEmbedding(nn.Module):
    # RoPE: encodes position by rotating Q and K vectors
    # Better than sinusoidal for long sequences
```

**Step 3b — Causal Self-Attention**
```python
class CausalSelfAttention(nn.Module):
    # Projects input to Q, K, V
    # Computes scaled dot-product attention
    # Applies causal mask (lower-triangular)
    # Splits into 8 heads, computes in parallel
    # Concatenates and projects back
```

**Step 3c — Transformer Block**
```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        x = x + self.attention(self.norm1(x))   # Pre-norm + residual
        x = x + self.mlp(self.norm2(x))          # Pre-norm + residual
        return x
```

**Step 3d — Full Model**
```python
class LLM(nn.Module):
    # Token embeddings
    # N × TransformerBlock
    # Final LayerNorm
    # LM Head → logits
    # Loss computation
```

---

### Phase 4 — Training Loop `[Planned]`

**Goal:** Make the model learn.

```python
optimizer = AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

# Cosine LR schedule with linear warmup
scheduler = CosineWithWarmup(optimizer, warmup_steps=200, total_steps=10000)
```

Key training decisions:
- **Loss:** Cross-entropy on next-token prediction
- **Gradient clipping:** `max_norm=1.0` — prevents exploding gradients
- **Warmup:** 200 steps of linear LR increase before cosine decay
- **Batch size:** Gradient accumulation to simulate large batches on small GPU
- **Logging:** Loss, perplexity, tokens/sec — logged every N steps

---

### Phase 5 — Inference & Sampling `[Planned]`

**Goal:** Generate text from the trained model.

```python
# Autoregressive generation
for _ in range(max_new_tokens):
    logits = model(context)          # Forward pass
    logits = logits[:, -1, :]        # Take last token's prediction
    probs = F.softmax(logits, dim=-1)
    next_token = sample(probs)        # Choose next token
    context = torch.cat([context, next_token], dim=1)
```

**Sampling strategies implemented:**

| Strategy | How it works | When to use |
|----------|-------------|-------------|
| Greedy | Always pick highest probability token | Deterministic, repetitive |
| Temperature | Scale logits before softmax — higher = more random | Creative text |
| Top-k | Sample only from top k tokens | Focused generation |
| Top-p (Nucleus) | Sample from tokens covering p% of probability mass | Best quality in practice |
| Beam Search | Keep top B candidate sequences at each step | Structured tasks |

---

### Phase 6 — Scale & Optimize `[Planned]`

**Goal:** Go from toy model to something real.

- **Mixed Precision (bf16):** ~2× memory reduction, same quality on modern GPUs
- **Gradient Checkpointing:** Trade compute for memory — recompute activations during backward pass
- **KV Cache:** Cache key/value matrices during inference — massive speedup
- **Flash Attention:** Fused GPU kernel for attention — faster and more memory efficient
- **Distributed Data Parallel (DDP):** Multi-GPU training via `torch.distributed`

---

## Hardware Requirements

| Training Stage | Minimum | Recommended | Cloud Option |
|----------------|---------|-------------|--------------|
| Tokenizer + Data (Phases 1–2) | Any CPU, 8GB RAM | — | Not needed |
| Model code (Phase 3) | Any CPU | Any GPU | Not needed |
| ~1M param model | CPU only | Any NVIDIA GPU | Google Colab Free |
| ~10M param model | RTX 3060 8GB | RTX 4090 24GB | Colab Pro |
| ~100M param model | RTX 4090 24GB | A100 40GB | Lambda Labs |
| 1B+ param model | 4× A100 80GB | 8× A100 NVLink | CoreWeave / RunPod |

**For following along with this repo:** A free Google Colab is more than enough to run Phase 1 through Phase 4 with the small model configuration.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/llm-from-scratch.git
cd llm-from-scratch
pip install -r requirements.txt
```

### 2. Prepare a dataset

```bash
# Using TinyShakespeare (built-in downloader)
python data/preprocess.py --dataset tinyshakespeare

# Or point to your own .txt file
python data/preprocess.py --file /path/to/corpus.txt
```

### 3. Train the tokenizer

```bash
python -m tokenizer.bpe --corpus data/raw/corpus.txt --vocab-size 32000
```

### 4. Train the model

```bash
python train.py --config config.py
```

### 5. Generate text

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
    # Architecture
    vocab_size:  int   = 32_000
    d_model:     int   = 256
    n_layers:    int   = 6
    n_heads:     int   = 8
    d_ff:        int   = 1024
    ctx_len:     int   = 512
    dropout:     float = 0.1

class TrainConfig:
    # Optimizer
    learning_rate:    float = 3e-4
    weight_decay:     float = 0.1
    beta1:            float = 0.9
    beta2:            float = 0.95
    grad_clip:        float = 1.0

    # Schedule
    warmup_steps:     int   = 200
    total_steps:      int   = 10_000
    decay_type:       str   = "cosine"

    # Batching
    batch_size:       int   = 64
    grad_accumulation:int   = 4      # Effective batch = 256

    # Logging & Checkpoints
    log_every:        int   = 50
    eval_every:       int   = 500
    save_every:       int   = 1_000
```

**Scaling up is just changing numbers:**

```python
# Small (10M)   → d_model=256,  n_layers=6,  n_heads=8
# Medium (100M) → d_model=512,  n_layers=12, n_heads=16
# Large (1B)    → d_model=2048, n_layers=24, n_heads=32
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
   8.  scheduler.step()                       # Update learning rate
   9.  optimizer.zero_grad()
  10.  log(loss, perplexity, tokens_per_sec)
```

### Understanding the loss

**Perplexity** = `exp(loss)` — the model's "average uncertainty" per token. A perplexity of 100 means the model is, on average, as uncertain as randomly choosing between 100 equally likely options.

| Perplexity | What it means |
|------------|---------------|
| ~1000 | Random / untrained |
| ~100 | Learning patterns |
| ~50 | Decent language model |
| ~20 | Good model — coherent sentences |
| <10 | Very strong — GPT-2 level |

---

## Inference & Generation

```python
from model.gpt import LLM
from tokenizer.bpe import BPETokenizer
from inference.generate import generate

# Load
model = LLM.from_checkpoint("checkpoints/best.pt")
tokenizer = BPETokenizer.load("checkpoints/tokenizer.json")

# Generate
output = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="The universe began",
    max_tokens=300,
    temperature=0.8,     # 1.0 = no change, <1.0 = sharper, >1.0 = wilder
    top_p=0.9,           # Nucleus sampling
    top_k=50,            # Also filter to top 50 tokens
)

print(output)
```

---

## Roadmap

```
[✅] Phase 0  — Architecture design, project planning
[🔨] Phase 1  — BPE Tokenizer (in progress)
[ ]  Phase 2  — Data pipeline & DataLoader
[ ]  Phase 3  — Transformer model (embeddings → attention → full model)
[ ]  Phase 4  — Training loop + optimizer
[ ]  Phase 5  — Inference & sampling strategies
[ ]  Phase 6  — Mixed precision + gradient checkpointing
[ ]  Phase 7  — KV Cache for fast inference
[ ]  Phase 8  — Flash Attention integration
[ ]  Phase 9  — DDP Multi-GPU training
[ ]  Phase 10 — Fine-tuning on domain-specific data
[ ]  Phase 11 — RLHF / instruction tuning (long-term)
```

---

## Technical Deep Dives

For the genuinely curious — here are the key concepts this repo covers in detail, with implementations you can read line by line:

**Attention is all you need — but why?**
Traditional RNNs process tokens sequentially — they forget distant context. Attention lets every token look at every other token simultaneously. This is why Transformers parallelized so well on GPUs and why they scaled so far.

**Why pre-norm instead of post-norm?**
Original "Attention is All You Need" used post-norm (LayerNorm after residual). Modern practice (GPT-3, LLaMA) uses pre-norm (LayerNorm before sub-layer). Pre-norm makes training more stable — gradients flow more cleanly at depth.

**Why RoPE instead of sinusoidal positional encoding?**
Sinusoidal PE adds position information to token embeddings. RoPE rotates the Q and K vectors by an angle proportional to position — this means the attention score between two tokens depends only on their *relative* position, not absolute. Better generalization to longer sequences.

**Why GELU instead of ReLU?**
ReLU is `max(0, x)` — hard zero for negatives. GELU is a smooth approximation: negative values aren't fully killed. This produces smoother gradients and empirically works better in language models.

**Why AdamW instead of Adam?**
Adam's weight decay is coupled to the adaptive learning rate — it doesn't actually regularize correctly. AdamW decouples weight decay, applying it directly to the weights. Standard in all modern LLM training.

---

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
wandb>=0.15.0          # optional — for experiment tracking
matplotlib>=3.7.0      # optional — for loss curves
```

---

## License

MIT — do whatever you want with this. If it helps you understand transformers, that's the whole point.

---

<div align="center">

**Built from scratch. Understood completely.**

*If you're reading this and want to learn — the code is the documentation.*
*Start from `tokenizer/bpe.py` and read forward.*

<br>

[![GitHub stars](https://img.shields.io/github/stars/yourusername/llm-from-scratch?style=social)](https://github.com/yourusername/llm-from-scratch)

</div>
