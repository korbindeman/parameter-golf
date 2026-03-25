# Engram Memory + Legal TTT + Parallel Muon

**val_bpb: TBD** | **~16 MB** | 8×H100 SXM

## Results

Pending H100 runs. Local M2 Pro ablation (200 steps, not directly comparable to H100):

| Config | val_bpb (M2 Pro, 200 steps) | Delta vs baseline |
|--------|----------------------------|-------------------|
| Baseline (no N-gram) | 2.3499 | — |
| BigramHash (4096, dim=128) | 2.3234 | -0.0265 |
| **Engram (bigram 4096 + trigram 8192, 2 heads, dim=64, gated)** | **~2.318** | **~-0.032** |

## Key Innovation: Engram Memory Module

Replaces BigramHash with a multi-order N-gram memory module inspired by [DeepSeek Engram](https://github.com/deepseek-ai/Engram):

```python
class EngramMemory(nn.Module):
    # Deterministic multi-head hashing into learned bigram + trigram tables
    # Context-aware gating modulates injection based on hidden state
```

**vs. BigramHash:**
- BigramHash: single hash function, bigrams only, fixed scale
- Engram: 2 hash heads per order, bigrams + trigrams, learned context-dependent gating

**Architecture:**
1. **Multi-head hashing**: 2 independent XOR-based hash functions per N-gram order, reducing collision impact
2. **Bigram table**: 4096 buckets × 64 dim — captures adjacent token pair patterns
3. **Trigram table**: 8192 buckets × 64 dim — captures 3-token context patterns
4. **Context-aware gating**: `gate = sigmoid(linear(hidden_state))` — the model learns when to use vs. ignore the memory signal
5. **Projection**: engram_dim → model_dim with learned scale

**Injection point**: After token embedding + RMSNorm, before transformer blocks. SmearGate is removed when Engram is enabled since Engram's bigram component covers the same bigram-level temporal context.

**Parameter cost**: ~820K parameters (bigram: 262K, trigram: 524K, proj+gate: ~33K). Replaces BigramHash (~400K) for a net increase of ~420K parameters.

## Training Architecture

Built on the PR #549 stack:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| **Engram** | bigram=4096, trigram=8192, dim=64, heads=2 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Parameter Banking + Parallel Muon |

### Legal TTT Protocol

Same as PR #549: score-first sliding window eval, then SGD adaptation on already-scored chunks. 3 epochs, all blocks unfrozen.

## Run Command

```bash
NUM_LAYERS=11 ENGRAM_ENABLED=1 ENGRAM_BIGRAM_BUCKETS=4096 \
ENGRAM_TRIGRAM_BUCKETS=8192 ENGRAM_DIM=64 ENGRAM_HEADS=2 \
BIGRAM_VOCAB_SIZE=0 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Tuning Notes

Table sizes are configurable via env vars. Worth sweeping on H100:
- `ENGRAM_TRIGRAM_BUCKETS`: 8192 (default), 12288, 16384
- `ENGRAM_DIM`: 64 (default), 96, 128
- `ENGRAM_HEADS`: 2 (default), 3, 4

## Credits

- **Engram concept**: [DeepSeek Engram](https://github.com/deepseek-ai/Engram) (arXiv 2601.07372)
- **Base model + LeakyReLU² + TTT + Parallel Muon**: PR #549 by @abaybektursun
- **Foundational architecture**: PR #414 by @signalrush
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
