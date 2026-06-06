# Reproduction Notes

Implementation of **"Attention, Learn to Solve Routing Problems!"** (Kool, van
Hoof, Welling — ICLR 2019, [arXiv:1803.08475](https://arxiv.org/abs/1803.08475)),
scoped to the **Euclidean TSP**.

This document records, for every implementation-relevant detail, whether the
paper **specified** it, **partially specified** it, or left it **unspecified**
(and what we chose). It is the honest map between the paper and this code.

Legend: ✅ SPECIFIED · 🟡 PARTIALLY_SPECIFIED · ❓ UNSPECIFIED · 📦 FROM_OFFICIAL_CODE

Official code (used only to resolve a few unspecified details, not copied):
<https://github.com/wouterkool/attention-learn-to-route> (PyTorch).

---

## Architecture (§3, Appendix A)

| Detail | Status | Value / Choice | Source |
|--------|--------|----------------|--------|
| Embedding dim `d_h` | ✅ | 128 | §3.1 "d_h = 128" |
| Encoder layers `N` | ✅ | 3 | §5 "N = 3 layers" |
| Attention heads `M` | ✅ | 8 | §3.1 "M = 8 heads" |
| Head dim `d_k = d_v` | ✅ | 16 = d_h/M | App. A |
| FF hidden dim | ✅ | 512, ReLU | §3.1 / App. A |
| Normalization | ✅ | BatchNorm (not LayerNorm) | §3.1 |
| Norm placement | ✅ | add-then-BN (`BN(h + sublayer(h))`) | §3.1 |
| Positional encoding | ✅ | none (order-invariant) | §3.1 |
| Graph embedding | ✅ | mean of node embeddings | §3.1 |
| Decoder context | ✅ | `[graph, last, first]` (3·d_h) | §3.2 |
| t=1 placeholders | ✅ | learned `v^l`, `v^f` | §3.2 |
| Glimpse | ✅ | M=8-head attn, single query | §3.2 |
| Pointer layer | ✅ | single head (M=1, d_k=d_h) | §3.2 |
| Logit clipping | ✅ | `C·tanh(·)`, C=10, before mask | §3.2 |
| Scaled dot-product | ✅ | `q·k / √d_k` | App. A |
| **QKV bias** | ❓ / 📦 | **bias = False** | not stated; matches official code |
| Input projection bias | ✅ | yes (`W^x x + b^x`) | §3.1 |

**Contradiction note.** Appendix A prints the head dimension as `d_k = d_v = M d_h
= 16`. With d_h=128 and M=8 the only consistent reading is `d_h / M = 16`; the
"M d_h" form is a typesetting/OCR artifact. We implement `d_k = d_h // M`. The
decoder glimpse has the same artifact; the single-head pointer correctly uses
`d_k = d_h` (since M=1).

---

## Training (§4, §5, Algorithm 1)

| Detail | Status | Value / Choice | Source |
|--------|--------|----------------|--------|
| Algorithm | ✅ | REINFORCE + greedy rollout baseline | §4, Alg. 1 |
| Optimizer | ✅ | Adam | §4 |
| Learning rate | ✅ | 1e-4 constant | §5 |
| **Adam betas** | ❓ | **(0.9, 0.999)** (PyTorch default) | not stated |
| **Adam eps** | ❓ | **1e-8** (PyTorch default) | not stated |
| **Weight decay** | ❓ | **0** | not stated |
| **Gradient clipping** | 📦 | **max_norm = 1.0** | official code (not in paper) |
| Epochs | ✅ | 100 | §5 |
| Steps / epoch | ✅ | 2500 | §5 |
| Batch size | ✅ | 512 | §5 |
| Graph size `n` | ✅ | 20 (also 50, 100 in paper) | §5 |
| Param init | ✅ | `Uniform(−1/√d, 1/√d)`, d = fan-in | §5 |
| First-epoch baseline | ✅ | exponential, β = 0.8 | §5 |
| Baseline update test | ✅ | one-sided paired t-test, α = 5% | §4 |
| t-test eval instances | ✅ | 10000 (fresh on update) | §4 |
| **Random seed** | ❓ | **1234** (paper reports robustness) | not stated |

---

## Data (§5, Appendix B.2)

| Detail | Status | Value / Choice | Source |
|--------|--------|----------------|--------|
| Instance generation | ✅ | on-the-fly per step | §5 |
| Node distribution | ✅ | Uniform in `[0,1]^2` | App. B.2 |
| Test set | ✅ | 10000 instances | §5 |

---

## Evaluation (§5, Table 1)

| Detail | Status | Value / Choice | Source |
|--------|--------|----------------|--------|
| Metric | ✅ | mean Euclidean tour length | §3, §4 |
| Greedy decode | ✅ | argmax per step | §5 |
| Sampling decode | ✅ | 1280 samples, report best | §5 |
| Optimality gap | ✅ | `cost / reference − 1` (%) | Table 1 |

---

## Out of scope (deliberately not implemented)

- **VRP, OP, PCTSP, SPCTSP** — the *same* AM with different input / mask /
  context / objective (appendices). Only TSP is implemented (the paper itself
  defines the model in terms of TSP, §3).
- **Critic (value-function) baseline** — discussed and compared in §4 but not the
  proposed method. (Exponential baseline appears only as the §5 first-epoch warmup.)
- **Beam search, 2-OPT, Gurobi / OR-Tools** baselines (§5) — comparison methods.
- Run-time benchmarking, model compression, multi-GPU.

## Known faithfulness caveats

- The paper trains 100 epochs × 2500 steps × 512 instances (~128M instances).
  The defaults reproduce that budget but require a GPU; `configs/base.yaml` keeps
  the paper values. Reduce `n_epochs` / `steps_per_epoch` for a quick CPU run.
- The t-test baseline update uses SciPy's `ttest_rel` (two-sided p halved for a
  one-sided test). The paper says "paired t-test (α = 5%)" without naming the
  one-sided variant explicitly; the official code uses the one-sided form, which
  we follow.
