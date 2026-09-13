# Video Moment Retrieval (VMR) — Experiment Log

Two-stage pipeline: CLIP-based coarse retrieval → MomentDETRLite grounding head
(fine-grained timestamp regression). Built from a scaffolded codebase, trained
and evaluated on Charades-STA (+ a QVHighlights subset).

Environment: Windows, RTX 4050 laptop GPU, project at `E:\vmr`.

---

## 1. Pipeline built and validated end-to-end

- `src/pipeline.py` runs CLIP retrieval → grounding head → final timestamps.
- Two real bugs found and fixed during initial validation:
  1. **Train/inference coordinate mismatch** (`src/grounding/dataset.py`) — training
     cropped differently than inference did. Fixed by cropping locally around GT
     + `context_pad_sec`, matching inference exactly.
  2. **`top_k_windows` not scaling for short videos** (`src/pipeline.py`) — fixed
     with `effective_k = min(top_k_windows, len(windows)//2)`.

## 2. Training loss rewritten (`src/grounding/train.py`)

- Added **GIoU loss** (fixes vanishing gradients when predicted/GT spans don't
  overlap at all — plain IoU gives zero gradient in that case).
- Added **full negative-slot confidence supervision** — previously only the
  matched query slot got a confidence target; unmatched slots got no supervision
  at all, which caused the model to hit false 1.0 confidence on wrong predictions.

## 3. Honest end-to-end evaluation built (`eval_e2e.py`)

- The original `src/grounding/eval.py` used an "oracle" GT-centered crop at
  inference time — this artificially inflates scores (was showing ~0.89/0.49,
  which is not achievable by the real two-stage pipeline).
- `eval_e2e.py` runs the actual two-stage pipeline with no GT leakage. All
  results below are from this honest script.

## 4. Data

- **QVHighlights**: 250-video subset downloaded via HF mirror
  (`ayushsdev/qvhighlights-videos`), features extracted, format-verified.
- **Charades-STA**: all 9,848 raw videos downloaded (AllenAI 480p set), CLIP
  features extracted for the full set (up from an initial 2,400-video subset).
  `train_subset.jsonl` / `test_subset.jsonl` rebuilt against full features:
  12,408/12,408 train, 3,720/3,720 test kept.

## 5. Checkpointing & resume added to `train.py`

- Periodic checkpoints every N epochs, rolling "latest" (for resume), and
  "best" (lowest train loss).
- Resume support added (`--resume`), storing model + optimizer + epoch state.

## 6. Confidence-loss bug found and fixed

- Diagnosis: the grounding head predicts `num_queries=5` candidate spans per
  window. Confidence was trained as **5 independent binary decisions**
  (`BCEWithLogitsLoss` per slot: "is this the match, yes/no").
- With exactly 1 positive vs 4 negatives every single example, this setup
  rewards the model for pushing all confidence scores down generally, rather
  than learning to sharply prefer the correct slot. Observed symptom: `conf`
  loss stayed flat (~0.47) for all 80 epochs of the first full-data run, while
  `l1`/`giou` improved steadily — the regression head was learning fine, the
  slot-selection head was not.
- **Fix applied**: switched to `cross_entropy(conf, best_q)` — a proper
  single "which of these 5 slots is correct" classification, instead of 5
  independent yes/no questions.
- Result after fix: `conf` loss still hovers close to the random-guess level
  for 5 classes (`ln(5) ≈ 1.609`), ending around ~1.53. Some improvement over
  pure chance, but the slot-confidence signal remains weak — see Section 8.

## 7. Results — honest end-to-end eval (200-example sample, `eval_e2e.py`)

| Run | R@1 IoU=0.5 | R@1 IoU=0.7 | Notes |
|---|---|---|---|
| 2,400-video subset, original loss | 0.32 | 0.16 | Baseline, before full-data retrain |
| 9,848-video full data, buggy conf loss (independent BCE) | 0.265 | 0.095 | Full data made things *worse* — traced to confidence-loss bug above |
| 9,848-video full data, fixed conf loss (cross-entropy) | 0.275 | 0.135 | Confirms the loss fix was a real improvement (esp. at IoU=0.7); still below the subset baseline at IoU=0.5 |

Reference point — published Moment-DETR at full scale (large-scale pretraining
+ full compute budget): ~0.53 / ~0.31. All runs above are meaningfully below
this, as expected for a from-scratch model on a single laptop GPU with no
large-scale video-text pretraining.

## 8. Current diagnosis (open issue)

Even after the loss-function fix, the confidence head is only weakly
discriminating between the 5 predicted query slots (loss close to random-guess
level). Two candidate explanations, not yet confirmed:
- The 5 learned query slots may not be differentiating from each other enough
  (near-identical outputs → no real signal for "which one is right").
- DETR-style query/slot-matching architectures are documented in the
  literature as generally needing large-scale pretraining data to train this
  mechanism well — plausible this is a structural limitation of the
  architecture choice at this data/compute scale, not a remaining
  implementation bug.

## 10. Breakthrough — Stage 1 + Stage 2 Score Fusion

### Problem Found
Previously, Stage 2 ranked predictions solely using the local grounding head's internal confidence logit (`conf`), completely discarding Stage 1's coarse retrieval cosine similarity (`sim`). A false-positive candidate window with high internal local confidence would override the true video moment that had strong global Stage 1 semantic alignment.

### Solution
Implemented score fusion in `src/pipeline.py` and `eval_e2e.py`:
$$\text{Score}_{\text{final}} = (\text{Score}_{\text{Stage 1}})^{\alpha} \cdot (\text{Confidence}_{\text{Stage 2}})^{1 - \alpha}$$
where $\alpha = 0.5$ balances global retrieval similarity with local moment regression.

### Benchmark Results (Honest 200-sample test set, `eval_e2e.py`)

| Run | R@1 IoU=0.5 | R@1 IoU=0.7 | Notes |
|---|---|---|---|
| 2,400-video subset, original loss | 0.3200 | 0.1600 | Previous subset baseline |
| 9,848-video full data, fixed conf loss (raw $\alpha=0$) | 0.2750 - 0.2950 | 0.1350 | Full data, no score fusion |
| Coarse retrieval only ($\alpha=1.0$) | 0.3150 | 0.1600 | No grounding head refinement |
| **9,848-video full data + Score Fusion ($\alpha=0.5$)** | **0.3500** | **0.1750** | **New Best Score! (+7.5% abs R@1@0.5, +4.0% abs R@1@0.7)** |

Cross-architecture comparison on 200 samples:
- **DETR v2 (`grounding_head_v2_confloss.pt`)**: **0.3500 / 0.1750** (Best overall)
- **Anchor Head (`grounding_anchor_v1_cropfix.pt`)**: **0.2150 / 0.1150** (Limited by discrete integer window scales)
- **Span Head (`grounding_span_v1_cropfix.pt`)**: **0.0850 / 0.0250** (Severely overfitted to position prior)

