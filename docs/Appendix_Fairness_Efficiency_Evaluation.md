# Appendix: Fairness and Efficiency Evaluation

This appendix reports the evaluation protocol and computational cost used to
compare MergeSlide-TTA with model merging and test-time adaptation baselines on
the same WSI benchmark.

## A. Fair Evaluation Protocol

All methods are evaluated under the same WSI continual learning protocol.

| Method | Same folds | Same fine-tuned checkpoints | Same patches / WSI | Source-free | Label-free | IND forward | OOD forward | IND reverse | Class-IL | Task-IL |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| MergeSlide | Yes | Yes | 400 | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| MergeSlide-TTA | Yes | Yes | 400 | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| AdaMerging++ | Yes | Yes | 400 | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| AdaRank | Yes | Yes | 400 | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Hi-Vec | Yes | Yes | 400 | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| MINGLE | Yes | Yes | 400 | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| CONCRETE | Yes | Yes | 400 | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| T3 / TCube | Yes | Yes | 400 | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| WEMoE | Yes | Yes | 400 | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

Notes:

- Fine-tuned checkpoints are reused from the same task-specific WSI training
  protocol.
- No method uses source training data during test-time adaptation.
- No method uses test labels during adaptation.
- Class-IL and Task-IL are used only during evaluation/inference.

## B. Main Performance Metrics

For each method, report mean and standard deviation over folds.

| Setting | Method | ACC | bACC | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| IND forward / Class-IL | MergeSlide |  |  |  |  |
| IND forward / Class-IL | MergeSlide-TTA |  |  |  |  |
| IND forward / Class-IL | AdaMerging++ |  |  |  |  |
| OOD forward / Class-IL | MergeSlide |  |  |  |  |
| OOD forward / Class-IL | MergeSlide-TTA |  |  |  |  |
| OOD forward / Class-IL | AdaMerging++ |  |  |  |  |
| IND reverse / Class-IL | MergeSlide |  |  |  |  |
| IND reverse / Class-IL | MergeSlide-TTA |  |  |  |  |
| IND reverse / Class-IL | AdaMerging++ |  |  |  |  |

Use the same table format for Task-IL.

## C. Test-Time Efficiency Metrics

This table measures test-time computational cost. `TTA steps` follows each
method's configured test-time adaptation procedure. MergeSlide-TTA is evaluated
with LayerNorm-only adaptation and `n_steps = 10`.

| Method | Updated object | Updated params | Update ratio | TTA steps | Backprop? | Time / slide | Throughput | Peak VRAM eval | Peak VRAM adapt |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| MergeSlide | None | 0 | 0.0000% | 0 | No |  |  |  | N/A |
| MergeSlide-TTA | LayerNorm affine `gamma, beta` |  |  | 10 | Yes |  |  |  |  |
| AdaMerging++ | Merge coefficients `lambda` |  |  | 20 | Yes |  |  |  |  |
| AdaRank | SVD/rank masks |  |  | 20 | Yes |  |  |  |  |
| Hi-Vec | Selected layer / propagated vector |  |  | 20 | Yes |  |  |  |  |
| MINGLE | Gating network |  |  | 20 | Yes |  |  |  |  |
| CONCRETE | Concrete mask + merge weights |  |  | 60 | Yes |  |  |  |  |
| T3 / TCube | Dynamic coefficient | 0 | 0.0000% | 0 | No |  |  |  |  |
| WEMoE | Router |  |  | 20 | Yes |  |  |  |  |

Recommended definitions:

- `Updated params`: number of parameters that receive gradients or are optimized
  during test-time adaptation.
- `Update ratio`: `updated params / total trainable-size backbone-or-adapter params`.
- `Time / slide`: wall-clock seconds per WSI, including adaptation and final
  inference when applicable.
- `Throughput`: `slides / second` or `slides / minute`.
- `Peak VRAM eval`: maximum CUDA memory during evaluation/inference.
- `Peak VRAM adapt`: maximum CUDA memory during test-time adaptation.

## D. Test-Time Data Budget

| Method | Patches / WSI | Adaptation data | Uses labels? | Uses source data? | Extra stored objects |
| --- | ---: | --- | --- | --- | --- |
| MergeSlide-TTA | 400 | Current test WSI / sub-bags | No | No | LN anchors |
| AdaMerging++ | 400 | Unlabeled test stream per task | No | No | Task vectors, lambdas |
| AdaRank | 400 | Unlabeled test stream per task | No | No | SVD components, masks |
| Hi-Vec | 400 | Unlabeled test stream | No | No | Layer task vectors |
| MINGLE | 400 | Seed samples from unlabeled test stream | No | No | Low-rank experts, gates |
| CONCRETE | 400 | Seed samples from unlabeled test stream | No | No | Concrete masks |
| T3 / TCube | 400 | Test WSI/batch for dynamic coefficient | No | No | Lambda cache |
| WEMoE | 400 | Unlabeled test stream | No | No | Experts, router |

## E. MergeSlide-TTA Diagnostics

These diagnostics explain why LayerNorm-only TTA improves prompt-aligned WSI
inference.

| Setting | Task routing accuracy before | Task routing accuracy after | Class entropy before | Class entropy after | Adapted / total slides | LN drift norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IND forward / Class-IL |  |  |  |  |  |  |
| OOD forward / Class-IL |  |  |  |  |  |  |
| IND reverse / Class-IL |  |  |  |  |  |  |

Suggested interpretation:

- Routing accuracy should improve or remain stable after TTA.
- Class entropy should decrease without collapsing to a single class.
- LN drift should remain small, showing that adaptation preserves the merged
  model structure.
- A high skip ratio is acceptable when confidence filtering identifies stable
  low-entropy slides.

## F. Method-Specific Diagnostics

These diagnostics are optional but useful for appendix analysis.

| Method | Diagnostic |
| --- | --- |
| AdaMerging++ | Lambda mean/std/min/max; entropy loss before/after adaptation |
| AdaRank | Mask sparsity; effective rank ratio; entropy loss before/after adaptation |
| Hi-Vec | Selected layer distribution; skipped/adapted ratio |
| MINGLE | Gate entropy; expert usage distribution; KL/entropy loss |
| CONCRETE | Mask sparsity; selected subspace ratio; inner/outer loss |
| T3 / TCube | Dynamic lambda mean/std; JS divergence distribution |
| WEMoE | Router entropy; expert balance; expert usage distribution |

## G. Reporting Recommendation

The main paper should include:

1. Main performance table: bACC, Macro F1, and Weighted F1.
2. Efficiency table: updated params, TTA steps, time/slide, throughput, and peak
   VRAM.
3. A compact protocol fairness statement: same folds, checkpoints, 400 patches,
   source-free adaptation, and label-free adaptation.

The appendix should include:

1. Full ACC/bACC/Macro F1/Weighted F1 for IND forward, OOD forward, and IND
   reverse under Class-IL and Task-IL.
2. Efficiency metrics for all methods.
3. MergeSlide-TTA diagnostics and method-specific diagnostics.
