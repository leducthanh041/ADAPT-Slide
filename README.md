# ADAPT-Slide

**ADAPT-Slide: Anchored Dual-Level Adaptation with Prompt and Normalization Tuning for Continual Model Merging in Whole-Slide Image Analysis**

ADAPT-Slide is a test-time adaptation framework for continual model merging in whole-slide image (WSI) analysis. It extends the MergeSlide inference pipeline by adapting the merged slide aggregator at test time while keeping the continual merging structure stable.

The default method updates only selected LayerNorm affine parameters in the TITAN slide aggregator. Merged weights, merge coefficients, class-aware prompts, task-level prompts, and the text encoder remain frozen unless an experimental branch explicitly enables otherwise.

## Requirements

Install the runtime stack used by this repo:

```text
numpy
pandas
scipy
scikit-learn
h5py
tqdm
omegaconf
torch
torchvision
transformers
matplotlib
seaborn
tensorboard
```

See `requirements.txt` for the full list.

## Dataset

The experiments use six TCGA WSI tasks:

- TCGA-BRCA
- TCGA-RCC
- TCGA-NSCLC
- TCGA-ESCA
- TCGA-TGCT
- TCGA-CESC

The WSI preprocessing and feature preparation pipeline follows the notebook:

```text
notebooks/WSI_processing.ipynb
```

This notebook documents how WSIs are prepared before running finetuning, merging, and inference in this project.

## Implementation

The implementation contains three main stages:

1. **Task-specific finetuning**: train one slide-level model per task.
2. **Continual model merging**: merge task-specific models into a unified model following the MergeSlide-style continual merging protocol.
3. **Test-time adaptation**: during inference, adapt the merged model with confidence-filtered dual-level objectives while updating only LayerNorm affine parameters by default.

Main entrypoints:

```bash
# Task-specific finetuning
bash scripts/finetune.sh

# Continual model merging
bash scripts/mergemodel.sh

# Class-IL TTA inference
bash scripts/test_classIL_tta.sh

# Task-IL TTA inference
bash scripts/test_taskIL_tta.sh
```

The Class-IL TTA script supports both TCP routing and naive Class-IL inference modes. Task-IL uses the known task identity and does not optimize TCP routing.

## Method Summary

ADAPT-Slide uses:

- confidence-filtered sub-bag selection;
- class-level and task-level entropy guidance;
- anchored regularization toward pre-TTA LayerNorm parameters;
- optional EMA teacher and task-prompt memory in the experimental prompt-adaptation branch;
- continual adaptation across slides by default.

The core research setting keeps adaptation lightweight and localized to normalization parameters, which reduces the risk of disrupting the merged model while improving robustness under domain shift.

## Acknowledgement

This project builds on the original MergeSlide codebase:

- https://github.com/caodoanh2001/MergeSlide

The slide aggregator and visual-language backbone follow TITAN:

- https://github.com/mahmoodlab/TITAN
