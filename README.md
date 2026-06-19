# STAD-Diff

**STAD-Diff: A Spatiotemporal Attention Diffusion Framework with Diagnosis-Aware
Connectivity Learning for Functional MRI Restoration and Autism Spectrum Disorder
Identification**

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Preprint-blue)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org)

</div>

---

## Overview

STAD-Diff is a unified conditional diffusion framework that achieves high-fidelity **4D fMRI super-resolution** while preserving the BOLD signal dynamics required for reliable functional connectivity analysis and downstream clinical tasks such as **Autism Spectrum Disorder (ASD) diagnosis**.

<div align="center">

| Metric | Value |
|--------|-------|
| PSNR | **36.83 dB** |
| SSIM | **0.958** |
| tCorr | **0.918** |
| Whole-brain FC-Sim | **0.867** |
| ASD Classification AUC | **0.843** |
| Inference speedup vs. DDPM | **14×** |
| Model parameters | **22M** |

</div>

---

## Key Contributions

**1. Hybrid Attention Engine**
Couples 3D Convolutional Block Attention Modules (CBAM) with a dedicated temporal self-attention module to jointly recover spatial cortical detail and cross-frame BOLD coherence within a single architecture.

**2. Accelerated Inference**
Combines observation-based conditional initialization (+1.2 dB PSNR) with a non-uniform timestep allocation strategy (+0.49 dB PSNR), enabling deterministic 10-step DDIM inference — a **14× speedup** over standard DDPM.

**3. Unified Multi-Modal Framework**
A single set of weights handles 2D, 3D, and 4D inputs via lightweight configuration flags (`is_3d`, `enable_temporal`, `multimodal`), covering CT denoising, MRI super-resolution, and fMRI temporal reconstruction without architectural modification.

**4. Clinically Validated**
Evaluated not only on image quality metrics but also through a downstream ASD classification task on the multi-site ABIDE dataset (871 subjects, 17 sites), closing **83.5%** of the performance gap between low-resolution baseline and ground-truth upper bound.

---

## Architecture

```
Low-Resolution fMRI  →  [Conditional U-Net + Hybrid Attention Engine]  →  High-Resolution fMRI
                                                                                    ↓
                                        Functional Connectivity Matrix  →  ASD Classification
```

The core components of STAD-Diff:

- **Conditional U-Net backbone** with depthwise separable 3D convolutions (62% parameter reduction vs. standard 3D U-Net)
- **3D CBAM** (channel + spatial attention) for functionally active region amplification
- **Temporal self-attention module** capturing cross-frame long-range dependencies (0.01–0.1 Hz BOLD oscillations)
- **Cosine noise schedule** with observation-based conditional initialization
- **Non-uniform timestep allocation** (60% steps in low-noise regime [0, 0.7T])
- **SRFusion module** for optional multi-modal condition fusion

---

## Results

### Image Quality & Functional Fidelity (ABIDE Test Set)

| Method | PSNR (dB) | SSIM | tCorr | FC-Sim (ROI) | AUC | Time (s) |
|--------|-----------|------|-------|--------------|-----|----------|
| Bilinear interpolation | 29.83 | 0.782 | 0.695 | 0.652 | 0.721 | — |
| 3D U-Net | 34.34 | 0.911 | 0.831 | 0.815 | 0.791 | 0.22 |
| 3D RCAN | 34.72 | 0.918 | 0.835 | 0.822 | 0.794 | 0.45 |
| DDPM (T=1000) | 33.91 | 0.905 | 0.854 | 0.842 | 0.806 | 18.52 |
| SR3 | 34.64 | 0.920 | 0.870 | 0.855 | 0.813 | 2.30 |
| Fast-DDPM | 35.74 | 0.941 | 0.895 | 0.876 | 0.829 | 1.24 |
| **STAD-Diff (ours)** | **36.83** | **0.958** | **0.918** | **0.907** | **0.843** | **1.31** |
| GT upper bound | — | — | — | — | 0.871 | — |

### ASD Classification (ABIDE Test Set)

| Method | ACC (%) | AUC | F1 (%) |
|--------|---------|-----|--------|
| Low-res direct (no SR) | 63.4 | 0.701 | 72.4 |
| **STAD-Diff (ours)** | **76.3** | **0.843** | **80.4** |
| GT upper bound | 78.9 | 0.871 | 83.9 |

### Cross-Site Generalization (Δ-PSNR)

| Method | NYU | UM | USM | UCLA | CMU | Δ-PSNR ↓ |
|--------|-----|----|-----|------|-----|----------|
| 3D U-Net | 29.1 | 29.8 | 28.3 | 29.4 | 27.9 | 1.9 |
| SR3 | 28.9 | 29.6 | 28.1 | 29.3 | 27.8 | 1.8 |
| **STAD-Diff** | **31.2** | **31.8** | **30.9** | **31.5** | **30.7** | **1.1** |

---

## Installation

```bash
git clone https://github.com/[your-org]/STAD-Diff.git
cd STAD-Diff
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.8
- PyTorch 2.1 with CUDA
- Ubuntu 22.04 (recommended)
- NVIDIA GPU with ≥ 8 GB VRAM (A100 recommended for training)

---

## Data Preparation

This project uses the publicly available **ABIDE (Autism Brain Imaging Data Exchange)** dataset.

1. Download the preprocessed ABIDE dataset from:
   ```
   http://fcon_1000.projects.nitrc.org/indi/abide/
   ```

2. Apply the fMRIPrep preprocessing pipeline (motion correction, MNI registration, bandpass filtering 0.01–0.1 Hz, nuisance regression).

3. Generate super-resolution training pairs by applying 6× trilinear downsampling along the z-axis:
   ```bash
   python scripts/generate_lr_pairs.py --data_dir /path/to/abide --scale 6
   ```

4. Use the provided site-independent train/test split (14 sites train, 5 sites test: NYU, UM, USM, UCLA, CMU).

---

## Training

```bash
# Single-GPU training
python train.py \
  --data_dir /path/to/abide \
  --modality fmri \
  --base_channels 128 \
  --diffusion_steps 1000 \
  --batch_size 16 \
  --lr 2e-4 \
  --epochs 300

# Multi-GPU distributed training (recommended)
torchrun --nproc_per_node=4 train.py \
  --data_dir /path/to/abide \
  --modality fmri \
  --batch_size 16 \
  --enable_temporal True \
  --is_3d True
```

Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| Base channels | 128 |
| Channel multipliers | [1, 2, 3, 4] |
| Residual blocks per stage | 2 |
| Diffusion timesteps T | 1000 |
| DDIM inference steps S | 10 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning rate | 2×10⁻⁴ (cosine annealing) |
| EMA decay | 0.9999 |

---

## Inference

```bash
python inference.py \
  --checkpoint /path/to/checkpoint.pth \
  --input /path/to/lr_fmri.nii.gz \
  --output /path/to/hr_fmri.nii.gz \
  --ddim_steps 10 \
  --enable_temporal True
```

Inference completes in **~1.31 seconds** per 128×256×256 volume on a single NVIDIA A100 GPU.

---

## Modality Configuration

STAD-Diff supports multiple imaging tasks with the same weights via configuration flags:

```python
# 4D fMRI super-resolution (primary use case)
model = UniMedDiff(is_3d=True, enable_temporal=True, multimodal=False)

# 3D MRI super-resolution
model = UniMedDiff(is_3d=True, enable_temporal=False, multimodal=False)

# 2D CT denoising
model = UniMedDiff(is_3d=False, enable_temporal=False, multimodal=False)

# Multi-modal conditional reconstruction
model = UniMedDiff(is_3d=True, enable_temporal=True, multimodal=True)
```

---

## ASD Downstream Classification

After obtaining super-resolved fMRI volumes, run the ASD classification pipeline:

```bash
python classify_asd.py \
  --fmri_dir /path/to/sr_fmri \
  --atlas AAL \
  --classifier linear_svm \
  --cv_folds 5
```

This extracts a 116×116 functional connectivity matrix (AAL atlas parcellation) and trains a linear SVM on the 6670 upper-triangular FC features.

---

## Ablation Study Results

| Configuration | tCorr | AUC |
|---------------|-------|-----|
| w/o CBAM attention | 0.871 | 0.796 |
| w/o 3D conv (2D only) | 0.875 | 0.804 |
| w/o temporal attention | 0.886 | 0.809 |
| w/o conditional init | 0.889 | 0.816 |
| w/o non-uniform schedule | 0.894 | 0.823 |
| **Complete STAD-Diff** | **0.918** | **0.843** |

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{tian2025unimediff,
  title={STAD-Diff: STAD-Diff: A Spatiotemporal Attention Diffusion Framework with Diagnosis-Aware
Connectivity Learning for Functional MRI Restoration and Autism Spectrum Disorder
Identification},
  author={Tian, Liang and Gu, Tianhe and Wang, Yingxi and Wei, Xiaotong and Lv, Dongfang and Liu, Xingyu and Li, Dazhou},
  journal={Preprint submitted to Elsevier},
  year={2025}
}
```

---

## Funding

This work was supported by:
- Scientific Research Project of Liaoning Provincial Department of Education (No. LJ212510149013)
- General Program of the National Natural Science Foundation of Liaoning Province (No. 2024-MSLH-377)
- Liaoning Province Doctoral Research Start-up Fund Program (No. 2025-BS-0604)

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

We thank the ABIDE consortium for providing the publicly available multi-site neuroimaging dataset. We also thank the editors and anonymous reviewers for their constructive comments.

> **Note on AI assistance:** During manuscript preparation, Claude Sonnet 4.6 (Anthropic) was used for language polishing and grammatical refinement only. All scientific content, experimental design, and conclusions are the sole responsibility of the authors.
