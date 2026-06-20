# STAD-Diff

**A Spatiotemporal Attention Diffusion Framework with Diagnosis-Aware Connectivity Learning for Functional MRI Restoration and Autism Spectrum Disorder Identification**

[![Paper](https://img.shields.io/badge/Paper-Preprint-blue)](#citation)
[![Dataset](https://img.shields.io/badge/Dataset-ABIDE-orange)](http://fcon_1000.projects.nitrc.org/indi/abide/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

Liang Tian, Tianhe Gu, Yingxi Wang, Xiaotong Wei, Dongfang Lv, Shiliang Shao, Xingyu Liu*, Dazhou Li*
(*Corresponding authors)

College of Computer Science and Technology, Shenyang University of Chemical Technology · China Medical University · Hospital of Liaoning University of Traditional Chinese Medicine · Shenyang Institute of Automation, CAS · The People's Hospital of Liaoning Province

---

## Overview

Functional MRI (fMRI) captures brain connectivity through the blood-oxygen-level-dependent (BOLD) signal, but fast clinical acquisition protocols sacrifice through-plane resolution, degrading the BOLD network dynamics that autism spectrum disorder (ASD) diagnosis depends on. Deterministic super-resolution models suppress the stochastic temporal fluctuations that encode functional connectivity, while standard diffusion models ignore the temporal axis entirely and require prohibitively long inference.

**STAD-Diff** is a conditional diffusion framework that restores 4D fMRI volumes while explicitly preserving the functional-connectivity biomarkers used for ASD identification. It couples spatial and temporal attention, supervises training with a differentiable connectivity-consistency objective, and accelerates inference to clinically practical speeds — all within a single backbone that also generalizes to CT and MRI super-resolution.

## Highlights

- 🧠 **Models BOLD network dynamics, not just pixels** — a learning framework built around the temporal structure of resting-state fMRI rather than frame-independent reconstruction.
- 🎯 **Diagnosis-aware training objective** — a differentiable connectivity-consistency loss aligns recovery directly with ASD biomarkers instead of optimizing pixel error alone.
- 🔀 **Hybrid spatiotemporal attention** — couples 3D CBAM (spatial) with temporal self-attention (cross-frame) in one architecture.
- ⚡ **14× faster inference** — observation-anchored deterministic DDIM sampling cuts inference to 10 steps with no quality trade-off.
- 🧩 **One backbone, many modalities** — lightweight configuration flags (`is_3d`, `enable_temporal`, `multimodal`) switch the same weights between 2D, 3D, and 4D inputs (fMRI, CT, multi-contrast MRI).
- 📊 **State-of-the-art results on 19-site ABIDE** — tCorr = 0.918, whole-brain FC-Sim = 0.867, ASD AUC = 0.843 (closing 83.5% of the gap to the ground-truth bound), with only 1.1 dB cross-site PSNR variation.

## Method

STAD-Diff is organized around five coupled components:

1. **Conditional diffusion probabilistic model** with cosine noise scheduling, conditioned on the low-resolution observation across both the forward and reverse processes.
2. **Accelerated inference scheme** combining deterministic DDIM sampling, observation-based conditional initialization, and non-uniform timestep allocation (60% of steps in the low-noise regime) — a 14× speedup over standard DDPM.
3. **Hybrid Attention Engine** in an enhanced conditional U-Net: 3D CBAM (channel + spatial attention) for cortical detail, a dedicated temporal self-attention module for cross-frame BOLD dynamics, and an optional Swin Transformer block for 2D tasks.
4. **Multi-modal condition fusion module** (SRUNet3D encoder + SRFusion: additive / concatenation / gated strategies) for incorporating auxiliary imaging data such as T2-weighted structural guidance.
5. **Diagnosis-aware connectivity-consistency learning** — a differentiable functional-connectivity operator (AAL-116 parcellation, Pearson correlation) embeds the FC estimate into the computational graph, supervising the network with a connectivity-consistency loss and a temporal-coherence regularizer alongside the standard diffusion loss.

```
L_total = L_simple + λ1 · L_FC + λ2 · L_temp
```

See the [paper](#citation) for full derivations (Sections 3–4).

## Results

### Comparison with state-of-the-art (ABIDE test set)

| Method | PSNR↑ | SSIM↑ | tSNR↑ | tCorr↑ | FC-Sim (ROI)↑ | AUC↑ | Time (s) |
|---|---|---|---|---|---|---|---|
| Bilinear interpolation | 29.83 | 0.782 | 5.61 | 0.695 | 0.652 | 0.721 | – |
| 3D U-Net | 34.34 | 0.911 | 7.15 | 0.831 | 0.815 | 0.801 | 0.22 |
| 3D RCAN | 34.72 | 0.918 | 7.18 | 0.835 | 0.822 | 0.794 | 0.45 |
| mGAN | 33.65 | 0.902 | 7.25 | 0.842 | 0.830 | 0.798 | 0.35 |
| DDPM (T=1000) | 33.91 | 0.905 | 7.35 | 0.854 | 0.842 | 0.793 | 18.52 |
| DDIM (S=10) | 34.25 | 0.912 | 7.32 | 0.861 | 0.848 | 0.796 | 0.71 |
| SR3 | 34.64 | 0.920 | 7.42 | 0.870 | 0.855 | 0.804 | 2.30 |
| Med-Diff | 35.15 | 0.928 | 7.51 | 0.882 | 0.866 | 0.821 | 1.45 |
| Fast-DDPM | 35.74 | 0.941 | 7.68 | 0.895 | 0.876 | 0.829 | 1.24 |
| **STAD-Diff (ours)** | **36.83** | **0.958** | **7.86** | **0.918** | **0.907** | **0.843** | 1.31 |

### Downstream ASD classification

| Method | ACC (%)↑ | AUC↑ | Precision (%)↑ | Recall (%)↑ |
|---|---|---|---|---|
| Low-res direct (no SR) | 63.4 | 0.701 | 65.9 | 75.1 |
| Bilinear interpolation | 67.8 | 0.721 | 69.4 | 77.2 |
| 3D U-Net reconstruction | 74.2 | 0.801 | 76.2 | 81.3 |
| SR3 reconstruction | 74.6 | 0.804 | 76.7 | 81.9 |
| **STAD-Diff (ours)** | **76.3** | **0.843** | **77.9** | **83.1** |
| GT upper bound | 78.9 | 0.871 | 82.3 | 85.6 |

### Cross-site generalization & efficiency

- Site-level PSNR ≥ 30.7 dB across all five held-out centers (NYU, UM, USM, UCLA, CMU), with **Δ-PSNR = 1.1 dB** — versus 1.7–1.9 dB for all baselines.
- **22M parameters** (62% smaller than a comparable 3D U-Net), **5.8 GB** training memory per GPU, **1.31 s** inference per volume.
- Generalizes to **low-dose CT denoising** (AAPM-Mayo, 41.24 dB PSNR) and **multi-contrast MRI super-resolution** (BraTS 2021, 35.62 dB PSNR) by switching configuration flags only — no architecture changes.

Full ablations, network-level functional connectivity analysis (DMN / SN / FPN), and discriminative brain-region saliency are reported in the paper (Sections 5.4–5.7).

## Dataset

Experiments use the multi-site **Autism Brain Imaging Data Exchange (ABIDE I)** dataset: 871 subjects (403 ASD, 468 typically developing controls) from 19 independent scanning centers across North America, Europe, and Asia.

- Preprocessing follows the standard **fMRIPrep** pipeline (motion correction, slice-timing correction, registration to MNI152, spatial smoothing, 0.01–0.1 Hz bandpass filtering, CompCor nuisance regression).
- Low-resolution inputs are generated via 6× trilinear downsampling along the slice-selection (*z*) axis, simulating the resolution loss of fast clinical acquisition.
- A strict **site-independent split** is used: 14 sites for training, 5 held-out sites (NYU, UM, USM, UCLA, CMU) for testing.

The dataset is publicly available via the ABIDE consortium: http://fcon_1000.projects.nitrc.org/indi/abide/

## Getting Started

> The training/evaluation scripts, configuration files, and pretrained weights referenced in the paper are provided in this repository. Please refer to the repository contents for exact entry points; a typical workflow looks like:

```bash
git clone https://github.com/2010156/STAD-Diff.git
cd STAD-Diff

# create environment
conda create -n stad-diff python=3.10
conda activate stad-diff
pip install -r requirements.txt
```

**Configuration flags** select the operating mode of the shared backbone:

| Flag | Effect | Task |
|---|---|---|
| `is_3d=True` | 3D convolutions, 3D group norm, 3D CBAM | 3D CT / 3D MRI / fMRI volumes |
| `enable_temporal=True` | Adds temporal self-attention at the bottleneck (requires `is_3d`) | 4D fMRI time series |
| `multimodal=True` | Activates SRUNet3D encoder + SRFusion module | Multi-modal conditional reconstruction |

Example commands (adjust paths/configs to match the repository):

```bash
# Train on ABIDE fMRI super-resolution
python train.py --config configs/abide_fmri.yaml

# Run accelerated (10-step DDIM) inference
python infer.py --config configs/abide_fmri.yaml --checkpoint checkpoints/stad_diff.pt --steps 10

# Evaluate (PSNR/SSIM/tCorr/FC-Sim + ASD classification)
python evaluate.py --config configs/abide_fmri.yaml --checkpoint checkpoints/stad_diff.pt
```

## Repository structure

```
STAD-Diff/
├── configs/        # Task configurations (fMRI / LDCT / BraTS MRI)
├── data/           # Dataset loading & preprocessing utilities
├── models/         # Conditional U-Net, Hybrid Attention Engine, SRFusion
├── losses/         # Diffusion, connectivity-consistency, temporal-coherence losses
├── scripts/        # Training / inference / evaluation entry points
└── checkpoints/    # Pretrained model weights
```

*(Reflects the structure described in the paper; see the actual repository tree for current file names.)*

## Key Results Summary

| Metric | Value |
|---|---|
| PSNR | 36.83 dB |
| SSIM | 0.958 |
| Temporal correlation (tCorr) | 0.918 |
| Whole-brain FC similarity | 0.867 |
| ASD classification AUC | 0.843 (GT bound: 0.871) |
| Cross-site Δ-PSNR | 1.1 dB |
| Inference speedup vs. DDPM | 14× |
| Parameters | 22M |

## Limitations & Future Work

Training and evaluation use synthetically degraded fMRI generated by retrospective trilinear downsampling, which does not fully replicate the *k*-space truncation and slice-profile effects of genuine thick-slice acquisition. Prospective validation on data acquired under real accelerated protocols is an important next step toward clinical translation. Planned extensions include other connectivity-dependent conditions (ADHD, major depressive disorder), structural/multi-contrast MRI reconstruction, and further inference acceleration via distillation.

## Citation

If you find this work useful, please cite:

```bibtex
@article{tian2025stad-diff,
  title   = {STAD-Diff: A Spatiotemporal Attention Diffusion Framework with Diagnosis-Aware
             Connectivity Learning for Functional MRI Restoration and Autism Spectrum
             Disorder Identification},
  author  = {Tian, Liang and Gu, Tianhe and Wang, Yingxi and Wei, Xiaotong and Lv, Dongfang
             and Shao, Shiliang and Liu, Xingyu and Li, Dazhou},
  journal = {Preprint submitted to Elsevier},
  year    = {2025}
}
```

## Code & Data Availability

- **Code:** https://github.com/2010156/STAD-Diff
- **ABIDE dataset:** http://fcon_1000.projects.nitrc.org/indi/abide/

## Contact

For questions about the method, please contact the corresponding authors:

- Xingyu Liu — Department of General Surgery, The People's Hospital of Liaoning Province — d6005@lnph.com
- Dazhou Li — College of Computer Science and Technology, Shenyang University of Chemical Technology — lidazhou@syuct.edu.cn

## License

This project is released under the [MIT License](LICENSE) (update to match the actual repository license).

## Acknowledgments

The authors thank the editors and anonymous reviewers for their constructive comments. Portions of the manuscript text were polished with the assistance of an AI writing tool; study design, analysis, and conclusions are solely the authors' own.
