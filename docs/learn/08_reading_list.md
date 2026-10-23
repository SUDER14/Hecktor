# 08 Reading list (15 papers)

Read in this order. "Verified" = I fetched the arXiv abstract page in this session; otherwise the title is from
memory and should be re-checked.

| # | Paper | arXiv | Why |
|---|---|---|---|
| 1 | Ronneberger, Fischer, Brox. U-Net | https://arxiv.org/abs/1505.04597 | the architecture |
| 2 | Cicek et al. 3D U-Net | https://arxiv.org/abs/1606.06650 | volumetric version |
| 3 | Isensee et al. nnU-Net | https://arxiv.org/abs/1809.10486 | the baseline that may win |
| 4 | Isensee et al. nnU-Net Revisited | https://arxiv.org/abs/2404.09556 | why well-tuned U-Nets still win |
| 5 | Iantsen et al. (HECKTOR PET/CT segmentation, squeeze-and-excitation normalisation) | https://arxiv.org/abs/2102.10446 | strong PET/CT head-and-neck baseline |
| 6 | Li et al. SwinCross | https://arxiv.org/abs/2302.03861 | cross-modal attention for PET/CT |
| 7 | Saeed et al. A Multimodal and Multi-centric Head and Neck Cancer Dataset for Segmentation, Diagnosis and Outcome Prediction | https://arxiv.org/abs/2509.00367 | the HECKTOR data (verified: 1,123 PET/CT scans, 10 centres) |
| 8 | Joutard, Pietsch, Prevost. HyperSpace | https://arxiv.org/abs/2407.03681 | **the prior art**: hypernetwork conditioned on spacing |
| 9 | You et al. Towards Voxel Spacing Consistency for Medical Image Segmentation (Consispace) | https://arxiv.org/abs/2606.31839 | verified: resampling to consistent axial spacing; competing/complementary approach |
| 10 | Diaz et al. Learning from spatially inhomogenous data: resolution-adaptive convolutions for multiple sclerosis lesion segmentation | https://arxiv.org/abs/2503.21829 | verified: fixed-physical-radius kernels, no resampling; another way to handle varying spacing |
| 11 | Ha, Dai, Le. HyperNetworks | https://arxiv.org/abs/1609.09106 | the hypernetwork idea |
| 12 | Perez et al. FiLM | https://arxiv.org/abs/1709.07871 | feature-wise modulation |
| 13 | Sensoy, Kaplan, Kandemir. Evidential Deep Learning | https://arxiv.org/abs/1806.01768 | Dirichlet uncertainty and the loss |
| 14 | Zou et al. TBraTS | https://arxiv.org/abs/2206.09309 | evidential segmentation in medical imaging |
| 15 | Guo et al. On Calibration of Modern Neural Networks | https://arxiv.org/abs/1706.04599 | ECE and reliability diagrams |

Papers 9 and 10 matter for the thesis: they show the spacing-heterogeneity problem is being attacked from other
angles (resampling, spacing-aware kernels). The related-work chapter must position ACF against all three of
HyperSpace, Consispace and Diaz et al. A systematic prior-art search has **not** been done yet (see
`docs/FINAL_REPORT.md`, open risks).

Reading strategy: papers 1, 3, 8 first (two evenings); then 11-13; then 5-7; finish with 14-15, 9-10, 4.
