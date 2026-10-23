# Experiment configs

Every run is driven by one YAML file; nothing is hard-coded in `src/`. Run with
`python -m src.training.run --config configs/<file>.yaml` (add `--resume` after a Colab/Kaggle
session dies; `--override train.n_epochs=2` changes a value that already exists in the file).

| Config | Thesis label | Input | Model | Conditioning | Head |
|---|---|---|---|---|---|
| `b1_unet_resample.yaml` | **B1** single-modality baseline | CT only, resampled to 1 mm | `UNet3D` | none | softmax |
| `b2_dual_resample.yaml` | **B2** dual-encoder baseline | CT + PET, both resampled to 1 mm | `DualEncoderUNet` (concat + 1x1x1 fusion) | none | softmax |
| `b3_acf_zeroed.yaml` | **B3** ACF, conditioning ablated | CT + PET at native spacing | `ACF` | hypernetwork present, input vector zeroed | softmax |
| `b4_acf_cond.yaml` | **B4** ACF (proposed) | CT + PET at native spacing | `ACF` | per-modality FiLM hypernetworks fed `[sx, sy, sz, flag]` | softmax |
| `b5_acf_evidential.yaml` | **B5** ACF + uncertainty | CT + PET at native spacing | `ACF` | as B4 | evidential (Dirichlet) |

Comparisons these support: B1 vs B2 = what a second modality buys; B2 vs B3 = what native
spacing + level-matched fusion buys (no conditioning information); B3 vs B4 = what acquisition
conditioning buys; B4 vs B5 = what the evidential head buys in calibration/uncertainty.

Prior art: spacing-conditioning of a U-Net by a hypernetwork is HyperSpace (Joutard et al.,
MICCAI 2024, arXiv:2407.03681). B3 vs B4 measures whether it carries over to two modalities.

Values marked as placeholders in each file (`split.test_centres`, `intensity.pet.suv_clip_max`)
must be confirmed against the real HECKTOR data once available. `n_epochs: 100` is a plan, not a
measured requirement. Test-centre choice must not be only tiny centres (HMR, USZ).

Other configs: `synth_native.yaml` (synthetic cohort), `cpu_smoke*.yaml`, `*decathlon*` (earlier
CT-only Decathlon baseline work).
