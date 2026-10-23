# 03 Hypernetwork and FiLM

## Prior art first
Conditioning a segmentation U-Net on voxel spacing with a hypernetwork is **HyperSpace** (Joutard, Pietsch,
Prevost; MICCAI 2024; arXiv:2407.03681), single modality. Related spacing-handling work: Consispace
(arXiv:2606.31839, resampling to a consistent axial spacing) and Diaz et al. (arXiv:2503.21829, convolutions with
a fixed physical radius, no resampling). FiLM: Perez et al. 2018 (arXiv:1709.07871). Hypernetworks: Ha et al.
2016 (arXiv:1609.09106). This project's scope is the *per-modality* use and the measurements; nothing here is
claimed first.

## FiLM
For a feature map F of shape (B, C, X, Y, Z) and parameters gamma, beta of shape (B, C):

    F' = gamma * F + beta        (broadcast over X, Y, Z)

`hypernet.py:film`. One scale and one shift per channel per sample: the network can amplify or mute channels
depending on the scan.

## The hypernetwork (Tier 1) — `FiLMHypernet`
v (B, 4) -> Linear(4, 64) -> SiLU -> Linear(64, 64) -> SiLU -> Linear(64, 2*sum(C_i)) -> split into
gamma-delta and beta for every block of one encoder. gamma = 1 + delta_gamma, beta = delta_beta.

**Identity at init.** The last layer is zero-initialised, so gamma = 1 and beta = 0 exactly (tested by
`test_film_identity_at_init_and_exact_formula`). The model therefore starts as the unconditioned dual encoder;
conditioning is learned, not injected by random initial weights.

**Does zero init block learning?** No. At step 0 only the last layer gets gradient (hidden layers see zero
because the last layer's weights are zero); after one update the weights are non-zero and the hidden layers train
too (`test_gradient_reaches_both_hypernetworks`).

## Per-modality independence
`ACF.ct_hyper` and `ACF.pet_hyper` are separate modules with separate weights. CT is told its spacing, PET is
told its (very different) spacing, each modulating its own encoder. Parameter counts of the configured model:
CT hypernet 68,960; PET hypernet 19,040 (measured from the default `ACF()`).

## The ablation flag
`conditioning_zeroed=True` feeds zeros to the hypernetworks. Only their bias path remains, i.e. a constant
modulation that carries no information about *this* scan. B3 (zeroed) vs B4 (conditioned) isolates the value of
the acquisition information, with identical parameter counts (both 6,229,986).

## Tier 2 (off by default)
`HyperFuse1x1`: kernel = W_base + A_b @ C_b with low-rank (r = 4) factors generated from the joint vector.
Initialisation lesson from the tests: zeroing **both** factors gives both a zero gradient (d delta/dA = C = 0,
d delta/dC = A = 0) and the layer can never train, so only C starts at zero and A is small random.
Implement-and-measure only after Tier 1 results exist.

## Why FiLM and not dilation proportional to spacing
Settled earlier: dilation is integer-only while spacing ratios are continuous (4 mm / 1 mm is fine, 3 mm / 4 mm
is not), and dilated kernels sample a sparse lattice (gridding artefacts). FiLM handles continuous values.
