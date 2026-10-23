# 05 Dirichlet (evidential) uncertainty

Source: Sensoy, Kaplan, Kandemir 2018 (arXiv:1806.01768); segmentation use e.g. TBraTS (arXiv:2206.09309).
Calibration measure: Guo et al. 2017 (arXiv:1706.04599).

## From softmax to evidence
Softmax gives probabilities that always sum to 1, even for a scan unlike anything seen. The evidential head
instead outputs K non-negative **evidence** values per voxel (`EvidentialHead`: 1x1x1 conv, softplus).

    alpha_k = e_k + 1     S = sum_k alpha_k     p_k = alpha_k / S     u = K / S

* No evidence: alpha = (1, ..., 1), p uniform, u = 1.
* Lots of evidence for one class: S large, u near 0, p near one-hot.
* u measures *how much total evidence*; p says *where it points*. Two voxels with p = (0.5, 0.5) can have
  u = 1 (don't know) or u = 0.2 (genuinely ambiguous, lots of evidence on both sides).

K = 2 (background, tumour), configurable.

## One forward pass
Mask = argmax p. Uncertainty map = u. No sampling, no ensemble. For softmax baselines `softmax_summary` gives a
normalised predictive entropy so evaluation code is head-agnostic; that is a *different* quantity from vacuity.

## Training it
Loss (chapter 06): evidential cross-entropy `sum_k y_k (digamma(S) - digamma(alpha_k))` pushes evidence up for
the true class; an annealed KL term to the uniform Dirichlet, on alpha~ = y + (1-y)*alpha (true-class evidence
removed), pushes misleading evidence for *wrong* classes back to zero. Weight min(1, epoch / anneal_epochs):
let the network fit first, then regularise.

## Averaging in sliding-window inference
Overlapping windows are combined by averaging **evidence** and then deriving alpha, p, u (`infer.py`).
Averaging u directly would not correspond to any Dirichlet.

## How uncertainty quality is measured (`src/metrics/calibration.py`)
* ECE, 15 bins, over all voxels and over an ROI (dilated union of label and prediction, since easy background
  flatters ECE).
* Uncertainty-vs-error: AUROC of u as a detector of wrong voxels; Dice between the k most uncertain voxels and the
  k errors.
Neither says the uncertainty is *useful clinically*; they say it ranks errors.

## Scope note
Uncertainty here is delineation-model uncertainty. It is **not** converted into a CTV margin (settled: CTV is a
biological margin; delineation uncertainty belongs to PTV; any margin display is decision support only).
