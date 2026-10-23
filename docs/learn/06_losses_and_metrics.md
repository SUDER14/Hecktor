# 06 Losses and metrics — worked 3-voxel example

Three voxels, K = 2 (class 0 background, class 1 tumour). Evidence per voxel (bg, tumour) and true class:

| voxel | evidence (e_bg, e_tum) | true class |
|---|---|---|
| 1 | (0.2, 9.0) | tumour |
| 2 | (0.4, 0.6) | tumour |
| 3 | (12, 0.1) | background |

Tested in `tests/test_phase3_losses_metrics.py`; the values below were also reproduced numerically before the
code was written.

## Step 1: Dirichlet quantities
alpha = e + 1: v1 (1.2, 10), S = 11.2; v2 (1.4, 1.6), S = 3.0; v3 (13, 1.1), S = 14.1.
Tumour probability p_tum = alpha_tum / S: v1 = 0.8929, v2 = 0.5333, v3 = 0.0780.
Vacuity u = 2/S: 0.179, 0.667, 0.142. Voxel 2 is the uncertain one.

## Step 2: soft Dice (tumour class only)
intersection = sum(p_tum * y_tum) = 0.8929 + 0.5333 = 1.4262
sum p_tum = 1.5042, sum y_tum = 2
Dice = 2 * 1.4262 / (1.5042 + 2) = 0.8140, **Dice loss = 1 - 0.8140 = 0.186**.
Background is excluded from the average because a tumour-free crop would otherwise be dominated by it.

## Step 3: evidential cross-entropy
per voxel `digamma(S) - digamma(alpha_true)`: v1 digamma(11.2) - digamma(10); v2 digamma(3) - digamma(1.6);
v3 digamma(14.1) - digamma(13). Mean over the three voxels = **0.333**. Bigger when evidence for the true class
is small relative to S (voxel 2 contributes most).

## Step 4: KL regulariser
alpha~ = y + (1 - y) * alpha: keep alpha for the wrong class, set the true class to 1 (order: bg, tumour):
v1 (1.2, 1), v2 (1.4, 1), v3 (1, 1.1). KL(Dir(alpha~) || Dir(1,1)) is 0 only if alpha~ = (1,1); here it is small
because the wrong-class evidence is small. Mean over voxels = **0.0236**. With annealing weight w = min(1, epoch /
anneal_epochs) the total is CE + w * KL (+ Dice).

## Other losses
`dice_ce_loss`: softmax Dice + cross-entropy on logits; `smooth` stops an empty crop giving an undefined 0/0.

## Metrics (all in `src/metrics`)
* **HD95**: surface voxels (mask minus its erosion), distance transform with the voxel spacing so units are mm
  and anisotropy is exact; larger of the two directed 95th percentiles. No `torch.quantile` (it fails on big
  tensors).
* **Surface Dice @ 2 / 3 mm**: fraction of both surfaces within tolerance of the other.
* **ECE**: bins on confidence max_k p_k; sum over bins of (n_bin / N) * |accuracy - confidence|.
* **Empty-mask convention** (one rule for every metric): reference empty -> NaN (excluded, and the count of
  excluded cases is reported); prediction empty but reference not -> Dice 0, Surface Dice 0, HD95 = the volume's
  physical diagonal, flagged `pred_empty`.

## Reading a result honestly
Report the number of NaN-excluded and empty-prediction cases next to every mean; report parameters and the
git hash. A mean Dice without them hides the failure cases.
