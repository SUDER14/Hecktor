"""Generates notebooks/01_tensor_walkthrough.ipynb and 02_uncertainty_walkthrough.ipynb.

Both run on the synthetic cohort (no GPU, no HECKTOR) and print every shape / formula value, so a
student can follow the data through the code. Run:  python scripts/make_walkthrough_notebooks.py
Execute (optional): jupyter nbconvert --to notebook --execute notebooks/01_tensor_walkthrough.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NB1 = [
    ("md", "# 01 Tensor walkthrough\nFollow one synthetic PET/CT case from files to network output, printing every shape.\n"
           "Synthetic data only: nothing here is a result."),
    ("code", "import sys, tempfile\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent) if Path.cwd().name == 'notebooks' else '.')\n"
             "import numpy as np, nibabel as nib, torch\n"
             "from src.data.synthetic import SynthConfig, write_synthetic_hecktor2026\n"
             "root = Path(tempfile.mkdtemp()) / 'hecktor2026_training'\n"
             "write_synthetic_hecktor2026(root, SynthConfig(fov_mm=(96, 96, 96), n_per_centre=1, seed=3))\n"
             "print(sorted(p.name for p in root.iterdir()))"),
    ("md", "## 1. Two scans, two grids\nSame field of view, different voxel size."),
    ("code", "for pid in ['CHUM-001', 'CHUP-001']:\n    for tag in ['CT', 'PT']:\n"
             "        img = nib.load(root / pid / f'{pid}__{tag}.nii.gz')\n"
             "        print(pid, tag, 'shape', img.shape, 'spacing mm', tuple(round(float(z), 2) for z in img.header.get_zooms()[:3]),\n"
             "              'extent mm', tuple(round(s * z) for s, z in zip(img.shape, img.header.get_zooms())))"),
    ("md", "## 2. A physical crop\nA 48 mm box: CT voxels = 48 / 1 = 48, PET voxels = 48 / 4 = 12 (CHUM). Both start at the same world position."),
    ("code", "from src.data.physical_crop import physical_crop, grid_from_affine\n"
             "pid = 'CHUM-001'\nct, pet, lab = (nib.as_closest_canonical(nib.load(root / pid / f)) for f in\n"
             "    [f'{pid}__CT.nii.gz', f'{pid}__PT.nii.gz', f'{pid}.nii.gz'])\n"
             "vols = {'ct': (np.asarray(ct.dataobj), ct.affine), 'pet': (np.asarray(pet.dataobj), pet.affine), 'label': (np.asarray(lab.dataobj), ct.affine)}\n"
             "centre = grid_from_affine(ct.shape, ct.affine).voxel_to_world(np.array(ct.shape) / 2)\n"
             "patches, info = physical_crop(vols, 'ct', (48, 48, 48), centre)\n"
             "for k, v in patches.items(): print(k, v.shape, 'start mm', np.round(info['start_mm'][k], 1), 'spacing', info['spacing'][k])"),
    ("md", "## 3. Into the network\n`ACF` takes CT (B,1,X,Y,Z), PET (B,1,x,y,z), the two conditioning vectors and the spacings."),
    ("code", "from src.models.acf import ACF\n"
             "m = ACF(channels=[8, 16, 32, 64, 128], pet_channels=[8, 16, 32]).eval()\n"
             "ct_t = torch.from_numpy(patches['ct'][None, None].astype('float32'))[..., :48, :48, :48]\n"
             "pet_t = torch.from_numpy(patches['pet'][None, None].astype('float32'))\n"
             "print('CT input', tuple(ct_t.shape), 'PET input', tuple(pet_t.shape))\n"
             "print(m.level_table((48, 48, 48), tuple(pet_t.shape[2:]), (1, 1, 1), (4, 4, 4)))"),
    ("md", "## 4. Shapes at every encoder level (hooks on the encoder blocks)"),
    ("code", "shapes = {}\n"
             "for name, enc in [('CT', m.ct_enc), ('PET', m.pet_enc)]:\n"
             "    for i, st in enumerate(enc.stages):\n"
             "        st.register_forward_hook(lambda mod, inp, out, n=name, i=i: shapes.__setitem__(f'{n} level {i+1}', tuple(out.shape)))\n"
             "cond = torch.zeros(1, 4)\n"
             "with torch.no_grad():\n"
             "    out = m(ct_t, pet_t, cond, cond, (1.0, 1.0, 1.0), (4.0, 4.0, 4.0))\n"
             "for k, v in shapes.items(): print(f'{k:14s}', v)\n"
             "print('output (B, K, X, Y, Z):', tuple(out.shape))"),
    ("md", "## 5. FiLM is the identity at initialisation"),
    ("code", "from src.models.hypernet import FiLMHypernet, film\n"
             "h = FiLMHypernet(4, [8, 16])\n(g0, b0), (g1, b1) = h(torch.randn(2, 4))\n"
             "print('gamma block 0', g0.shape, 'all ones:', bool((g0 == 1).all()), '| beta all zeros:', bool((b0 == 0).all()))\n"
             "x = torch.randn(2, 8, 4, 4, 4)\nprint('film(x) == x:', bool(torch.equal(film(x, g0, b0), x)))"),
    ("md", "## 6. Parameter counts"),
    ("code", "print('small ACF', sum(p.numel() for p in m.parameters()))\n"
             "print('CT hypernet', sum(p.numel() for p in m.ct_hyper.parameters()), '| PET hypernet', sum(p.numel() for p in m.pet_hyper.parameters()))"),
]

NB2 = [
    ("md", "# 02 Uncertainty walkthrough\nDirichlet quantities, the three losses, and calibration, with numbers you can check by hand."),
    ("code", "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent) if Path.cwd().name == 'notebooks' else '.')\n"
             "import numpy as np, torch\n"
             "from src.models.evidential import dirichlet_summary\n"
             "from src.training.losses import soft_dice_loss, evidential_ce, evidential_kl\n"
             "# 3 voxels, K=2 (bg, tumour); the worked example of docs/learn/06\n"
             "ev = torch.tensor([[0.2, 9.0], [0.4, 0.6], [12.0, 0.1]]).T.reshape(1, 2, 3)\n"
             "y = torch.tensor([[1, 1, 0]])\nprint('evidence', tuple(ev.shape), 'labels', tuple(y.shape))"),
    ("code", "d = dirichlet_summary(ev)\n"
             "for k in ['alpha', 'S', 'p', 'u']: print(k, tuple(d[k].shape), d[k].squeeze().numpy().round(4).tolist())"),
    ("code", "print('Dice loss        ', round(soft_dice_loss(d['p'], y).item(), 4), '(hand: 0.186)')\n"
             "print('evidential CE   ', round(evidential_ce(ev, y).item(), 4), '(hand: 0.333)')\n"
             "print('KL to uniform   ', round(evidential_kl(ev, y).item(), 4), '(hand: 0.024)')"),
    ("md", "## No evidence vs lots of evidence"),
    ("code", "for e in [(0, 0), (3, 0), (30, 0), (30, 30)]:\n"
             "    s = dirichlet_summary(torch.tensor(e, dtype=torch.float32).reshape(1, 2, 1, 1, 1))\n"
             "    print('evidence', e, '-> p', s['p'].flatten().numpy().round(3).tolist(), 'u', round(s['u'].item(), 3))"),
    ("md", "## Calibration\nA model that is always 90% confident but right 70% of the time has ECE about 0.2."),
    ("code", "from src.metrics.calibration import ece, reliability_bins\n"
             "rng = np.random.default_rng(0); n = 50_000\n"
             "prob = np.stack([np.full(n, 0.1), np.full(n, 0.9)])\n"
             "labels = (rng.random(n) < 0.7).astype(int)  # argmax is class 1; it is right when label == 1\n"
             "print('ECE', round(ece(prob, labels), 3))\n"
             "b = reliability_bins(prob, labels)\nprint('non-empty bins', int((b['count'] > 0).sum()), 'of', len(b['count']))"),
    ("md", "## Uncertainty should point at errors"),
    ("code", "from src.metrics.calibration import uncertainty_error_overlap\n"
             "label = (rng.random((20, 20, 20)) < 0.3).astype(int); pred = label.copy()\n"
             "flip = rng.random(label.shape) < 0.1; pred[flip] = 1 - pred[flip]\n"
             "good_u = flip * 0.9 + 0.05 * rng.random(label.shape); random_u = rng.random(label.shape)\n"
             "print('informative u:', {k: round(v, 3) for k, v in uncertainty_error_overlap(good_u, pred, label).items() if k in ('auroc', 'overlap_dice')})\n"
             "print('random u     :', {k: round(v, 3) for k, v in uncertainty_error_overlap(random_u, pred, label).items() if k in ('auroc', 'overlap_dice')})"),
]


def to_nb(cells) -> dict:
    out = []
    for i, (kind, text) in enumerate(cells):
        src = text.splitlines(keepends=True)
        c = {"cell_type": "markdown" if kind == "md" else "code", "id": f"c{i}", "metadata": {}, "source": src}
        if kind == "code":
            c.update({"execution_count": None, "outputs": []})
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    nb = ROOT / "notebooks"
    for name, cells in (("01_tensor_walkthrough.ipynb", NB1), ("02_uncertainty_walkthrough.ipynb", NB2)):
        (nb / name).write_text(json.dumps(to_nb(cells), indent=1), encoding="utf-8")
        print("wrote", nb / name)


if __name__ == "__main__":
    main()
