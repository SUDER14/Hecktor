"""Single seeding entry point so every run is reproducible end to end."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy, torch (CPU + CUDA) and MONAI's own RNG.

    MONAI's randomized dict transforms (RandCropByPosNegLabeld etc.) keep a
    separate RNG from numpy/torch, so set_determinism must be called too --
    seeding numpy and torch alone does not make those transforms reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from monai.utils import set_determinism

    set_determinism(seed=seed)
