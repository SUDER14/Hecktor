"""Generates notebooks/kaggle_runner.py (percent-format cells) and notebooks/colab_runner.ipynb
from one list of cells, so the two runners cannot drift apart.

    python scripts/make_runner_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMON = [
    ("md", "# ACF runner\nGPU check -> (optional) CUDA fix -> clone -> install -> data -> cache on fast local disk -> "
           "2-epoch timing probe -> full runs -> experiments.\n\nNothing in this notebook has been run on a GPU by the "
           "author of the repo yet: the timings it prints are the first real measurements."),
    ("code", "# 1. GPU check\n!nvidia-smi || echo 'NO GPU: switch the accelerator in the notebook settings'"),
    ("code", "# 2. CUDA-enabled torch? (must say True before any training)\nimport torch\n"
             "print(torch.__version__, '| built for CUDA', torch.version.cuda, '| available', torch.cuda.is_available())\n"
             "if torch.cuda.is_available():\n    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"),
    ("code", "# 2b. OPTIONAL: only if the cell above says False or the GPU is too old for the preinstalled build\n"
             "# (the 'CUDA-build fix'). Restart the runtime afterwards, then re-run cell 2.\n"
             "!pip install -q --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124"),
    ("code", "# 3. clone + install\nimport os\nREPO = 'https://github.com/SUDER14/Hecktor'\n"
             "if not os.path.exists('Hecktor'):\n    !git clone -q $REPO\n%cd Hecktor\n!git log --oneline -1\n"
             "!pip install -q monai nibabel SimpleITK scipy pandas scikit-image matplotlib pyyaml"),
    ("code", "# 4. DATA. Kaggle: add your dataset to the notebook (it appears under /kaggle/input), or download it\n"
             "# with a Kaggle API token stored as a notebook secret. Colab: mount Drive and point DATA at the folder.\n"
             "# Set SOURCE to the folder that contains the <CENTRE-NNN>/ patient directories.\n"
             "import os, shutil\nSOURCE = '/kaggle/input/REPLACE_WITH_YOUR_DATASET/hecktor2026_training'  # <-- edit\n"
             "# --- Kaggle secrets example (uncomment if downloading) ---\n"
             "# from kaggle_secrets import UserSecretsClient\n"
             "# s = UserSecretsClient(); os.environ['KAGGLE_USERNAME'] = s.get_secret('KAGGLE_USERNAME'); os.environ['KAGGLE_KEY'] = s.get_secret('KAGGLE_KEY')\n"
             "# !kaggle datasets download -d <owner>/<slug> -p /tmp/dl --unzip\n"
             "# --- Colab Drive example ---\n# from google.colab import drive; drive.mount('/content/drive'); SOURCE = '/content/drive/MyDrive/hecktor2026_training'"),
    ("code", "# 5. copy to fast LOCAL disk (/tmp), NOT /kaggle/working (synced, slow). Caches also live in /tmp.\n"
             "DATA = '/tmp/hecktor2026_training'\nif not os.path.exists(DATA):\n    shutil.copytree(SOURCE, DATA)\n"
             "print(len(os.listdir(DATA)), 'entries in', DATA)\n!df -h /tmp | tail -1"),
    ("code", "# 6. spacing manifest (header-only; shows whether spacing really varies by centre)\n"
             "!python scripts/spacing_manifest.py --data_dir $DATA --out results/manifest"),
    ("code", "# 7. TIMING PROBE: 2 epochs of the proposed model. Read 'time_data_s' vs 'time_gpu_s' to see the bottleneck.\n"
             "# RUNS must be on persistent storage so a dead session can --resume (Kaggle: /kaggle/working; Colab: Drive).\n"
             "RUNS = '/kaggle/working/runs'   # Colab: '/content/drive/MyDrive/acf_runs'\n"
             "OV = f'--override data_dir={DATA}'\n"
             "!python -m src.training.run --config configs/b4_acf_cond.yaml {OV} --override run.dir={RUNS}/probe "
             "--override train.n_epochs=2 --override train.val_every=1\n"
             "import pandas as pd; t = pd.read_csv(f'{RUNS}/probe/metrics.csv'); print(t[['epoch','time_data_s','time_gpu_s','time_total_s','peak_vram_mb']])\n"
             "print('projected 100-epoch hours (from the measured mean epoch time, validation excluded):', t.time_total_s.mean()*100/3600)"),
    ("code", "# 8. FULL RUNS (resumable: re-running this cell continues from the last checkpoint)\n"
             "for name in ['b1_unet_resample', 'b2_dual_resample', 'b3_acf_zeroed', 'b4_acf_cond', 'b5_acf_evidential']:\n"
             "    mode = 'resampled' if name in ('b1_unet_resample', 'b2_dual_resample') else 'native'\n"
             "    !python -m src.training.run --config configs/{name}.yaml {OV} --override run.dir={RUNS}/{name} "
             "--override cache.dir=/tmp/acf_cache_{mode} --resume"),
    ("code", "# 9. EXPERIMENTS (E1 held-out centre, E2 anisotropy, E3 resources, E4 calibration)\n"
             "R = ' '.join(f'{k}={RUNS}/{v}' for k, v in [('B1','b1_unet_resample'),('B2','b2_dual_resample'),('B3','b3_acf_zeroed'),('B4','b4_acf_cond'),('B5','b5_acf_evidential')])\n"
             "!python scripts/run_experiments.py e1 --runs {R} --out results/e1 --device cuda\n"
             "!python scripts/run_experiments.py e2 --runs {R} --out results/e2 --device cuda\n"
             "!python scripts/run_experiments.py e3 --runs {R} --out results/e3 --device cuda\n"
             "!python scripts/run_experiments.py e4 --runs B4={RUNS}/b4_acf_cond B5={RUNS}/b5_acf_evidential --out results/e4 --device cuda\n"
             "!python scripts/run_experiments.py e5 --per-case results/e1/e1_per_case.csv --pairs B2:B1 B4:B3 B5:B4 --out results/e5\n"
             "!python scripts/run_experiments.py e6 --per-case results/e1/e1_per_case.csv --reference B4 --out results/e6"),
]


def kaggle_py() -> str:
    out = ["# Kaggle runner: paste each '# %%' cell into a notebook cell (or import the file as a notebook).\n"
           "# Lines starting with '!' / '%' are notebook magics, so this file is not meant to be run as a plain script.\n"]
    for kind, text in COMMON:
        out.append("# %% [markdown]\n" + "\n".join("# " + ln for ln in text.splitlines()) if kind == "md" else "# %%\n" + text)
    return "\n\n".join(out) + "\n"


def colab_ipynb() -> dict:
    cells = []
    for i, (kind, text) in enumerate(COMMON):
        src = [ln + "\n" for ln in text.splitlines()]
        if src:
            src[-1] = src[-1].rstrip("\n")
        base = {"cell_type": "markdown" if kind == "md" else "code", "id": f"cell{i}", "metadata": {}, "source": src}
        if kind == "code":
            base.update({"execution_count": None, "outputs": []})
        cells.append(base)
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    nb = ROOT / "notebooks"
    nb.mkdir(exist_ok=True)
    (nb / "kaggle_runner.py").write_text(kaggle_py(), encoding="utf-8")
    (nb / "colab_runner.ipynb").write_text(json.dumps(colab_ipynb(), indent=1), encoding="utf-8")
    print("wrote", nb / "kaggle_runner.py", "and", nb / "colab_runner.ipynb")


if __name__ == "__main__":
    main()
