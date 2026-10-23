# Kaggle runner: paste each '# %%' cell into a notebook cell (or import the file as a notebook).
# Lines starting with '!' / '%' are notebook magics, so this file is not meant to be run as a plain script.


# %% [markdown]
# # ACF runner
# GPU check -> (optional) CUDA fix -> clone -> install -> data -> cache on fast local disk -> 2-epoch timing probe -> full runs -> experiments.
# 
# Nothing in this notebook has been run on a GPU by the author of the repo yet: the timings it prints are the first real measurements.

# %%
# 1. GPU check
!nvidia-smi || echo 'NO GPU: switch the accelerator in the notebook settings'

# %%
# 2. CUDA-enabled torch? (must say True before any training)
import torch
print(torch.__version__, '| built for CUDA', torch.version.cuda, '| available', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

# %%
# 2b. OPTIONAL: only if the cell above says False or the GPU is too old for the preinstalled build
# (the 'CUDA-build fix'). Restart the runtime afterwards, then re-run cell 2.
!pip install -q --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124

# %%
# 3. clone + install
import os
REPO = 'https://github.com/SUDER14/Hecktor'
if not os.path.exists('Hecktor'):
    !git clone -q $REPO
%cd Hecktor
!git log --oneline -1
!pip install -q monai nibabel SimpleITK scipy pandas scikit-image matplotlib pyyaml

# %%
# 4. DATA. Kaggle: add your dataset to the notebook (it appears under /kaggle/input), or download it
# with a Kaggle API token stored as a notebook secret. Colab: mount Drive and point DATA at the folder.
# Set SOURCE to the folder that contains the <CENTRE-NNN>/ patient directories.
import os, shutil
SOURCE = '/kaggle/input/REPLACE_WITH_YOUR_DATASET/hecktor2026_training'  # <-- edit
# --- Kaggle secrets example (uncomment if downloading) ---
# from kaggle_secrets import UserSecretsClient
# s = UserSecretsClient(); os.environ['KAGGLE_USERNAME'] = s.get_secret('KAGGLE_USERNAME'); os.environ['KAGGLE_KEY'] = s.get_secret('KAGGLE_KEY')
# !kaggle datasets download -d <owner>/<slug> -p /tmp/dl --unzip
# --- Colab Drive example ---
# from google.colab import drive; drive.mount('/content/drive'); SOURCE = '/content/drive/MyDrive/hecktor2026_training'

# %%
# 5. copy to fast LOCAL disk (/tmp), NOT /kaggle/working (synced, slow). Caches also live in /tmp.
DATA = '/tmp/hecktor2026_training'
if not os.path.exists(DATA):
    shutil.copytree(SOURCE, DATA)
print(len(os.listdir(DATA)), 'entries in', DATA)
!df -h /tmp | tail -1

# %%
# 6. spacing manifest (header-only; shows whether spacing really varies by centre)
!python scripts/spacing_manifest.py --data_dir $DATA --out results/manifest

# %%
# 7. TIMING PROBE: 2 epochs of the proposed model. Read 'time_data_s' vs 'time_gpu_s' to see the bottleneck.
# RUNS must be on persistent storage so a dead session can --resume (Kaggle: /kaggle/working; Colab: Drive).
RUNS = '/kaggle/working/runs'   # Colab: '/content/drive/MyDrive/acf_runs'
OV = f'--override data_dir={DATA}'
!python -m src.training.run --config configs/b4_acf_cond.yaml {OV} --override run.dir={RUNS}/probe --override train.n_epochs=2 --override train.val_every=1
import pandas as pd; t = pd.read_csv(f'{RUNS}/probe/metrics.csv'); print(t[['epoch','time_data_s','time_gpu_s','time_total_s','peak_vram_mb']])
print('projected 100-epoch hours (from the measured mean epoch time, validation excluded):', t.time_total_s.mean()*100/3600)

# %%
# 8. FULL RUNS (resumable: re-running this cell continues from the last checkpoint)
for name in ['b1_unet_resample', 'b2_dual_resample', 'b3_acf_zeroed', 'b4_acf_cond', 'b5_acf_evidential']:
    mode = 'resampled' if name in ('b1_unet_resample', 'b2_dual_resample') else 'native'
    !python -m src.training.run --config configs/{name}.yaml {OV} --override run.dir={RUNS}/{name} --override cache.dir=/tmp/acf_cache_{mode} --resume

# %%
# 9. EXPERIMENTS (E1 held-out centre, E2 anisotropy, E3 resources, E4 calibration)
R = ' '.join(f'{k}={RUNS}/{v}' for k, v in [('B1','b1_unet_resample'),('B2','b2_dual_resample'),('B3','b3_acf_zeroed'),('B4','b4_acf_cond'),('B5','b5_acf_evidential')])
!python scripts/run_experiments.py e1 --runs {R} --out results/e1 --device cuda
!python scripts/run_experiments.py e2 --runs {R} --out results/e2 --device cuda
!python scripts/run_experiments.py e3 --runs {R} --out results/e3 --device cuda
!python scripts/run_experiments.py e4 --runs B4={RUNS}/b4_acf_cond B5={RUNS}/b5_acf_evidential --out results/e4 --device cuda
!python scripts/run_experiments.py e5 --per-case results/e1/e1_per_case.csv --pairs B2:B1 B4:B3 B5:B4 --out results/e5
!python scripts/run_experiments.py e6 --per-case results/e1/e1_per_case.csv --reference B4 --out results/e6
