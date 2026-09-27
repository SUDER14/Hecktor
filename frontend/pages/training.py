"""Page 3: training curves. Placeholder until Colab runs exist.

Expected (to be wired later): one folder per run under results/runs/<run_name>/
holding a metrics.csv with columns epoch, train_loss and optionally val_loss,
val_dice, val_hd95 -- the same keys train.py logs to wandb. Nothing writes
that file yet, so this page shows the "no runs yet" state.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from common import RESULTS_DIR, read_csv

st.title("Training progress")

runs_dir = RESULTS_DIR / "runs"
metric_files = sorted(runs_dir.glob("*/metrics.csv")) if runs_dir.exists() else []

if not metric_files:
    st.info(
        "**No runs yet.** Training happens on Colab; once runs exist, this page will plot "
        "loss and Dice per run from `results/runs/<run>/metrics.csv` (or wandb). "
        "CPU smoke-test logs in `results/logs/` are deliberately not shown here: they are "
        "pipeline checks, not results."
    )
    st.stop()

selected = st.multiselect(
    "Runs", metric_files, default=metric_files, format_func=lambda p: p.parent.name
)
for column, label in [("train_loss", "Train loss"), ("val_loss", "Val loss"), ("val_dice", "Val Dice")]:
    frames = {}
    for path in selected:
        df = read_csv(str(path))
        if "epoch" in df and column in df:
            frames[path.parent.name] = df.set_index("epoch")[column]
    if frames:
        st.subheader(label)
        st.line_chart(pd.DataFrame(frames))
