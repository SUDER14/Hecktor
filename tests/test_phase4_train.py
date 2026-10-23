"""Train -> checkpoint -> resume -> evaluate on the synthetic cohort (CPU, tiny model)."""
import json

from src.data.conditioning import load_stats
from src.data.dual_cache import build_dual_datasets
from src.eval.evaluate_cases import evaluate_dataset
from src.training.run import train
from src.training.smoke import smoke_config


def test_train_resume_and_logs_for_evidential_acf(tmp_path):
    cfg = smoke_config(tmp_path, "acf_evidential", n_epochs=1)
    run = train(cfg)
    env = json.loads((run / "env.json").read_text())
    assert {"git_commit", "torch", "monai", "seed", "device"} <= set(env) and env["seed"] == 42
    assert (run / "config.yaml").exists() and (run / "checkpoints" / "epoch_0001.pt").exists()
    cfg["train"]["n_epochs"] = 2
    train(cfg, resume=True)
    rows = (run / "metrics.csv").read_text().strip().splitlines()
    assert len(rows) == 3 and rows[2].startswith("2,")      # epoch 2 appended, epoch 1 not repeated
    ds = build_dual_datasets(cfg)
    from src.training.run import build_model
    import torch
    m = build_model(cfg)
    m.load_state_dict(torch.load(run / "checkpoints" / "epoch_0002.pt", weights_only=False)["model_state"])
    df = evaluate_dataset(m, ds["test"], cfg, load_stats(cfg["conditioning"]["stats_path"]), max_cases=1,
                          degrade_z_mm=3.0, out_dir=run / "p", save_maps=True)
    assert len(df) == 1 and df.loc[0, "effective_sz_mm"] >= 3.0 and (run / "p").exists()
    assert df.attrs["reliability"]["count"].sum() > 0
