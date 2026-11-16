"""One test per finding of results/review/REVIEW.txt (2026-11-16). Each failed before its fix."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.eval.infer import sliding_window, sliding_window_dual

ROOT = Path(__file__).resolve().parents[1]


# R1 ------------------------------------------------------------------------------------------ double softmax
@pytest.mark.parametrize("dual", [False, True])
def test_softmax_head_probabilities_are_softmaxed_once(dual):
    """The windows are turned into probabilities before they are averaged; the average must not be pushed through
    softmax a second time. With logits (4, 0, 0) everywhere the answer is softmax(4, 0, 0) = (0.965, 0.018, 0.018);
    softmax applied twice gives (0.563, 0.218, 0.218): same argmax, so Dice hides it, but ECE, Brier, NLL, the
    temperature and u are all wrong."""
    logits = torch.tensor([4.0, 0.0, 0.0])
    want = torch.softmax(logits, 0).numpy()

    def out(x):
        return logits.view(1, 3, 1, 1, 1).expand(x.shape[0], 3, *x.shape[2:])

    if dual:
        vol, aff = np.zeros((20, 20, 20), np.float32), np.eye(4)
        res = sliding_window_dual(lambda c, p: out(c), vol, vol, aff, aff, (16, 16, 16), 3, "softmax", overlap=0.5)
    else:
        res = sliding_window(out, np.zeros((2, 20, 20, 20), np.float32), (16, 16, 16), 3, "softmax", overlap=0.5)
    np.testing.assert_allclose(res["p"][:, 3, 3, 3], want, atol=1e-6)
    np.testing.assert_allclose(res["p"].sum(0), 1.0, atol=1e-6)
    entropy = float(-(want * np.log(want)).sum() / np.log(3))
    np.testing.assert_allclose(res["u"], entropy, atol=1e-6)
    assert (res["pred"] == 0).all()


# R2 --------------------------------------------------------------------------------------------- lock bypass
def test_lock_cannot_be_bypassed_by_renaming_the_run_or_deleting_the_lock_file(tmp_path):
    from src.eval import locked_test as lt

    root = tmp_path / "test"
    ident = {"run": "runA", "commit": "c1", "plan_sha256": "p1", "checkpoint_sha256": "k1", "synthetic": False}
    assert lt.acquire_lock(root / "runA", root, ident, None) == "fresh"
    lock = json.loads((root / "runA" / "LOCK.json").read_text())
    (root / "runA" / "LOCK.json").write_text(json.dumps({**lock, "state": "finished"}))

    # (a) the same checkpoint under another run name (a copied or renamed run folder, another --name)
    with pytest.raises(SystemExit, match="already been evaluated on the test set as runA"):
        lt.acquire_lock(root / "runA_copy", root, {**ident, "run": "runA_copy"}, None)
    assert not (root / "runA_copy").exists()
    assert lt.acquire_lock(root / "runA_copy", root, {**ident, "run": "runA_copy"}, "second name on purpose, see notes") == "fresh"
    assert "runA_copy" in (root / "RERUN_LOG.txt").read_text()
    # a different checkpoint under a new name is an ordinary first run
    assert lt.acquire_lock(root / "runB", root, {**ident, "run": "runB", "checkpoint_sha256": "k2"}, None) == "fresh"

    # (b) results exist but LOCK.json was deleted: not a fresh run, and the old per-case files must not be reused
    (root / "runB" / "cases" / "test").mkdir(parents=True)
    (root / "runB" / "cases" / "test" / "X.npz").write_bytes(b"old")
    (root / "runB" / "LOCK.json").unlink()
    identb = {**ident, "run": "runB", "checkpoint_sha256": "k2"}
    with pytest.raises(SystemExit, match="holds results but no LOCK.json"):
        lt.acquire_lock(root / "runB", root, identb, None)
    assert lt.acquire_lock(root / "runB", root, identb, "lock file was removed by mistake") == "fresh"
    assert not (root / "runB" / "cases").exists() and len(list(root.glob("runB__superseded_*/cases/test/X.npz"))) == 1


# R3 ------------------------------------------------------------------------- machine paths in checkpoint config
MACHINE = ["split.path=/content/Hecktor/configs/splits/hecktor2026_seed42.json",
           "conditioning.stats_path=/content/Hecktor/configs/splits/hecktor2026_seed42_conditioning_stats.json",
           "cache.dir=/content/drive/MyDrive/acf_cache", "run.dir=/content/out/runs/x"]


def test_checkpoint_config_paths_of_the_training_machine_are_not_used_or_recorded(tmp_path):
    """scripts/train.py stores the config in best.pt with the training machine's absolute paths (split, statistics,
    cache). The test script must use this checkout's files and the --data-root it was given, and its manifest
    must not carry another machine's paths (or this machine's absolute folders)."""
    import yaml

    from src.eval import locked_test as lt
    from src.training.train import apply_overrides

    cfg = apply_overrides(yaml.safe_load((ROOT / "configs" / "shared_b1_unet_resample.yaml").read_text(encoding="utf-8")), MACHINE)
    local = lt.localise_config(cfg, Path("D:/somewhere/cache"))
    assert local["split"]["path"] == "configs/splits/hecktor2026_seed42.json"
    assert local["conditioning"]["stats_path"] == "configs/splits/hecktor2026_seed42_conditioning_stats.json"
    assert Path(local["cache"]["dir"]) == Path("D:/somewhere/cache") and local["run"]["dir"] is None and local["data_dir"] is None
    assert cfg["cache"]["dir"] == "/content/drive/MyDrive/acf_cache"          # the input is not modified
    assert lt.localise_config(cfg, None)["cache"]["dir"] is None
    with pytest.raises(SystemExit, match="not in this checkout"):
        lt.localise_config({**cfg, "split": {"path": "/content/other/my_split.json"}}, None)

    small = ["patch.spatial_size=[32, 32, 32]", "patch.min_extent_mm=[8.0, 8.0, 8.0]", "infer.sw_batch_size=4"]
    res = lt.evaluate(None, None, tmp_path, synthetic=True, config=ROOT / "configs" / "shared_b1_unet_resample.yaml",
                      overrides=small + MACHINE, device="cpu", n_synthetic=(2, 1))
    text = (res / "manifest.json").read_text(encoding="utf-8")
    for bad in ("/content", "E:/HECKTOR", "hecktor_cache", tmp_path.name):
        assert bad not in text, bad
    man = json.loads(text)
    assert man["checkpoint"] == "checkpoints/best.pt" and man["predictions_dir"] == f"test_predictions/{res.name}"
    assert man["config"]["split"]["path"] == "configs/splits/hecktor2026_seed42.json"


# R4 ------------------------------------------------------------------------------------- line-ending hashes
def test_text_hashes_do_not_depend_on_the_checkout_line_endings(tmp_path):
    """The same commit is checked out with LF on Colab and with CRLF on Windows (core.autocrlf=true). A hash of
    the raw bytes then differs for the same split file, and the test script would refuse a correctly trained
    run (or, worse, a habit of overriding the check would form). Text files are hashed after CRLF -> LF."""
    import hashlib
    import subprocess

    from src.eval import locked_test as lt

    lf = b'{\n  "train": ["a"],\n  "val": ["b"],\n  "test": ["c"]\n}\n'
    (tmp_path / "split_lf.json").write_bytes(lf)
    (tmp_path / "split_crlf.json").write_bytes(lf.replace(b"\n", b"\r\n"))
    (tmp_path / "split_other.json").write_bytes(lf.replace(b'"a"', b'"z"'))
    want = hashlib.sha256(lf).hexdigest()
    assert lt.sha256_text(tmp_path / "split_lf.json") == lt.sha256_text(tmp_path / "split_crlf.json") == want
    assert lt.sha256_text(tmp_path / "split_other.json") != want
    assert lt.sha256_file(tmp_path / "split_lf.json") != lt.sha256_file(tmp_path / "split_crlf.json")   # raw bytes do differ

    # a run trained from an LF checkout, scored from a CRLF checkout (and the reverse, as the Phase 6 trainer
    # recorded raw bytes): accepted. Different content: refused.
    run = tmp_path / "run"
    run.mkdir()
    for recorded in (want, hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()):
        (run / "manifest.json").write_text(json.dumps({"finished": True, "synthetic": False, "split_sha256": recorded,
                                                       "conditioning_stats_sha256": recorded}))
        for local in ("split_lf.json", "split_crlf.json"):
            assert lt._check_run_manifest(run, tmp_path / local, tmp_path / local)["finished"]
        with pytest.raises(SystemExit, match="is not the file the run was trained with"):
            lt._check_run_manifest(run, tmp_path / "split_other.json", tmp_path / "split_lf.json")

    # the plan hash is the same whether the plan was committed with LF or with CRLF bytes
    shas = []
    for i, eol in enumerate((b"\n", b"\r\n")):
        repo = tmp_path / f"repo{i}"
        (repo / "docs").mkdir(parents=True)
        for args in (["init", "-q"], ["config", "user.name", "t"], ["config", "user.email", "t@example.invalid"],
                     ["config", "core.autocrlf", "false"]):
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
        (repo / lt.PLAN).write_bytes(b"# plan" + eol + b"line two" + eol)
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "plan"], cwd=repo, check=True, capture_output=True)
        shas.append(lt.check_plan_committed(repo)["plan_sha256"])
    assert shas[0] == shas[1] == hashlib.sha256(b"# plan\nline two\n").hexdigest()
