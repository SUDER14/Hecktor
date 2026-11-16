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
