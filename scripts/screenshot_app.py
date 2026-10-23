"""Screenshots of the viewer.

    python scripts/screenshot_app.py --out results/app_screens            # figures, always works
    python scripts/screenshot_app.py --out results/app_screens --browser  # full page via Playwright

Default mode renders the three planes straight from app/viewer_core.py on the synthetic cohort
(no server, no browser). --browser additionally starts `streamlit run app/streamlit_app.py`
headless and captures the real page; it needs `pip install playwright && playwright install chromium`
and exits with a clear message if Playwright is missing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import viewer_core as vc  # noqa: E402
from src.data.synthetic import SynthConfig, write_synthetic_hecktor2026  # noqa: E402


def figures(out: Path) -> list[Path]:
    root = Path(tempfile.mkdtemp(prefix="acf_shot_")) / "d"
    write_synthetic_hecktor2026(root, SynthConfig(fov_mm=(64, 64, 64), n_per_centre=1, seed=11))
    case = vc.demo_prediction(vc.load_case_from_dir(root, "CHUP-001"))  # the 1x1x3 mm CT centre
    paths = []
    for plane in vc.PLANES:
        fig = vc.render(case, plane, vc.axis_length(case, plane) // 2)
        p = out / f"viewer_{plane}.png"
        fig.savefig(p, dpi=120)
        paths.append(p)
    print("metrics (DEMO heuristic on synthetic data, not a model result):", vc.case_metrics(case))
    return paths


def browser(out: Path, port: int = 8765) -> Path:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("Playwright is not installed: pip install playwright && playwright install chromium")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless", "true",
                             "--server.port", str(port), "--server.address", "localhost"])
    try:
        time.sleep(8)
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            page = b.new_page(viewport={"width": 1500, "height": 1000})
            page.goto(f"http://localhost:{port}")
            page.wait_for_selector("img, canvas", timeout=60000)
            time.sleep(3)
            p = out / "app_full_page.png"
            page.screenshot(path=str(p), full_page=True)
            b.close()
        return p
    finally:
        proc.terminate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/app_screens")
    ap.add_argument("--browser", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in figures(out):
        print("wrote", p)
    if a.browser:
        print("wrote", browser(out))


if __name__ == "__main__":
    main()
