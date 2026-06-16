#!/usr/bin/env python3
"""Record an annotated demo GIF of the edu-synth Streamlit app.

Usage:
    uv run python scripts/record-demo.py

Prerequisites (one-time):
    uv sync --all-extras
    uv run playwright install chromium
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
VIDEO_TMP = ROOT / "scripts" / "video-tmp"
GIF_OUT = ROOT / "docs" / "assets" / "demo.gif"
FIXTURE_CSV = ROOT / "tests" / "fixtures" / "mini_inschrijving.csv"
APP_URL = "http://localhost:8502"  # 8502 to avoid conflicts with dev server


# ── annotation helpers ─────────────────────────────────────────────────────────

# Playwright evaluate passes the second arg as the first JS function parameter.
_ANN_JS = """
({ selector, label, side, color }) => {
    document.querySelectorAll('.__ann').forEach(n => n.remove())

    if (!document.querySelector('#__ann-style')) {
        const s = document.createElement('style')
        s.id = '__ann-style'
        s.textContent = `
            @keyframes ann-in    { from { opacity:0; transform:scale(.85) } to { opacity:1; transform:scale(1) } }
            @keyframes ann-pulse { 0%,100% { box-shadow:0 0 0 0 ${color}55 } 50% { box-shadow:0 0 0 8px transparent } }
            .__ann-ring  { animation: ann-in .25s ease, ann-pulse 1.6s .25s ease infinite }
            .__ann-badge { animation: ann-in .25s ease }
        `
        document.head.appendChild(s)
    }

    const el = document.querySelector(selector)
    if (!el) { console.warn('ann: not found', selector); return }
    const r = el.getBoundingClientRect()
    const pad = 8

    const ring = document.createElement('div')
    ring.className = '__ann __ann-ring'
    ring.style.cssText = [
        'position:fixed', 'pointer-events:none', 'z-index:2147483647',
        `left:${r.left - pad}px`, `top:${r.top - pad}px`,
        `width:${r.width + pad * 2}px`, `height:${r.height + pad * 2}px`,
        `border:3px solid ${color}`, 'border-radius:8px',
    ].join(';')
    document.body.appendChild(ring)

    const badge = document.createElement('div')
    badge.className = '__ann __ann-badge'
    badge.textContent = label
    const bw = Math.max(label.length * 7.5 + 20, 120)
    let bLeft, bTop
    if (side === 'right') { bLeft = r.right + 14;     bTop = r.top + r.height/2 - 13 }
    if (side === 'left')  { bLeft = r.left - bw - 14; bTop = r.top + r.height/2 - 13 }
    if (side === 'above') { bLeft = r.left;            bTop = r.top - 36 }
    if (side === 'below') { bLeft = r.left;            bTop = r.bottom + 8 }
    badge.style.cssText = [
        'position:fixed', 'pointer-events:none', 'z-index:2147483647',
        `left:${bLeft}px`, `top:${bTop}px`,
        `background:${color}`, 'color:#fff',
        'padding:3px 10px', 'font:bold 12px/22px monospace',
        'border-radius:3px', 'white-space:nowrap',
        'box-shadow:0 2px 10px rgba(0,0,0,.5)',
    ].join(';')
    document.body.appendChild(badge)
}
"""


def ann(page, selector: str, label: str, side: str = "right", color: str = "#4F46E5") -> None:
    page.evaluate(_ANN_JS, {"selector": selector, "label": label, "side": side, "color": color})


def clear(page) -> None:
    page.evaluate("() => document.querySelectorAll('.__ann').forEach(n => n.remove())")


def scroll_to(page, selector: str) -> None:
    """Scroll element into center of viewport via plain JS (no Playwright retry)."""
    page.evaluate(
        "sel => { const el = document.querySelector(sel); if (el) el.scrollIntoView({block:'center', behavior:'smooth'}); }",
        selector,
    )
    page.wait_for_timeout(500)


def pause(page, ms: int) -> None:
    page.wait_for_timeout(ms)


# ── startup helpers ────────────────────────────────────────────────────────────

def _wait_for_app(url: str, timeout: int = 45) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"App not ready after {timeout}s at {url}")


def _get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    sys.exit("ffmpeg not found — install imageio-ffmpeg:  uv sync --all-extras")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    from playwright.sync_api import sync_playwright

    ffmpeg = _get_ffmpeg()
    GIF_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(VIDEO_TMP, ignore_errors=True)
    VIDEO_TMP.mkdir(parents=True, exist_ok=True)

    print("Starting Streamlit…")
    proc = subprocess.Popen(
        [
            "uv", "run", "streamlit", "run",
            str(ROOT / "src" / "edu_synth" / "app.py"),
            "--server.headless", "true",
            "--server.port", "8502",
            "--server.address", "localhost",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_for_app(APP_URL)
        print("App ready — recording…")

        with sync_playwright() as pw:
            browser = pw.chromium.launch()

            # Warm up: wait for Streamlit to fully render WITHOUT recording,
            # so the recording context starts with the app already loaded.
            ctx_warm = browser.new_context(viewport={"width": 1280, "height": 800})
            p_warm = ctx_warm.new_page()
            p_warm.goto(APP_URL)
            p_warm.wait_for_selector("[data-testid='stRadio']", timeout=30_000)
            ctx_warm.close()

            ctx = browser.new_context(
                record_video_dir=str(VIDEO_TMP),
                record_video_size={"width": 1280, "height": 800},
                viewport={"width": 1280, "height": 800},
            )
            page = ctx.new_page()

            # ── scene 1: landing ───────────────────────────────────────────────
            page.goto(APP_URL)
            # App is already warmed up — loads quickly this time.
            page.wait_for_selector("[data-testid='stRadio']", timeout=15_000)
            pause(page, 600)

            ann(page, "[data-testid='stRadio']", "Kies databron", side="right", color="#4F46E5")
            pause(page, 2000)
            clear(page)

            # ── scene 2: highlight upload zone ────────────────────────────────
            ann(page, "[data-testid='stFileUploader']", "Upload CSV of Parquet", side="below", color="#059669")
            pause(page, 2000)
            clear(page)

            # ── scene 3: upload fixture CSV ────────────────────────────────────
            page.locator("[data-testid='stFileUploader'] input[type='file']").set_input_files(
                str(FIXTURE_CSV)
            )
            page.wait_for_selector("[data-testid='stMetric']", timeout=20_000)
            pause(page, 500)

            # CSV triggers the longitudinal question; click "Nee" for tabular flow.
            nee = page.locator("label").filter(has_text="Nee")
            if nee.count():
                nee.first.click()
                pause(page, 600)

            # ── scene 4: scroll to top, annotate metrics ───────────────────────
            page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
            pause(page, 500)
            scroll_to(page, "[data-testid='stMetric']")
            ann(page, "[data-testid='stMetric']", "Automatisch gedetecteerd", side="right", color="#059669")
            pause(page, 2000)
            clear(page)

            # ── scene 5: open column-type hints expander ───────────────────────
            scroll_to(page, "[data-testid='stExpander']")
            page.locator("[data-testid='stExpander'] summary").first.click()
            pause(page, 600)
            ann(page, "[data-testid='stExpander']", "Kolomtypes bijstellen (optioneel)", side="below", color="#D97706")
            pause(page, 2500)
            clear(page)

            # ── scene 6: scroll to generate button ────────────────────────────
            scroll_to(page, "[data-testid='stButton']")
            ann(page, "[data-testid='stButton']", "1 klik → synthetische data", side="above", color="#4F46E5")
            pause(page, 2000)
            clear(page)

            # ── scene 7: generate ──────────────────────────────────────────────
            page.get_by_role("button", name="Genereer synthetische data").click()
            page.wait_for_selector("[data-testid='stTabs']", timeout=90_000)
            pause(page, 600)

            # ── scene 8: scroll to tabs, annotate ─────────────────────────────
            scroll_to(page, "[data-testid='stTabs']")
            ann(page, "[data-testid='stTabs']", "Validatierapport · Distributies · Download", side="below", color="#059669")
            pause(page, 2000)
            clear(page)

            # ── scene 9: scorecard in validation tab ───────────────────────────
            page.evaluate("window.scrollBy({top: 150, behavior: 'smooth'})")
            pause(page, 500)
            ann(page, "[data-testid='stMetric']", "Statistische kwaliteitsscore", side="right", color="#7C3AED")
            pause(page, 2000)
            clear(page)

            # ── scene 10: distributies tab ─────────────────────────────────────
            page.get_by_role("tab", name="Distributies").click()
            pause(page, 1000)
            scroll_to(page, "[data-testid='stPlotlyChart']")
            ann(page, "[data-testid='stPlotlyChart']", "Distributies: echt vs. synthetisch", side="below", color="#059669")
            pause(page, 2200)
            clear(page)
            pause(page, 400)

            ctx.close()
            browser.close()

        # ── convert WebM → GIF ─────────────────────────────────────────────────
        webms = list(VIDEO_TMP.glob("*.webm"))
        if not webms:
            sys.exit("No video found in " + str(VIDEO_TMP))

        print(f"Converting {webms[0].name} → {GIF_OUT}")
        subprocess.run(
            [
                ffmpeg, "-i", str(webms[0]),
                "-vf", ",".join([
                    "fps=8",
                    "scale=720:-1:flags=lanczos",
                    "split[s0][s1]",
                    "[s0]palettegen=max_colors=128:stats_mode=diff[p]",
                    "[s1][p]paletteuse=dither=bayer:bayer_scale=5",
                ]),
                "-y", str(GIF_OUT),
            ],
            check=True,
        )
        shutil.rmtree(VIDEO_TMP, ignore_errors=True)
        print(f"Done → {GIF_OUT}")

    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
