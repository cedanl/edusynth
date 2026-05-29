"""Download publieke ontwikkeldatasets naar data/.

Gebruik:
    uv run scripts/download_datasets.py
    uv run scripts/download_datasets.py --dataset duo
    uv run scripts/download_datasets.py --dataset uci
    uv run scripts/download_datasets.py --dataset oulad
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

DATASETS: dict[str, dict] = {
    "uci": {
        "description": "UCI Student Performance (649 rijen, 33 kolommen, CC-BY 4.0)",
        "url": "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
        "filename": "student_performance.zip",
    },
    "oulad": {
        "description": "Open University Learning Analytics Dataset (7 tabellen, CC-BY 4.0)",
        "url": "https://analyse.kmi.open.ac.uk/open-dataset/download",
        "filename": "oulad.zip",
    },
}


def download(name: str) -> None:
    info = DATASETS[name]
    dest = DATA_DIR / info["filename"]
    print(f"Downloaden: {info['description']}")
    print(f"  → {dest}")
    DATA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(info["url"], dest)
    if dest.suffix == ".zip":
        extract_dir = DATA_DIR / name
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(dest, "r") as zf:
            zf.extractall(extract_dir)
        print(f"  uitgepakt naar {extract_dir}")
    print("  klaar.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ontwikkeldatasets naar data/.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Welke dataset downloaden (standaard: all).",
    )
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        download(name)


if __name__ == "__main__":
    main()
