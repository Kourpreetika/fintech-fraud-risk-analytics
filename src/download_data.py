"""
Downloads the Credit Card Fraud Detection dataset (ULB / Worldline, via
OpenML dataset 1597) and writes it to data/raw/transactions.csv.

The dataset is not committed to the repository (145MB, over GitHub's
comfortable size limit) — run this script once after cloning.
See data/README.md for the full dataset documentation.
"""

import csv
import urllib.request
from pathlib import Path

OPENML_URL = "https://api.openml.org/data/v1/download/1673544/creditcard.arff"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
ARFF_PATH = RAW_DIR / "creditcard.arff"
CSV_PATH = RAW_DIR / "transactions.csv"


def download_arff():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {OPENML_URL} ...")
    urllib.request.urlretrieve(OPENML_URL, ARFF_PATH)
    print(f"Saved {ARFF_PATH} ({ARFF_PATH.stat().st_size / 1e6:.1f} MB)")


def arff_to_csv():
    with open(ARFF_PATH, encoding="utf-8") as f:
        lines = f.readlines()

    columns = []
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("@attribute"):
            columns.append(stripped.split()[1])
        if stripped.lower().startswith("@data"):
            data_start = i + 1
            break

    with open(ARFF_PATH, encoding="utf-8") as f, open(CSV_PATH, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(columns)
        for i, line in enumerate(f):
            if i < data_start:
                continue
            line = line.strip()
            if not line:
                continue
            row = next(csv.reader([line]))
            row[-1] = row[-1].strip("'")  # Class: '0' / '1' -> 0 / 1
            writer.writerow(row)

    print(f"Saved {CSV_PATH}")
    ARFF_PATH.unlink()


if __name__ == "__main__":
    download_arff()
    arff_to_csv()
