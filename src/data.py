"""Data loading utilities for the Criteo Uplift project.

Note: sklift.datasets.fetch_criteo() points to an S3 bucket that now returns 404,
so we download the official file from Criteo's Hugging Face repo and take our own
reproducible 10% random sample (seed=42).
"""
from pathlib import Path
import urllib.request

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_FILE = DATA_DIR / "criteo-research-uplift-v2.1.csv.gz"
RAW_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)

FEATURES = [f"f{i}" for i in range(12)]
TARGETS = ["visit", "conversion"]
TREATMENT = "treatment"
EXPOSURE = "exposure"
SEED = 42

# Step 1 business assumptions (used later for ROI in Step 9)
AD_COST = 0.50      # ₹ per ad shown
VISIT_VALUE = 5.00  # ₹ value of one website visit


def download_raw() -> Path:
    """Download the full dataset (~311 MB gz, ~13.9M rows) if not present."""
    DATA_DIR.mkdir(exist_ok=True)
    if not RAW_FILE.exists():
        print(f"Downloading {RAW_URL} ...")
        urllib.request.urlretrieve(RAW_URL, RAW_FILE)
    return RAW_FILE


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Features -> float32, binary columns -> int8 (cuts memory ~4x)."""
    df = df.copy()
    for c in FEATURES:
        df[c] = df[c].astype("float32")
    for c in TARGETS + [TREATMENT, EXPOSURE]:
        df[c] = df[c].astype("int8")
    return df


def load_criteo(percent10: bool = True, use_cache: bool = True) -> pd.DataFrame:
    """Return the Criteo uplift data (10% random sample by default).

    Columns: f0..f11 (float32), treatment, exposure, visit, conversion (int8).
    First call reads the raw CSV (a few minutes) and caches parquet files in data/.
    """
    cache10 = DATA_DIR / "criteo10.parquet"
    cache_full = DATA_DIR / "criteo_full.parquet"
    target_cache = cache10 if percent10 else cache_full
    if use_cache and target_cache.exists():
        return pd.read_parquet(target_cache)

    if cache_full.exists():
        full = pd.read_parquet(cache_full)
    else:
        dtypes = {c: "float32" for c in FEATURES}
        dtypes.update({c: "int8" for c in TARGETS + [TREATMENT, EXPOSURE]})
        full = pd.read_csv(download_raw(), dtype=dtypes)
        full = optimize_dtypes(full)
        full.to_parquet(cache_full, index=False)

    if not percent10:
        return full
    df = full.sample(frac=0.10, random_state=SEED).reset_index(drop=True)
    df.to_parquet(cache10, index=False)
    return df
