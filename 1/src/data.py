import os
from typing import List

import pandas as pd


REQUIRED_COLUMNS = {"image_path", "stock", "timeframe", "label"}


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset CSV and validate required columns.

    CSV must contain columns: image_path, stock, timeframe, label
    """
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    # Expand relative paths relative to CSV location
    base = os.path.dirname(os.path.abspath(csv_path))
    def _resolve(p):
        if os.path.isabs(p):
            return p
        return os.path.join(base, p)

    df["image_path"] = df["image_path"].map(_resolve)
    return df


def validate_images_exist(df: pd.DataFrame) -> List[str]:
    """Return a list of missing image paths (empty if all exist)."""
    missing = []
    for p in df["image_path"]:
        if not os.path.exists(p):
            missing.append(p)
    return missing
