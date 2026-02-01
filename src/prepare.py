from __future__ import annotations

from pathlib import Path

import pandas as pd

def main() -> None:
    raw_path = Path("data/raw/diabetes.csv")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)

    #minmal realistic preprocessing

    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))

    out_path = out_dir / "diabetes_processed.csv"
    df.to_csv(out_path, index = False)

    print(f"Wrote {out_path} with shape = {df.shape}")


if __name__ == "__main__":
    main()