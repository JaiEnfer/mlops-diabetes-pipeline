from __future__ import annotations

from pathlib import Path

from sklearn.datasets import load_diabetes


def main() -> None:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_diabetes(as_frame=True)
    df = data.frame  # includes features + target

    out_path = out_dir / "diabetes.csv"
    df.to_csv(out_path, index=False)

    print(f"Wrote {out_path} with shape={df.shape}")


if __name__ == "__main__":
    main()
