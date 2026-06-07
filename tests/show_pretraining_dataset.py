import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import PreTrainingDataset
from src.utils import view_dataframe


def dataset_sample_to_dataframe(dataset: PreTrainingDataset, idx: int = 0) -> pd.DataFrame:
    x, y = dataset[idx]
    feature_df = pd.DataFrame(x.numpy(), columns=dataset.feature_cols)
    feature_df["Target_Log_Ret_Next"] = float(y.item())
    return feature_df


def show_parquet_as_dataframe(
    parquet_path: Optional[Path] = None,
    rows: int = 20,
    output_file: Optional[str] = None,
) -> None:
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        raise SystemExit(
            f"Processed data directory not found: {processed_dir}. "
            "Run the feature pipeline first or update processed_dir."
        )

    if parquet_path is None:
        files = sorted(processed_dir.glob("*.parquet"))
        if not files:
            raise SystemExit(f"No parquet files found in {processed_dir}.")
        parquet_path = files[0]
    else:
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise SystemExit(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"Showing parquet: {parquet_path}\n")
    view_dataframe(df, rows=rows, output_file=output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show a processed parquet file as a DataFrame using src.utils.view_dataframe."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to the parquet file to view. If omitted, the first file in data/processed is used.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Number of rows to display from the DataFrame.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the displayed DataFrame (CSV/Excel).",
    )
    args = parser.parse_args()

    show_parquet_as_dataframe(
        parquet_path=Path(args.file) if args.file else None,
        rows=args.rows,
        output_file=args.output,
    )
