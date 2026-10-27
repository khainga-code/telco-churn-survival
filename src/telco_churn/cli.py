"""Command-line scoring entry point.

Example:
    telco-score --input data/telco_churn.csv --capacity 500 --output reports/contact_queue.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from telco_churn.economics import Economics
from telco_churn.model import MODEL_DIR, load_bundle
from telco_churn.scoring import build_queue
from telco_churn.targeting import segment_report

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "telco_churn.csv")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "churn_logistic.joblib")
    parser.add_argument("--capacity", type=int, default=500,
                        help="maximum number of customers to contact")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "contact_queue.csv")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input)
    pipe, metadata = load_bundle(args.model)
    econ = Economics(**metadata.get("economics", {}))
    queue = build_queue(df, pipe, econ, args.capacity)

    id_cols = [c for c in ("customerID", "Contract", "InternetService", "PaymentMethod")
               if c in queue.columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    queue[id_cols + ["rank", "p_churn", "expected_value"]].to_csv(args.output, index=False)

    print(f"Scored {len(df):,} customers; contacting {len(queue):,} "
          f"(capacity {args.capacity:,}).")
    print(f"Expected margin of this queue: {queue['expected_value'].sum():,.0f} "
          f"at a contact cost of {econ.contact_cost:,.0f} each.")
    line = f"Mean predicted churn in queue: {queue['p_churn'].mean():.1%}"
    if "Churn" in df.columns:
        raw = df["Churn"]
        base = float((raw == "Yes").mean()) if raw.dtype == object else float(raw.mean())
        line += f" (base rate {base:.1%})"
    print(line)
    print(f"\nComposition:\n{segment_report(queue).round(3).to_string(index=False)}")
    print(f"\nWrote {args.output}")
