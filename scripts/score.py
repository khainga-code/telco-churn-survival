"""Score a customer file and write a capacity-constrained contact queue.

Thin wrapper around the installed ``telco-score`` entry point so the script
works without installing the package.

Usage:
    python scripts/score.py --input data/telco_churn.csv --capacity 500
"""
from telco_churn.cli import main

if __name__ == "__main__":
    main()
