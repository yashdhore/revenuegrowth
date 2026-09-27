"""Generate synthetic Salesforce client records.

Output: <data-dir>/input_salesforce_clients.csv (one row per client, C001..C100).

Mirrors the original file: every attribute is drawn uniformly and independently,
and LastPurchaseDate steps 15 days per client starting 1/1/2024.

Run:  python scripts/generate_salesforce_clients.py [--data-dir data] [--seed 7]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_CLIENTS = 100
FIRST_PURCHASE = pd.Timestamp("2024-01-01")
PURCHASE_STEP_DAYS = 15

STATUSES = ["Active", "At Risk", "Inactive"]
REGIONS = ["North", "East", "South", "West"]
FUNNEL_STEPS = [
    "application educate",
    "application start",
    "application submit",
    "application complete",
    "application abandon",
]
APP_STATUSES = ["app_received", "app_approved", "app_denied", "app_cancelled"]
SALES_CHANNELS = ["chat", "phone"]

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def generate(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [FIRST_PURCHASE + pd.Timedelta(days=PURCHASE_STEP_DAYS * i) for i in range(N_CLIENTS)]
    return pd.DataFrame({
        "ClientID": [f"C{i + 1:03d}" for i in range(N_CLIENTS)],
        "Name": [f"Client_{i + 1}" for i in range(N_CLIENTS)],
        "LastPurchaseDate": [f"{d.month}/{d.day}/{d.year}" for d in dates],
        "Status": rng.choice(STATUSES, N_CLIENTS),
        "LifetimeValue": rng.uniform(500, 10000, N_CLIENTS).round(2),
        "Region": rng.choice(REGIONS, N_CLIENTS),
        "FunnelStep": rng.choice(FUNNEL_STEPS, N_CLIENTS),
        "AppStatus": rng.choice(APP_STATUSES, N_CLIENTS),
        "SalesChannel": rng.choice(SALES_CHANNELS, N_CLIENTS),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    out = args.data_dir / "input_salesforce_clients.csv"
    generate(args.seed).to_csv(out, index=False)
    print(f"Wrote {N_CLIENTS} clients to {out}")


if __name__ == "__main__":
    main()
