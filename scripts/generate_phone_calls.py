"""Generate synthetic phone-call interactions.

Output: <data-dir>/input_phone_calls.csv (P001..P100, one call every 2 days from 5/1/2024).
ClientIDs are sampled from input_salesforce_clients.csv, so generate that first.

Run:  python scripts/generate_phone_calls.py [--data-dir data] [--seed 13]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from interaction_common import DEFAULT_DATA_DIR, InteractionSpec, generate_interactions, load_client_ids

PHONE_SPEC = InteractionSpec(
    id_column="CallID",
    id_prefix="P",
    id_width=3,
    date_column="CallDate",
    days_between=2,
    time_spent_range=(30, 600),
    transcripts=[
        "I love the support team!",
        "I'm not happy with the service.",
        "I might cancel soon.",
        "Everything is fine.",
        "Is there a better plan?",
    ],
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    df = generate_interactions(PHONE_SPEC, load_client_ids(args.data_dir), args.seed)
    out = args.data_dir / "input_phone_calls.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} phone calls to {out}")


if __name__ == "__main__":
    main()
