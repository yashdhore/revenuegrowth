"""Generate synthetic chat interactions.

Output: <data-dir>/input_chats.csv (CH001..CH100, one chat per day from 5/1/2024).
ClientIDs are sampled from input_salesforce_clients.csv, so generate that first.

Run:  python scripts/generate_chats.py [--data-dir data] [--seed 11]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from interaction_common import DEFAULT_DATA_DIR, InteractionSpec, generate_interactions, load_client_ids

CHAT_SPEC = InteractionSpec(
    id_column="ChatID",
    id_prefix="CH",
    id_width=3,
    date_column="ChatDate",
    days_between=1,
    time_spent_range=(15, 500),
    transcripts=[
        "I need a refund.",
        "Thanks for the quick help!",
        "Service has been slow.",
        "I'm satisfied with the product.",
        "I'm not sure if I will continue using this.",
    ],
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    df = generate_interactions(CHAT_SPEC, load_client_ids(args.data_dir), args.seed)
    out = args.data_dir / "input_chats.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} chats to {out}")


if __name__ == "__main__":
    main()
