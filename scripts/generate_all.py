"""Regenerate every synthetic input file in dependency order.

  1. Salesforce clients  (master list of ClientIDs)
  2. Chats               (samples ClientIDs from Salesforce)
  3. Phone calls         (samples ClientIDs from Salesforce)
  4. Adobe digital       (reuses Salesforce/chat/phone clients so channels join on ClientID)

WARNING: with the default --data-dir this overwrites the CSVs in data/.

Run:  python scripts/generate_all.py [--data-dir data]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ORDER = [
    "generate_salesforce_clients.py",
    "generate_chats.py",
    "generate_phone_calls.py",
    "generate_adobe_digital_data.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=SCRIPTS_DIR.parent / "data")
    args = parser.parse_args()

    for script in ORDER:
        subprocess.run([sys.executable, str(SCRIPTS_DIR / script), "--data-dir", str(args.data_dir)], check=True)


if __name__ == "__main__":
    main()
