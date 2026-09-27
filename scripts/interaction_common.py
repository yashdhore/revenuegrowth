"""Shared generator for the chat and phone-call interaction files.

Both files have the same 14-column layout; only the ID/date column names,
transcript phrases, date spacing and handle-time range differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

AGENTS = ["Alex", "Casey", "Jordan", "Morgan", "Sam", "Taylor"]
FUNNEL_STEPS = [
    "application educate",
    "application start",
    "application submit",
    "application complete",
    "application abandon",
]
INTENTS = ["Billing", "Cancel Request", "Technical Support", "Product Info", "Feedback"]
NEXT_BEST_ACTIONS = ["Offer", "SMS", "Survey", "Phone Call", "Escalate"]
FIRST_INTERACTION = pd.Timestamp("2024-05-01")

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class InteractionSpec:
    id_column: str
    id_prefix: str
    id_width: int
    date_column: str
    days_between: int
    time_spent_range: tuple[int, int]
    transcripts: list[str]
    n_rows: int = 100


def load_client_ids(data_dir: Path) -> list[str]:
    """Interactions reference Salesforce clients, so read IDs from that file when present."""
    path = data_dir / "input_salesforce_clients.csv"
    if path.exists():
        return pd.read_csv(path)["ClientID"].tolist()
    return [f"C{i + 1:03d}" for i in range(100)]


def generate_interactions(spec: InteractionSpec, client_ids: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = spec.n_rows
    dates = [FIRST_INTERACTION + pd.Timedelta(days=spec.days_between * i) for i in range(n)]
    low, high = spec.time_spent_range
    return pd.DataFrame({
        spec.id_column: [f"{spec.id_prefix}{i + 1:0{spec.id_width}d}" for i in range(n)],
        "ClientID": rng.choice(client_ids, n),  # with replacement -> repeat contacts
        "Transcript": rng.choice(spec.transcripts, n),
        spec.date_column: [f"{d.month}/{d.day}/{d.year}" for d in dates],
        "TimeSpent": rng.integers(low, high + 1, n),
        "AgentName": rng.choice(AGENTS, n),
        "FunnelStep": rng.choice(FUNNEL_STEPS, n),
        "SentimentScore": rng.uniform(-1, 1, n).round(2),
        "Resolved": rng.choice(["Yes", "No"], n),
        "InteractionIntent": rng.choice(INTENTS, n),
        "PreviousContactCount": rng.integers(0, 6, n),
        "AgentPerformanceScore": rng.uniform(1, 5, n).round(2),
        "CampaignSent": rng.choice(["Yes", "No"], n),
        "NextBestAction": rng.choice(NEXT_BEST_ACTIONS, n),
    })
