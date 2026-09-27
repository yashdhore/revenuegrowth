from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Frame name -> file name inside the local data folder.
LOCAL_FILES = {
    "chats": "input_chats.csv",
    "phone_calls": "input_phone_calls.csv",
    "salesforce_clients": "input_salesforce_clients.csv",
    "adobe_digital": "input_adobe_digital.csv",
}
REQUIRED_FRAMES = ("chats", "phone_calls", "salesforce_clients")


@dataclass(frozen=True)
class GitHubCsvSources:
    chats_url: str
    phone_calls_url: str
    salesforce_clients_url: str
    adobe_digital_url: str = ""

    def validate(self) -> None:
        missing = [
            name
            for name, value in {
                "chats_url": self.chats_url,
                "phone_calls_url": self.phone_calls_url,
                "salesforce_clients_url": self.salesforce_clients_url,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing GitHub CSV URL(s): {', '.join(missing)}")


class GitHubCsvLoader:
    def __init__(self, sources: GitHubCsvSources, timeout_seconds: int = 30):
        sources.validate()
        self.sources = sources
        self.timeout_seconds = timeout_seconds

    def load_all(self) -> dict[str, pd.DataFrame]:
        frames = {
            "chats": self._read_csv(self.sources.chats_url),
            "phone_calls": self._read_csv(self.sources.phone_calls_url),
            "salesforce_clients": self._read_csv(self.sources.salesforce_clients_url),
        }
        frames["adobe_digital"] = self._read_optional_csv(self.sources.adobe_digital_url)
        return clean_frames(frames)

    def _read_optional_csv(self, url: str) -> pd.DataFrame:
        """Digital data is optional: a missing URL or file leaves the digital tabs empty instead of failing the app."""
        if not url:
            return pd.DataFrame()
        try:
            return self._read_csv(url)
        except requests.RequestException:
            return pd.DataFrame()

    def _read_csv(self, url: str) -> pd.DataFrame:
        response = requests.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))


def local_data_available(data_dir: str | Path) -> bool:
    folder = Path(data_dir)
    return all((folder / LOCAL_FILES[name]).exists() for name in REQUIRED_FRAMES)


def load_local_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    folder = Path(data_dir)
    frames = {}
    for name, file_name in LOCAL_FILES.items():
        path = folder / file_name
        if path.exists():
            frames[name] = pd.read_csv(path)
        elif name in REQUIRED_FRAMES:
            raise FileNotFoundError(f"Missing local data file: {path}")
        else:
            frames[name] = pd.DataFrame()
    return clean_frames(frames)


def clean_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Strip stray whitespace from headers (e.g. 'AppStatus ') so joins and lookups are reliable."""
    for df in frames.values():
        df.columns = df.columns.astype(str).str.strip()
    return frames


def save_output(df: pd.DataFrame, output_path: Optional[str]) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        df.to_csv(fallback_path, index=False)
