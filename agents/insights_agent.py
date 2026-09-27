from __future__ import annotations

from datetime import datetime, timezone
import uuid

import pandas as pd

from .data_loader import save_output
from .llm_client import InteractionLlmClient


class InsightsAgent:
    def __init__(self, config: dict):
        self.config = config
        self.output_path = config.get("output", {}).get("save_to")
        self.llm_client = InteractionLlmClient()

    def run(self, chats: pd.DataFrame, phone_calls: pd.DataFrame, salesforce_clients: pd.DataFrame) -> pd.DataFrame:
        combined = self._combine_sources(chats, phone_calls, salesforce_clients)
        records = [self._build_record(row) for _, row in combined.iterrows()]
        df_out = pd.DataFrame(records)
        save_output(df_out, self.output_path)
        return df_out

    def _combine_sources(
        self,
        chats: pd.DataFrame,
        phone_calls: pd.DataFrame,
        salesforce_clients: pd.DataFrame,
    ) -> pd.DataFrame:
        chats = chats.copy()
        phone_calls = phone_calls.copy()
        salesforce_clients = salesforce_clients.copy()

        chats["SalesChannel"] = chats.get("SalesChannel", "chat")
        phone_calls["SalesChannel"] = phone_calls.get("SalesChannel", "phone")

        combined = pd.concat([chats, phone_calls], ignore_index=True, sort=False)
        # Salesforce's SalesChannel is the client's preferred channel; keep it separate so it
        # doesn't overwrite the channel this interaction actually came through.
        salesforce_clients = salesforce_clients.rename(columns={"SalesChannel": "PreferredChannel"})
        client_columns = [
            column
            for column in ["ClientID", "PreferredChannel", "Region", "Status", "LifetimeValue"]
            if column in salesforce_clients.columns
        ]

        if "ClientID" in combined.columns and len(client_columns) > 1:
            combined = combined.merge(salesforce_clients[client_columns], on="ClientID", how="left")

        return combined

    def _build_record(self, row: pd.Series) -> dict:
        analysis = self.llm_client.analyze(
            str(row.get("Transcript", "")),
            str(row.get("InteractionIntent", "Unknown")),
            str(row.get("Resolved", "No")),
        )
        return {
            "id": str(uuid.uuid4()),
            "Timestamp": datetime.now(timezone.utc).isoformat(),
            "ClientID": row.get("ClientID"),
            "AgentName": row.get("AgentName"),
            "Transcript": row.get("Transcript"),
            "FunnelStep": row.get("FunnelStep"),
            "TimeSpent": row.get("TimeSpent"),
            "SalesChannel": row.get("SalesChannel"),
            "PreferredChannel": row.get("PreferredChannel"),
            "Region": row.get("Region"),
            "ClientStatus": row.get("Status"),
            "LifetimeValue": row.get("LifetimeValue"),
            "Sentiment": analysis["Sentiment"],
            "SentimentScore": analysis["SentimentScore"],
            "Intent": analysis["Intent"],
            "Resolved": analysis["Resolved"],
            "RiskLevel": analysis["RiskLevel"],
            "ExecutiveInsight": analysis["ExecutiveInsight"],
            "NextBestAction": analysis["NextBestAction"],
        }
