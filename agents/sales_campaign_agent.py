from __future__ import annotations

from datetime import datetime, timezone
import uuid

import pandas as pd

from .data_loader import save_output
from .digital_conversion_agent import prepare_digital

SIGNAL_LABELS = {
    "negative_sentiment": "negative sentiment",
    "high_risk": "high churn risk",
    "cancel_request": "cancel request",
    "high_lifetime_value": "high lifetime value",
    "contacted_via_chat": "reached out via chat",
    "contacted_via_phone": "reached out via phone",
    "support_or_product_question": "support/product question",
    "billing_question": "billing question",
    "feedback": "left feedback",
    "prefers_chat": "prefers chat in Salesforce",
    "prefers_phone": "prefers phone in Salesforce",
    "active_on_digital": "active on the website",
    "abandoned_digital_application": "abandoned an online application",
    "mobile_user": "mostly on mobile",
}


class SalesCampaignAgent:
    def __init__(self, config: dict):
        self.config = config
        self.output_path = config.get("output", {}).get("save_to")

    def run(self, insights: pd.DataFrame, digital: pd.DataFrame | None = None) -> pd.DataFrame:
        filtered = self._apply_filters(insights.copy())
        recommendations = self._recommend(filtered, self._digital_profiles(digital))
        save_output(recommendations, self.output_path)
        return recommendations

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        filters = self.config.get("input", {}).get("filters", {})
        for column, allowed_values in filters.items():
            if column in df.columns and allowed_values:
                df = df[df[column].isin(allowed_values)]
        return df

    @staticmethod
    def _digital_profiles(digital: pd.DataFrame | None) -> dict[str, dict]:
        """Per-client digital behaviour from Adobe visits: activity, latest abandon, main device."""
        if digital is None or digital.empty:
            return {}
        df = prepare_digital(digital)
        df = df[df["ClientID"] != ""].sort_values("VisitDate")
        profiles = {}
        for client_id, visits in df.groupby("ClientID"):
            latest = visits.iloc[-1]
            profiles[client_id] = {
                "visits": len(visits),
                "abandoned": not latest["ConvertedFlag"] and latest["FunnelStepNumber"] >= 3,
                "mobile": (visits.get("Device", pd.Series(dtype=str)) == "Mobile").mean() > 0.5,
            }
        return profiles

    def _recommend(self, df: pd.DataFrame, digital_profiles: dict[str, dict]) -> pd.DataFrame:
        strategy = self.config["recommendation_strategy"]
        weights = strategy["score_weights"]
        priority = strategy["priority_order"]
        min_score = strategy["minimum_score_threshold"]

        records = []
        for _, row in df.iterrows():
            signals = self._signals(row, digital_profiles.get(str(row.get("ClientID")), {}))
            scores, contributions = self._score_channels(signals, weights)
            recommended_channel, score = sorted(scores.items(), key=lambda item: (-item[1], priority.index(item[0])))[0]
            reason = ", ".join(SIGNAL_LABELS[signal] for signal in contributions[recommended_channel]) or "default channel weighting"
            if score < min_score or str(row.get("Resolved", "No")).strip().lower() == "yes":
                recommended_channel, score, reason = "None", 0.0, "Issue already resolved - no outreach needed."

            records.append(
                {
                    "id": str(uuid.uuid4()),
                    "Timestamp": datetime.now(timezone.utc).isoformat(),
                    "ClientID": row.get("ClientID"),
                    "RecommendedChannel": recommended_channel,
                    "RecommendationScore": round(score, 3),
                    "Reason": reason,
                    **{f"{channel}Score": round(value, 3) for channel, value in scores.items()},
                    "Sentiment": row.get("Sentiment"),
                    "Intent": row.get("Intent"),
                    "Resolved": row.get("Resolved"),
                    "SalesChannel": row.get("SalesChannel"),
                    "PreferredChannel": row.get("PreferredChannel"),
                    "Region": row.get("Region"),
                }
            )

        return pd.DataFrame(records)

    def _signals(self, row: pd.Series, digital_profile: dict) -> set[str]:
        threshold = self.config["recommendation_strategy"].get("high_lifetime_value_threshold", 7000)
        intent = str(row.get("Intent", ""))
        lifetime_value = pd.to_numeric(row.get("LifetimeValue"), errors="coerce")
        checks = {
            "negative_sentiment": row.get("Sentiment") == "Negative",
            "high_risk": row.get("RiskLevel") == "High",
            "cancel_request": intent == "Cancel Request",
            "high_lifetime_value": pd.notna(lifetime_value) and lifetime_value >= threshold,
            "contacted_via_chat": row.get("SalesChannel") == "chat",
            "contacted_via_phone": row.get("SalesChannel") == "phone",
            "support_or_product_question": intent in ("Technical Support", "Product Info"),
            "billing_question": intent == "Billing",
            "feedback": intent == "Feedback",
            "prefers_chat": row.get("PreferredChannel") == "chat",
            "prefers_phone": row.get("PreferredChannel") == "phone",
            "active_on_digital": digital_profile.get("visits", 0) > 0,
            "abandoned_digital_application": digital_profile.get("abandoned", False),
            "mobile_user": digital_profile.get("mobile", False),
        }
        return {signal for signal, active in checks.items() if active}

    def _score_channels(self, signals: set[str], weights: dict[str, float]) -> tuple[dict[str, float], dict[str, list[str]]]:
        boosts = self.config["recommendation_strategy"].get("signal_boosts", {})
        scores = dict(weights)
        contributions: dict[str, list[str]] = {channel: [] for channel in weights}
        for signal in (name for name in SIGNAL_LABELS if name in signals):  # stable order for readable reasons
            for channel, boost in boosts.get(signal, {}).items():
                if channel in scores:
                    scores[channel] += boost
                    contributions[channel].append(signal)
        return scores, contributions
