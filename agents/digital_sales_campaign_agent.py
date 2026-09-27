from __future__ import annotations

from datetime import datetime, timezone
import uuid

import pandas as pd

from .data_loader import save_output
from .digital_conversion_agent import FUNNEL_STEPS, prepare_digital


class DigitalSalesCampaignAgent:
    """Win-back targets for digital abandoners, enriched with Salesforce and chat/phone signals."""

    def __init__(self, config: dict):
        self.config = config
        self.output_path = config.get("output", {}).get("save_to")

    def run(
        self,
        digital: pd.DataFrame,
        salesforce_clients: pd.DataFrame,
        insights: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        if digital.empty:
            return {"targets": pd.DataFrame(), "retargeting": pd.DataFrame()}

        df = prepare_digital(digital)
        plan_values = df.dropna(subset=["QuoteValue"]).groupby("PlanViewed")["QuoteValue"].mean().to_dict()
        min_step = self.config.get("min_abandon_step", 3)

        targets = self._targets(df, salesforce_clients, insights, plan_values, min_step)
        save_output(targets, self.output_path)
        return {"targets": targets, "retargeting": self._retargeting(df, plan_values, min_step)}

    def _targets(
        self,
        df: pd.DataFrame,
        salesforce_clients: pd.DataFrame,
        insights: pd.DataFrame,
        plan_values: dict,
        min_step: int,
    ) -> pd.DataFrame:
        df = df.assign(ContactKey=df["ClientID"].where(df["ClientID"] != "", df["LeadID"]))
        known = df[df["ContactKey"] != ""].sort_values("VisitDate")
        latest = known.groupby("ContactKey").tail(1)
        abandoners = latest[~latest["ConvertedFlag"] & (latest["FunnelStepNumber"] >= min_step)]
        if abandoners.empty:
            return pd.DataFrame()

        visit_counts = known.groupby("ContactKey").size()
        clients = salesforce_clients.set_index("ClientID") if "ClientID" in salesforce_clients.columns else pd.DataFrame()
        service = self._service_signals(insights)

        records = []
        for _, visit in abandoners.iterrows():
            client_id = visit["ClientID"]
            client = clients.loc[client_id] if client_id and client_id in clients.index else None
            signals = service.get(client_id, {})
            quote = visit["QuoteValue"] if pd.notna(visit["QuoteValue"]) else plan_values.get(visit["PlanViewed"], 0.0)
            channel, reason = self._choose_channel(visit, quote, signals)
            recovery_rate = self.config.get("recovery_rates", {}).get(channel, 0.05)
            records.append(
                {
                    "id": str(uuid.uuid4()),
                    "Timestamp": datetime.now(timezone.utc).isoformat(),
                    "ContactKey": visit["ContactKey"],
                    "ClientID": client_id,
                    "LeadID": visit["LeadID"],
                    "CustomerType": visit.get("CustomerType"),
                    "LastVisitDate": visit["VisitDate"].date() if pd.notna(visit["VisitDate"]) else None,
                    "DigitalVisits": int(visit_counts.get(visit["ContactKey"], 1)),
                    "AbandonStep": int(visit["FunnelStepNumber"]),
                    "AbandonStepName": FUNNEL_STEPS[int(visit["FunnelStepNumber"]) - 1],
                    "Device": visit.get("Device"),
                    "MarketingChannel": visit.get("MarketingChannel"),
                    "PlanViewed": visit.get("PlanViewed"),
                    "QuoteValue": round(float(quote), 2),
                    "ClientStatus": client.get("Status") if client is not None else None,
                    "LifetimeValue": client.get("LifetimeValue") if client is not None else None,
                    "Region": visit.get("Region"),
                    "ServiceInteractions": signals.get("interactions", 0),
                    "ServiceSentiment": signals.get("sentiment"),
                    "OpenServiceIssue": "Yes" if signals.get("unresolved") else "No",
                    "RecommendedChannel": channel,
                    "Reason": reason,
                    "Offer": self.config.get("offers", {}).get(channel, ""),
                    "RecoveryRate": recovery_rate,
                    "ExpectedRevenue": round(float(quote) * recovery_rate, 2),
                }
            )

        targets = pd.DataFrame(records).sort_values("ExpectedRevenue", ascending=False).reset_index(drop=True)
        # Top third by expected revenue is High priority, middle third Medium, rest Low.
        ranks = targets["ExpectedRevenue"].rank(pct=True, method="first")
        targets.insert(2, "Priority", pd.cut(ranks, [0, 1 / 3, 2 / 3, 1], labels=["Low", "Medium", "High"]).astype(str))
        return targets

    @staticmethod
    def _service_signals(insights: pd.DataFrame) -> dict[str, dict]:
        """Summarize chat/phone insights per client: volume, worst sentiment, any unresolved issue."""
        if insights.empty or "ClientID" not in insights.columns:
            return {}
        signals = {}
        for client_id, group in insights.groupby(insights["ClientID"].astype(str)):
            sentiments = group.get("Sentiment", pd.Series(dtype=str)).astype(str)
            worst = next((s for s in ["Negative", "Neutral", "Positive"] if (sentiments == s).any()), None)
            signals[client_id] = {
                "interactions": len(group),
                "sentiment": worst,
                "unresolved": (group.get("Resolved", pd.Series(dtype=str)).astype(str).str.lower() != "yes").any(),
                "high_risk": (group.get("RiskLevel", pd.Series(dtype=str)).astype(str) == "High").any(),
            }
        return signals

    def _choose_channel(self, visit: pd.Series, quote: float, signals: dict) -> tuple[str, str]:
        step = int(visit["FunnelStepNumber"])
        high_value = self.config.get("high_value_quote", 5000)
        if signals.get("unresolved") and (signals.get("sentiment") == "Negative" or signals.get("high_risk")):
            return "Phone Outreach", "Existing customer with an open, negative service issue - resolve before selling."
        if step >= 6 and quote >= high_value:
            return "Phone Outreach", f"High-value quote (${quote:,.0f}) abandoned late in the funnel."
        if step in (4, 5) and str(visit.get("ChatWidgetClicked", "")).lower() == "yes":
            return "Live Chat Assist", "Clicked chat during the application - wants help to finish."
        if step == 5 and visit.get("Device") in ("Mobile", "Tablet"):
            return "SMS Resume Link", "Dropped at identity verification on a mobile device."
        if step >= 6:
            return "Email Saved Quote", "Received a quote but did not sign or pay."
        return "Email Nurture", "Started the application but left before verification."

    def _retargeting(self, df: pd.DataFrame, plan_values: dict, min_step: int) -> pd.DataFrame:
        anonymous = df[
            (df["ClientID"] == "") & (df["LeadID"] == "") & ~df["ConvertedFlag"] & (df["FunnelStepNumber"] >= min_step)
        ]
        if anonymous.empty:
            return pd.DataFrame()
        rate = self.config.get("recovery_rates", {}).get("Paid Retargeting", 0.03)
        value = anonymous["QuoteValue"].fillna(anonymous["PlanViewed"].map(plan_values)).fillna(0)
        audience = (
            anonymous.assign(Value=value)
            .groupby("FunnelStepNumber")
            .agg(AudienceSize=("VisitID", "count"), AvgQuoteValue=("Value", "mean"))
            .reset_index()
        )
        audience["AbandonStepName"] = audience["FunnelStepNumber"].map(lambda step: FUNNEL_STEPS[step - 1])
        audience["ExpectedRevenue"] = (audience["AudienceSize"] * audience["AvgQuoteValue"] * rate).round(2)
        return audience[["FunnelStepNumber", "AbandonStepName", "AudienceSize", "AvgQuoteValue", "ExpectedRevenue"]]
