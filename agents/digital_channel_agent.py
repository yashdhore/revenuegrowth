from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .data_loader import save_output
from .digital_conversion_agent import FUNNEL_STEPS, prepare_digital


class DigitalChannelAgent:
    """Per-marketing-channel funnel: where each channel drops off versus the site, and whether it pays back."""

    def __init__(self, config: dict, step_playbook: dict | None = None):
        self.config = config
        self.step_playbook = step_playbook or {}
        self.output_path = config.get("output", {}).get("save_to")

    def run(self, digital: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if digital.empty or "MarketingChannel" not in digital.columns:
            return {"channel_funnel": pd.DataFrame(), "leaks": pd.DataFrame(), "scorecard": pd.DataFrame()}

        df = prepare_digital(digital)
        aov = float(df.loc[df["ConvertedFlag"], "Revenue"].mean()) if df["ConvertedFlag"].any() else 0.0
        channel_funnel = self._channel_funnel(df)
        leaks = self._leaks(df, channel_funnel, aov)
        scorecard = self._scorecard(df, leaks)
        save_output(scorecard, self.output_path)
        return {"channel_funnel": channel_funnel, "leaks": leaks, "scorecard": scorecard}

    @staticmethod
    def _reach(df: pd.DataFrame) -> list[int]:
        return [int((df["FunnelStepNumber"] >= step).sum()) for step in range(1, len(FUNNEL_STEPS) + 1)]

    def _channel_funnel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Long table: one row per channel (plus 'All Channels') per step."""
        groups = [("All Channels", df)] + list(df.groupby("MarketingChannel"))
        rows = []
        for channel, group in groups:
            reach = self._reach(group)
            for i, step_name in enumerate(FUNNEL_STEPS):
                has_next = i + 1 < len(FUNNEL_STEPS)
                rows.append(
                    {
                        "MarketingChannel": channel,
                        "Step": i + 1,
                        "StepName": step_name,
                        "VisitsReached": reach[i],
                        "PctOfChannelVisits": reach[i] / reach[0] if reach[0] else 0.0,
                        "ContinueRate": (reach[i + 1] / reach[i]) if has_next and reach[i] else None,
                        "DropOff": (reach[i] - reach[i + 1]) if has_next else None,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _downstream_conversion(df: pd.DataFrame, step: int) -> float:
        reached = df["FunnelStepNumber"] >= step
        return float(df.loc[reached, "ConvertedFlag"].mean()) if reached.any() else 0.0

    def _leaks(self, df: pd.DataFrame, channel_funnel: pd.DataFrame, aov: float) -> pd.DataFrame:
        """Channel/step cells where the channel continues materially below the site rate, sized in lost revenue."""
        min_visits = self.config.get("min_step_visits", 20)
        gap_threshold = self.config.get("leak_gap_threshold", 0.08)
        steps = channel_funnel.dropna(subset=["ContinueRate"])
        site = steps[steps["MarketingChannel"] == "All Channels"].set_index("Step")["ContinueRate"]
        rows = []
        for _, cell in steps[steps["MarketingChannel"] != "All Channels"].iterrows():
            step = int(cell["Step"])
            gap = site[step] - cell["ContinueRate"]
            if cell["VisitsReached"] < min_visits or gap < gap_threshold:
                continue
            lost_visits = cell["VisitsReached"] * gap
            lost_sales = lost_visits * self._downstream_conversion(df, step + 1)
            rows.append(
                {
                    "MarketingChannel": cell["MarketingChannel"],
                    "Step": step,
                    "LeakStep": f"{FUNNEL_STEPS[step - 1]} -> {FUNNEL_STEPS[step]}",
                    "VisitsReached": int(cell["VisitsReached"]),
                    "ChannelRate": cell["ContinueRate"],
                    "SiteRate": site[step],
                    "GapPts": gap,
                    "LostVisits": round(lost_visits, 1),
                    "LostSales": round(lost_sales, 1),
                    "LostRevenue": round(lost_sales * aov, 2),
                    "Fix": self._fix(cell["MarketingChannel"], step),
                }
            )
        leaks = pd.DataFrame(rows)
        return leaks.sort_values("LostRevenue", ascending=False).reset_index(drop=True) if not leaks.empty else leaks

    def _fix(self, channel: str, step: int) -> str:
        channel_fix = self.config.get("channel_step_fixes", {}).get(channel, {}).get(step)
        return channel_fix or self.step_playbook.get(step, "Review the page experience for this channel's visitors.")

    def _scorecard(self, df: pd.DataFrame, leaks: pd.DataFrame) -> pd.DataFrame:
        cost_per_visit = self.config.get("cost_per_visit", {})
        site_rate = float(df["ConvertedFlag"].mean())
        grouped = (
            df.groupby("MarketingChannel")
            .agg(
                Visits=("VisitID", "count"),
                StartedApplication=("FunnelStepNumber", lambda steps: int((steps >= 3).sum())),
                Conversions=("ConvertedFlag", "sum"),
                Revenue=("Revenue", "sum"),
            )
            .reset_index()
        )
        grouped["ConversionRate"] = grouped["Conversions"] / grouped["Visits"]
        grouped["ConversionIndex"] = grouped["ConversionRate"] / site_rate if site_rate else 0.0
        grouped["StartToSaleRate"] = grouped["Conversions"] / grouped["StartedApplication"].where(grouped["StartedApplication"] > 0)
        grouped["RevenuePerVisit"] = grouped["Revenue"] / grouped["Visits"]
        grouped["MediaCost"] = grouped.apply(lambda row: row["Visits"] * cost_per_visit.get(row["MarketingChannel"], 0.0), axis=1)
        grouped["ROAS"] = grouped["Revenue"] / grouped["MediaCost"].where(grouped["MediaCost"] > 0)
        grouped["CostPerSale"] = (grouped["MediaCost"] / grouped["Conversions"].where(grouped["Conversions"] > 0)).where(grouped["MediaCost"] > 0)

        worst_leak = leaks.groupby("MarketingChannel").head(1).set_index("MarketingChannel") if not leaks.empty else pd.DataFrame()
        grouped["BiggestLeak"] = grouped["MarketingChannel"].map(worst_leak["LeakStep"]) if not worst_leak.empty else None
        grouped["LeakRevenue"] = grouped["MarketingChannel"].map(worst_leak["LostRevenue"]).fillna(0.0) if not worst_leak.empty else 0.0
        grouped["BiggestLeak"] = grouped["BiggestLeak"].fillna("No major leak vs site")

        tiers = grouped.apply(self._tier, axis=1, result_type="expand")
        grouped["Tier"], grouped["Action"] = tiers[0], tiers[1]
        grouped["Timestamp"] = datetime.now(timezone.utc).isoformat()
        return grouped.sort_values("Revenue", ascending=False).reset_index(drop=True)

    def _tier(self, row: pd.Series) -> tuple[str, str]:
        if pd.isna(row["ROAS"]):
            return "Owned", self.config.get("owned_channel_action", "Owned traffic - protect and grow.")
        for tier in self.config.get("tiers", []):
            if row["ROAS"] >= tier["min_roas"] and row["ConversionIndex"] >= tier["min_conversion_index"]:
                return tier["name"], tier["action"]
        return "Review", "Review channel performance."
