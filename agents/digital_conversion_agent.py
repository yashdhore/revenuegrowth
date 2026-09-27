from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .data_loader import save_output

FUNNEL_STEPS = [
    "Home Page",
    "Product Comparison",
    "Application Start",
    "Personal Details",
    "Identity Verification",
    "Plan Customization & Quote",
    "Review & e-Sign",
    "Payment & Confirmation",
]


def prepare_digital(digital: pd.DataFrame) -> pd.DataFrame:
    """Normalize types in the Adobe visit file; shared by both digital agents."""
    df = digital.copy()
    df["FunnelStepNumber"] = pd.to_numeric(df.get("FunnelStepNumber"), errors="coerce").fillna(1).astype(int)
    for column in ["Revenue", "QuoteValue", "FormErrors", "AvgPageLoadSec"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["ConvertedFlag"] = df.get("Converted", "No").astype(str).str.strip().str.lower().eq("yes")
    df["VisitDate"] = pd.to_datetime(df.get("VisitDate"), errors="coerce")
    for column in ["ClientID", "LeadID"]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()
    return df


class DigitalConversionAgent:
    """Funnel diagnostics and sized conversion opportunities for the digital team."""

    def __init__(self, config: dict):
        self.config = config
        self.output_path = config.get("output", {}).get("save_to")

    def run(self, digital: pd.DataFrame) -> dict[str, pd.DataFrame]:
        empty = {name: pd.DataFrame() for name in ["funnel", "device_funnel", "channels", "opportunities", "monthly"]}
        if digital.empty:
            return empty

        df = prepare_digital(digital)
        aov = float(df.loc[df["ConvertedFlag"], "Revenue"].mean()) if df["ConvertedFlag"].any() else 0.0
        funnel = self._funnel(df)
        device_funnel = self._device_funnel(df)
        channels = self._channels(df)
        opportunities = pd.concat(
            [self._step_opportunities(df, funnel, device_funnel, aov), self._channel_opportunities(df, channels, aov)],
            ignore_index=True,
        )
        if not opportunities.empty:
            opportunities = opportunities.sort_values("IncrementalRevenue", ascending=False).reset_index(drop=True)
            opportunities.insert(0, "Rank", range(1, len(opportunities) + 1))
            opportunities["Timestamp"] = datetime.now(timezone.utc).isoformat()
        save_output(opportunities, self.output_path)
        return {
            "funnel": funnel,
            "device_funnel": device_funnel,
            "channels": channels,
            "opportunities": opportunities,
            "monthly": self._monthly(df),
        }

    @staticmethod
    def _reach(df: pd.DataFrame) -> list[int]:
        return [int((df["FunnelStepNumber"] >= step).sum()) for step in range(1, len(FUNNEL_STEPS) + 1)]

    def _funnel(self, df: pd.DataFrame) -> pd.DataFrame:
        reach = self._reach(df)
        rows = []
        for i, step_name in enumerate(FUNNEL_STEPS):
            next_reach = reach[i + 1] if i + 1 < len(reach) else None
            rows.append(
                {
                    "Step": i + 1,
                    "StepName": step_name,
                    "VisitsReached": reach[i],
                    "PctOfVisits": reach[i] / reach[0] if reach[0] else 0.0,
                    "ContinueRate": (next_reach / reach[i]) if next_reach is not None and reach[i] else None,
                    "DropOff": (reach[i] - next_reach) if next_reach is not None else None,
                    "Benchmark": self.config.get("benchmark_continue_rates", {}).get(i + 1),
                }
            )
        return pd.DataFrame(rows)

    def _device_funnel(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for device, group in df.groupby("Device"):
            reach = self._reach(group)
            for i in range(len(FUNNEL_STEPS) - 1):
                rows.append(
                    {
                        "Device": device,
                        "Step": i + 1,
                        "Transition": f"{FUNNEL_STEPS[i]} -> {FUNNEL_STEPS[i + 1]}",
                        "VisitsReached": reach[i],
                        "ContinueRate": reach[i + 1] / reach[i] if reach[i] else 0.0,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _channels(df: pd.DataFrame) -> pd.DataFrame:
        grouped = (
            df.groupby("MarketingChannel")
            .agg(Visits=("VisitID", "count"), Conversions=("ConvertedFlag", "sum"), Revenue=("Revenue", "sum"))
            .reset_index()
        )
        grouped["ConversionRate"] = grouped["Conversions"] / grouped["Visits"]
        grouped["RevenuePerVisit"] = grouped["Revenue"] / grouped["Visits"]
        return grouped.sort_values("Revenue", ascending=False).reset_index(drop=True)

    @staticmethod
    def _monthly(df: pd.DataFrame) -> pd.DataFrame:
        dated = df.dropna(subset=["VisitDate"])
        if dated.empty:
            return pd.DataFrame()
        monthly = (
            dated.groupby(dated["VisitDate"].dt.to_period("M"))
            .agg(Visits=("VisitID", "count"), Conversions=("ConvertedFlag", "sum"), Revenue=("Revenue", "sum"))
            .reset_index()
        )
        monthly["Month"] = monthly["VisitDate"].dt.to_timestamp()
        return monthly.drop(columns="VisitDate")

    @staticmethod
    def _downstream_conversion(df: pd.DataFrame, step: int) -> float:
        """Share of visits reaching `step` that go on to convert."""
        reached = df["FunnelStepNumber"] >= step
        return float(df.loc[reached, "ConvertedFlag"].mean()) if reached.any() else 0.0

    def _step_opportunities(self, df: pd.DataFrame, funnel: pd.DataFrame, device_funnel: pd.DataFrame, aov: float) -> pd.DataFrame:
        playbook = self.config.get("step_playbook", {})
        gap_threshold = self.config.get("device_gap_threshold", 0.12)
        rows = []
        for _, step in funnel.dropna(subset=["ContinueRate", "Benchmark"]).iterrows():
            step_no = int(step["Step"])
            actual, target = float(step["ContinueRate"]), float(step["Benchmark"])
            if actual >= target:
                continue
            extra_visits = step["VisitsReached"] * (target - actual)
            extra_conversions = extra_visits * self._downstream_conversion(df, step_no + 1)
            rows.append(
                {
                    "Area": "Funnel Step",
                    "Opportunity": f"{FUNNEL_STEPS[step_no - 1]} -> {FUNNEL_STEPS[step_no]}",
                    "CurrentRate": actual,
                    "TargetRate": target,
                    "Diagnosis": self._diagnose(df, device_funnel, step_no, gap_threshold),
                    "Recommendation": playbook.get(step_no, "Review page experience and exit behavior."),
                    "IncrementalConversions": round(extra_conversions, 1),
                    "IncrementalRevenue": round(extra_conversions * aov, 2),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _diagnose(df: pd.DataFrame, device_funnel: pd.DataFrame, step_no: int, gap_threshold: float) -> str:
        notes = []
        rates = device_funnel[device_funnel["Step"] == step_no].set_index("Device")["ContinueRate"]
        if "Desktop" in rates:
            for device in ["Mobile", "Tablet"]:
                if device in rates and rates["Desktop"] - rates[device] >= gap_threshold:
                    notes.append(f"{device} continues at {rates[device]:.0%} vs Desktop {rates['Desktop']:.0%}")

        exits = df[(df["FunnelStepNumber"] == step_no) & ~df["ConvertedFlag"]]
        progressed = df[df["FunnelStepNumber"] > step_no]
        if "FormErrors" in df.columns and step_no >= 4 and not exits.empty and not progressed.empty:
            exit_errors, progressed_errors = exits["FormErrors"].mean(), progressed["FormErrors"].mean()
            if exit_errors >= progressed_errors + 0.5:
                notes.append(f"exiting visitors hit {exit_errors:.1f} form errors vs {progressed_errors:.1f}")
        if "AvgPageLoadSec" in df.columns and not exits.empty and not progressed.empty:
            exit_load, progressed_load = exits["AvgPageLoadSec"].mean(), progressed["AvgPageLoadSec"].mean()
            if exit_load >= progressed_load + 0.4:
                notes.append(f"pages load in {exit_load:.1f}s for exiting visitors vs {progressed_load:.1f}s")
        if not exits.empty and "MarketingChannel" in exits.columns:
            top_channel = exits["MarketingChannel"].value_counts()
            share = top_channel.iloc[0] / len(exits)
            if share >= 0.25:
                notes.append(f"{share:.0%} of exits come from {top_channel.index[0]}")
        if not notes:
            return "Drop-off is broad-based across devices and channels."
        text = "; ".join(notes)
        return text[0].upper() + text[1:]

    def _channel_opportunities(self, df: pd.DataFrame, channels: pd.DataFrame, aov: float) -> pd.DataFrame:
        site_rate = float(df["ConvertedFlag"].mean())
        ratio = self.config.get("low_quality_channel_ratio", 0.5)
        share = self.config.get("traffic_reallocation_share", 0.25)
        rows = []
        for _, channel in channels.iterrows():
            if channel["ConversionRate"] >= site_rate * ratio:
                continue
            shifted = channel["Visits"] * share
            extra_conversions = shifted * (site_rate - channel["ConversionRate"])
            rows.append(
                {
                    "Area": "Traffic Quality",
                    "Opportunity": f"{channel['MarketingChannel']} traffic quality",
                    "CurrentRate": channel["ConversionRate"],
                    "TargetRate": site_rate,
                    "Diagnosis": (
                        f"{channel['MarketingChannel']} drives {channel['Visits']:,} visits but converts at "
                        f"{channel['ConversionRate']:.1%} vs {site_rate:.1%} site average"
                    ),
                    "Recommendation": (
                        f"Shift {share:.0%} of {channel['MarketingChannel']} budget to higher-intent channels "
                        "and tighten audience targeting."
                    ),
                    "IncrementalConversions": round(extra_conversions, 1),
                    "IncrementalRevenue": round(extra_conversions * aov, 2),
                }
            )
        return pd.DataFrame(rows)
