from __future__ import annotations

from io import StringIO
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from agent_runner import load_config, run_pipeline
from agents.data_loader import GitHubCsvLoader, GitHubCsvSources, clean_frames, load_local_data, local_data_available
from agents.digital_conversion_agent import FUNNEL_STEPS
from agents.llm_client import InteractionLlmClient


st.set_page_config(page_title="Revenue Growth Intelligence Platform", page_icon="RG", layout="wide")

LOGO_PATH = Path("logo/AnalyticsAI Logo_B7.jpg")


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1480px;
    }
    .stApp {
        background: linear-gradient(180deg, #f8fbfd 0%, #ffffff 42%);
    }
    h1 {
        color: #062b5f !important;
        border-left: 6px solid #18c99a;
        padding-left: 0.8rem;
    }
    h1, h2, h3 {
        color: #062b5f;
    }
    .main-subtitle {
        color: #006fd6;
        font-size: 1.04rem;
        font-weight: 600;
        margin-top: -0.25rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d9e1e8;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetricLabel"] {
        color: #4b5f6f;
        font-size: 0.82rem;
    }
    .executive-note {
        border-left: 4px solid #18c99a;
        background: #f1fbf8;
        padding: 0.85rem 1rem;
        margin: 0.75rem 0 1.1rem;
        color: #263845;
    }
    .sample-note {
        border-left: 4px solid #006fd6;
        background: #eef7ff;
        padding: 0.8rem 0.95rem;
        margin: 0.65rem 0 1rem;
        color: #1f3344;
        font-size: 0.94rem;
    }
    .agent-copy {
        color: #2f3f4a;
        font-size: 1.02rem;
        line-height: 1.55;
        margin-bottom: 0.8rem;
    }
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(90deg, #006fd6 0%, #18c99a 100%) !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #0059ad 0%, #10a77f 100%) !important;
        color: #ffffff !important;
        border: 0 !important;
    }
    .stButton > button:disabled {
        background: #e3e9ee !important;
        color: #9aa9b6 !important;
        cursor: not-allowed;
    }
    /* Tab styling uses ARIA roles, which are stable across Streamlit versions (1.45 BaseWeb and 1.5x+ React Aria). */
    div[data-testid="stTabs"] [role="tablist"] {
        flex-wrap: wrap;               /* wrap onto extra rows so every tab stays visible */
        gap: 0.35rem 0.3rem;
        overflow: visible;
        border-bottom: 0;
    }
    div[data-testid="stTabs"] [role="tab"] {
        background: #e7f3ff;
        color: #062b5f;
        border: 0;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0;
        font-weight: 700;
        cursor: pointer;
        flex: 0 0 auto;
    }
    div[data-testid="stTabs"] [role="tab"]:hover {
        background: #d3e9ff;
    }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #006fd6 0%, #18c99a 100%);
        color: #ffffff;
    }
    div[data-testid="stTabs"] [role="tab"] p {
        color: inherit;
        font-weight: 700;
        margin: 0;
    }
    /* Hide the default underline / scroll chrome; the pill colours show the active tab. */
    div[data-testid="stTabs"] .react-aria-SelectionIndicator,
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"] {
        display: none;
    }
    .tab-position {
        text-align: center;
        color: #062b5f;
        font-weight: 700;
        padding-top: 0.45rem;
    }
    .insight-card {
        background: #ffffff;
        border: 1px solid #d9e8f2;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin: 0.75rem 0 1.15rem;
        box-shadow: 0 1px 2px rgba(6, 43, 95, 0.06);
    }
    .insight-card ul {
        margin: 0;
        padding-left: 1.15rem;
    }
    .insight-card li {
        margin: 0.35rem 0;
        color: #1f3344;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_github_data(
    chats_url: str, phone_calls_url: str, salesforce_clients_url: str, adobe_digital_url: str
) -> dict[str, pd.DataFrame]:
    sources = GitHubCsvSources(
        chats_url=chats_url.strip(),
        phone_calls_url=phone_calls_url.strip(),
        salesforce_clients_url=salesforce_clients_url.strip(),
        adobe_digital_url=adobe_digital_url.strip(),
    )
    return GitHubCsvLoader(sources).load_all()


def load_input_data(
    chats_file,
    calls_file,
    clients_file,
    digital_file,
    use_local: bool,
    local_dir: str,
    urls: dict[str, str],
) -> tuple[dict[str, pd.DataFrame], str]:
    uploaded_files = [chats_file, calls_file, clients_file]
    if any(uploaded_files) or digital_file:
        if not all(uploaded_files):
            raise ValueError("Upload all three core CSV files: chats, phone calls, and Salesforce clients.")
        frames = {
            "chats": pd.read_csv(chats_file),
            "phone_calls": pd.read_csv(calls_file),
            "salesforce_clients": pd.read_csv(clients_file),
            "adobe_digital": pd.read_csv(digital_file) if digital_file else pd.DataFrame(),
        }
        return clean_frames(frames), "local CSV uploads"
    if use_local:
        return load_local_data(local_dir), f"the local '{local_dir}' data folder"
    return (
        load_github_data(urls["chats"], urls["phone_calls"], urls["salesforce_clients"], urls["adobe_digital"]),
        "sample GitHub CSVs",
    )


@st.cache_data(show_spinner=False)
def run_agents(
    chats_csv: str,
    calls_csv: str,
    clients_csv: str,
    digital_csv: str,
    config: dict,
    max_interactions: int,
) -> dict:
    chats = pd.read_json(StringIO(chats_csv), orient="split")
    calls = pd.read_json(StringIO(calls_csv), orient="split")
    clients = pd.read_json(StringIO(clients_csv), orient="split")
    digital = pd.read_json(StringIO(digital_csv), orient="split", dtype=False)
    if max_interactions > 0:
        chats, calls = trim_interactions(chats, calls, max_interactions)

    frames = {"chats": chats, "phone_calls": calls, "salesforce_clients": clients, "adobe_digital": digital}
    return run_pipeline(frames, config)


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def trim_interactions(chats: pd.DataFrame, calls: pd.DataFrame, max_interactions: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    chat_limit = max_interactions // 2
    call_limit = max_interactions - chat_limit
    return chats.head(chat_limit), calls.head(call_limit)


def show_logo(width: int = 150) -> None:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=width)


def show_llm_note() -> None:
    st.markdown(
        "<div class='executive-note'><strong>LLM-driven insight layer:</strong> "
        "The bullets below are derived from the currently loaded CSV data and enriched by the configured "
        "backend LLM, with deterministic fallback logic available when the LLM is not enabled.</div>",
        unsafe_allow_html=True,
    )


def show_sample_note(data_source_label: str) -> None:
    st.markdown(
        f"<div class='sample-note'><strong>Data context:</strong> This dashboard is currently using "
        f"<strong>{data_source_label}</strong>. The metrics, bullets, and LLM-enriched insights shown here "
        "are based only on the CSV files loaded in this session. Use the sidebar to upload local CSVs for a client-specific readout.</div>",
        unsafe_allow_html=True,
    )


def show_bullets(items: list[str]) -> None:
    html_items = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(f"<div class='insight-card'><ul>{html_items}</ul></div>", unsafe_allow_html=True)


def top_value(df: pd.DataFrame, column: str, default: str = "N/A") -> tuple[str, int]:
    if df.empty or column not in df.columns:
        return default, 0
    counts = df[column].dropna().astype(str).value_counts()
    if counts.empty:
        return default, 0
    return counts.index[0], int(counts.iloc[0])


def outreach_only(campaign: pd.DataFrame) -> pd.DataFrame:
    """Campaign rows with a real outreach channel ("None" means the issue is already resolved)."""
    if campaign.empty or "RecommendedChannel" not in campaign.columns:
        return campaign
    return campaign[campaign["RecommendedChannel"].astype(str) != "None"]


def pct(numerator: int | float, denominator: int | float) -> str:
    if not denominator:
        return "0%"
    return f"{(numerator / denominator) * 100:.0f}%"


def build_insight_bullets(insights: pd.DataFrame) -> list[str]:
    total = len(insights)
    top_intent, top_intent_count = top_value(insights, "Intent")
    top_sentiment, top_sentiment_count = top_value(insights, "Sentiment")
    high_risk = int((insights.get("RiskLevel", pd.Series(dtype=str)).astype(str) == "High").sum())
    unresolved = int((insights.get("Resolved", pd.Series(dtype=str)).astype(str).str.lower() != "yes").sum())
    top_channel, top_channel_count = top_value(insights, "SalesChannel")
    return [
        f"{top_intent} is the leading customer intent, representing {top_intent_count:,} of {total:,} analyzed interactions.",
        f"{top_sentiment} is the dominant sentiment signal across {top_sentiment_count:,} interactions, shaping the current customer experience narrative.",
        f"{unresolved:,} interactions remain unresolved, creating a {pct(unresolved, total)} service-to-growth recovery opportunity.",
        f"{high_risk:,} interactions are classified as high risk, giving leadership a focused queue for retention intervention.",
        f"{top_channel} is the highest-volume interaction channel with {top_channel_count:,} records, indicating where customer voice is currently concentrated.",
    ]


def build_performance_bullets(performance: pd.DataFrame, insights: pd.DataFrame) -> list[str]:
    total_agents = performance["AgentName"].nunique() if not performance.empty and "AgentName" in performance.columns else 0
    top_agent = "N/A"
    top_resolution = 0.0
    coaching_count = 0
    avg_resolution = 0.0
    sentiment_leader = "N/A"
    if not performance.empty:
        sorted_resolution = performance.sort_values("ResolutionRate", ascending=False)
        top_agent = str(sorted_resolution.iloc[0]["AgentName"])
        top_resolution = float(sorted_resolution.iloc[0]["ResolutionRate"])
        coaching_count = int((performance["Category"] == "Train").sum()) if "Category" in performance.columns else 0
        avg_resolution = float(performance["ResolutionRate"].mean()) if "ResolutionRate" in performance.columns else 0.0
        if "AvgSentimentScore" in performance.columns:
            sentiment_leader = str(performance.sort_values("AvgSentimentScore", ascending=False).iloc[0]["AgentName"])
    unresolved = int((insights.get("Resolved", pd.Series(dtype=str)).astype(str).str.lower() != "yes").sum())
    return [
        f"{total_agents:,} frontline agents are scored from the current interaction dataset.",
        f"{top_agent} has the strongest resolution signal at {top_resolution:.0%}, creating a benchmark for playbook replication.",
        f"The average resolution rate is {avg_resolution:.0%}, showing the current operating baseline for customer issue closure.",
        f"{coaching_count:,} agents fall into the coaching category, highlighting where enablement can lift customer outcomes.",
        f"{sentiment_leader} leads the sentiment quality signal, while {unresolved:,} unresolved cases show where execution can still protect revenue.",
    ]


def build_campaign_bullets(campaign: pd.DataFrame, insights: pd.DataFrame) -> list[str]:
    targets = len(campaign)
    top_channel, top_channel_count = top_value(outreach_only(campaign), "RecommendedChannel")
    top_intent, top_intent_count = top_value(campaign, "Intent")
    negative = int((campaign.get("Sentiment", pd.Series(dtype=str)).astype(str) == "Negative").sum())
    none_count = int((campaign.get("RecommendedChannel", pd.Series(dtype=str)).astype(str) == "None").sum())
    actionable = targets - none_count
    high_risk = int((insights.get("RiskLevel", pd.Series(dtype=str)).astype(str) == "High").sum())
    return [
        f"{targets:,} customer records qualify for campaign review from the loaded data.",
        f"{top_channel} is the leading recommended outreach path, assigned to {top_channel_count:,} customer opportunities.",
        "Outreach mix: " + ", ".join(f"{channel} {count:,}" for channel, count in outreach_only(campaign)["RecommendedChannel"].value_counts().items())
        + " - each recommendation carries the signals behind it in the Reason column.",
        f"{top_intent} is the top campaign-triggering intent, appearing in {top_intent_count:,} recommended outreach records.",
        f"{negative:,} campaign candidates carry negative sentiment, giving marketing a focused retention and recovery segment.",
        f"{actionable:,} records have an actionable channel recommendation, with {high_risk:,} high-risk signals available for prioritization.",
    ]


def build_executive_summary_bullets(insights: pd.DataFrame, performance: pd.DataFrame, campaign: pd.DataFrame) -> list[str]:
    total = len(insights)
    unresolved = int((insights.get("Resolved", pd.Series(dtype=str)).astype(str).str.lower() != "yes").sum())
    high_risk = int((insights.get("RiskLevel", pd.Series(dtype=str)).astype(str) == "High").sum())
    top_intent, top_intent_count = top_value(insights, "Intent")
    top_channel, top_channel_count = top_value(outreach_only(campaign), "RecommendedChannel")
    avg_resolution = float(performance["ResolutionRate"].mean()) if not performance.empty and "ResolutionRate" in performance.columns else 0.0
    return [
        f"The CMO-level signal is concentrated around {top_intent}, which accounts for {top_intent_count:,} of {total:,} analyzed interactions.",
        f"{unresolved:,} unresolved interactions create an immediate recovery opportunity across service, sales, and lifecycle marketing.",
        f"{high_risk:,} customer conversations are tagged high risk, making them candidates for priority retention or escalation workflows.",
        f"The current agent resolution baseline is {avg_resolution:.0%}, giving leadership a measurable operating lever for improving experience quality.",
        f"{top_channel} is the leading campaign action path with {top_channel_count:,} recommendations, translating customer voice into activation.",
    ]


def build_financial_bullets(insights: pd.DataFrame, campaign: pd.DataFrame) -> list[str]:
    lifetime_value = pd.to_numeric(insights.get("LifetimeValue", pd.Series(dtype=float)), errors="coerce")
    avg_ltv = float(lifetime_value.dropna().mean()) if not lifetime_value.dropna().empty else 0.0
    unresolved = int((insights.get("Resolved", pd.Series(dtype=str)).astype(str).str.lower() != "yes").sum())
    high_risk = int((insights.get("RiskLevel", pd.Series(dtype=str)).astype(str) == "High").sum())
    actionable = int((campaign.get("RecommendedChannel", pd.Series(dtype=str)).astype(str) != "None").sum())
    estimated_risk = high_risk * avg_ltv
    recovery_pool = unresolved * avg_ltv * 0.15
    return [
        f"Average customer lifetime value in the loaded CSVs is approximately ${avg_ltv:,.0f}, based on available Salesforce client records.",
        f"High-risk interactions imply an indicative revenue-at-risk pool of ${estimated_risk:,.0f} if those accounts are not recovered.",
        f"Unresolved interactions represent a modeled recovery opportunity of ${recovery_pool:,.0f}, assuming a 15% save or conversion impact.",
        f"{actionable:,} records have a concrete follow-up channel recommendation, giving marketing a measurable activation queue.",
        "These estimates are directional for executive prioritization and should be recalibrated with client-specific margin, churn, and conversion assumptions.",
    ]


def build_customer_journey_bullets(selected_row: pd.Series) -> list[str]:
    return [
        f"Customer {selected_row.get('ClientID', 'N/A')} entered through the {selected_row.get('SalesChannel', 'unknown')} channel with intent classified as {selected_row.get('Intent', 'Unknown')}.",
        f"The LLM-enriched sentiment is {selected_row.get('Sentiment', 'Unknown')} with risk level {selected_row.get('RiskLevel', 'Unknown')}.",
        f"Resolution status is {selected_row.get('Resolved', 'Unknown')}, which determines whether this record should move into campaign or service recovery.",
        f"The recommended next action is: {selected_row.get('NextBestAction', 'Review and follow up with the customer.')}",
        f"Executive readout: {selected_row.get('ExecutiveInsight', 'Use this interaction to validate the customer journey and campaign response path.')}",
    ]


def digital_totals(digital_visits: pd.DataFrame) -> dict[str, float]:
    if digital_visits.empty:
        return {"visits": 0, "conversions": 0, "revenue": 0.0, "conversion_rate": 0.0, "aov": 0.0}
    converted = digital_visits["Converted"].astype(str).str.lower() == "yes"
    revenue = float(pd.to_numeric(digital_visits["Revenue"], errors="coerce").sum())
    return {
        "visits": len(digital_visits),
        "conversions": int(converted.sum()),
        "revenue": revenue,
        "conversion_rate": float(converted.mean()),
        "aov": revenue / converted.sum() if converted.any() else 0.0,
    }


def build_revenue_levers(insights: pd.DataFrame, digital: dict, digital_campaign: dict, digital_visits: pd.DataFrame) -> pd.DataFrame:
    """Growth levers identified across chat, phone, Salesforce and digital, on top of booked digital revenue."""
    lifetime_value = pd.to_numeric(insights.get("LifetimeValue", pd.Series(dtype=float)), errors="coerce").dropna()
    avg_ltv = float(lifetime_value.mean()) if not lifetime_value.empty else 0.0
    unresolved = int((insights.get("Resolved", pd.Series(dtype=str)).astype(str).str.lower() != "yes").sum())
    opportunities = digital.get("opportunities", pd.DataFrame())
    targets = digital_campaign.get("targets", pd.DataFrame())
    retargeting = digital_campaign.get("retargeting", pd.DataFrame())
    levers = [
        ("Digital funnel fixes", "Digital", float(opportunities["IncrementalRevenue"].sum()) if not opportunities.empty else 0.0),
        ("Digital win-back campaign", "Digital", float(targets["ExpectedRevenue"].sum()) if not targets.empty else 0.0),
        ("Anonymous retargeting", "Digital", float(retargeting["ExpectedRevenue"].sum()) if not retargeting.empty else 0.0),
        ("Chat & phone recovery", "Service", unresolved * avg_ltv * 0.15),  # same 15% assumption as the Financial bullets
    ]
    return pd.DataFrame(levers, columns=["Lever", "Source", "Revenue"])


def build_channel_coverage(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, int, int]:
    """How many Salesforce clients appear in each channel, and how many are seen in 2+ / all channels."""
    clients = set(frames["salesforce_clients"].get("ClientID", pd.Series(dtype=str)).astype(str))
    touch = {
        "Chat": set(frames["chats"].get("ClientID", pd.Series(dtype=str)).astype(str)),
        "Phone": set(frames["phone_calls"].get("ClientID", pd.Series(dtype=str)).astype(str)),
        "Digital (Adobe)": set(frames["adobe_digital"].get("ClientID", pd.Series(dtype=str)).dropna().astype(str)),
    }
    coverage = pd.DataFrame(
        [{"Channel": "Salesforce (CRM)", "Clients": len(clients)}]
        + [{"Channel": name, "Clients": len(ids & clients)} for name, ids in touch.items()]
    )
    channel_counts = pd.Series({client: sum(client in ids for ids in touch.values()) for client in clients})
    return coverage, int((channel_counts >= 2).sum()), int((channel_counts == len(touch)).sum())


def build_growth_bullets(levers: pd.DataFrame, totals: dict, multi_channel: int, all_channel: int, total_clients: int) -> list[str]:
    growth = float(levers["Revenue"].sum())
    top = levers.sort_values("Revenue", ascending=False).iloc[0]
    return [
        f"Digital booked ${totals['revenue']:,.0f} from {totals['conversions']:,} online sales at a {totals['conversion_rate']:.1%} visit conversion rate.",
        f"The agents identify ${growth:,.0f} of incremental revenue across digital and service channels, a "
        f"{pct(growth, totals['revenue'])} lift on current digital revenue.",
        f"{top['Lever']} is the largest single lever at ${top['Revenue']:,.0f}.",
        f"{multi_channel:,} of {total_clients:,} Salesforce clients are seen in two or more channels and {all_channel:,} in chat, phone and digital, "
        "so each customer can be worked with one connected view instead of three siloed ones.",
        "Estimates are directional: funnel uplift assumes each step reaches its benchmark rate, and campaign value applies configured recovery rates to quote values.",
    ]


def build_digital_conversion_bullets(digital: dict, totals: dict) -> list[str]:
    funnel = digital["funnel"]
    opportunities = digital["opportunities"]
    worst = funnel.dropna(subset=["ContinueRate"]).iloc[1:].sort_values("DropOff", ascending=False).iloc[0]
    bullets = [
        f"{totals['visits']:,} digital visits produced {totals['conversions']:,} completed sales ({totals['conversion_rate']:.1%}) "
        f"and ${totals['revenue']:,.0f} revenue at a ${totals['aov']:,.0f} average order value.",
        f"The largest loss after the home page is {worst['StepName']} -> {FUNNEL_STEPS[int(worst['Step'])]}: "
        f"{int(worst['DropOff']):,} visits drop out ({worst['ContinueRate']:.0%} continue).",
    ]
    for _, row in opportunities.head(3).iterrows():
        bullets.append(f"#{row['Rank']} {row['Opportunity']}: +${row['IncrementalRevenue']:,.0f}. {row['Diagnosis'].rstrip('.')}.")
    bullets.append(f"Closing all identified gaps is worth an estimated ${opportunities['IncrementalRevenue'].sum():,.0f} in additional digital revenue.")
    return bullets


def build_digital_campaign_bullets(digital_campaign: dict) -> list[str]:
    targets = digital_campaign["targets"]
    retargeting = digital_campaign["retargeting"]
    existing = targets[targets["ClientID"].astype(str) != ""]
    open_issue = int((targets["OpenServiceIssue"] == "Yes").sum())
    by_channel = targets.groupby("RecommendedChannel")["ExpectedRevenue"].sum().sort_values(ascending=False)
    audience = int(retargeting["AudienceSize"].sum()) if not retargeting.empty else 0
    return [
        f"{len(targets):,} known digital abandoners ({len(existing):,} existing customers, {len(targets) - len(existing):,} new leads) "
        f"left ${targets['QuoteValue'].sum():,.0f} of quoted value on the table.",
        f"Targeted outreach is expected to recover ${targets['ExpectedRevenue'].sum():,.0f}; "
        f"{by_channel.index[0]} carries the most value at ${by_channel.iloc[0]:,.0f}.",
        f"{open_issue:,} abandoners also have an open chat or phone issue, so they go to a rep to resolve the issue before the sales ask.",
        f"{int((targets['Priority'] == 'High').sum()):,} high-priority contacts account for "
        f"${targets.loc[targets['Priority'] == 'High', 'ExpectedRevenue'].sum():,.0f} of expected recovery - start there.",
        f"A further {audience:,} anonymous abandoners can be reached through paid retargeting audiences.",
    ]


def funnel_chart(funnel: pd.DataFrame) -> alt.Chart:
    data = funnel.assign(Label=funnel["Step"].astype(str) + ". " + funnel["StepName"])
    base = alt.Chart(data).encode(
        y=alt.Y("Label:N", sort=list(data["Label"]), title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("VisitsReached:Q", title="Visits reaching step", scale=alt.Scale(domain=[0, int(data["VisitsReached"].max() * 1.12)])),
    )
    bars = base.mark_bar(color="#006fd6", cornerRadiusEnd=3).encode(
        tooltip=["StepName", "VisitsReached", alt.Tooltip("PctOfVisits:Q", format=".0%"), alt.Tooltip("ContinueRate:Q", format=".0%")]
    )
    labels = base.mark_text(align="left", dx=4, color="#062b5f").encode(text=alt.Text("VisitsReached:Q", format=","))
    return (bars + labels).properties(height=300)


def device_chart(device_funnel: pd.DataFrame) -> alt.Chart:
    data = device_funnel.assign(Label=device_funnel["Step"].astype(str) + "->" + (device_funnel["Step"] + 1).astype(str))
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("Label:N", title="Funnel transition (step -> step)", sort=None, axis=alt.Axis(labelAngle=0)),
            xOffset="Device:N",
            y=alt.Y("ContinueRate:Q", title="Continue rate", axis=alt.Axis(format="%")),
            color=alt.Color("Device:N", scale=alt.Scale(range=["#006fd6", "#18c99a", "#9aa9b6"])),
            tooltip=["Device", "Transition", alt.Tooltip("ContinueRate:Q", format=".0%"), "VisitsReached"],
        )
        .properties(height=300)
    )


def value_bar_chart(data: pd.DataFrame, category: str, value: str, title: str, color: str = "#18c99a") -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_bar(color=color, cornerRadiusEnd=3)
        .encode(
            y=alt.Y(f"{category}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(format="$,.0f")),
            tooltip=[category, alt.Tooltip(f"{value}:Q", format="$,.0f")],
        )
        .properties(height=240)
    )


def join_names(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def build_channel_insight_bullets(channels: dict, totals: dict) -> list[str]:
    scorecard, leaks = channels["scorecard"], channels["leaks"]
    total_visits, total_revenue = scorecard["Visits"].sum(), scorecard["Revenue"].sum()
    bullets = []

    top_revenue = scorecard.iloc[0]
    best_converters = scorecard.sort_values("ConversionIndex", ascending=False).head(2)
    bullets.append(
        f"{top_revenue['MarketingChannel']} is the largest revenue channel (${top_revenue['Revenue']:,.0f}, "
        f"{pct(top_revenue['Revenue'], total_revenue)} of digital revenue). "
        f"{join_names(best_converters['MarketingChannel'].tolist())} convert best at "
        f"{best_converters['ConversionIndex'].iloc[0]:.1f}x the site average."
    )

    weak = scorecard[scorecard["Tier"] == "Fix or Cut"]
    if not weak.empty:
        weak_leaks = weak["BiggestLeak"].value_counts()
        bullets.append(
            f"{join_names(weak['MarketingChannel'].tolist())} bring {pct(weak['Visits'].sum(), total_visits)} of visits but only "
            f"{pct(weak['Revenue'].sum(), total_revenue)} of revenue; most of their visitors leave at {weak_leaks.index[0]}."
        )
        scale = scorecard[(scorecard["Tier"] == "Scale") & scorecard["CostPerSale"].notna()].sort_values("Revenue", ascending=False)
        if not scale.empty:
            target = scale.iloc[0]
            funded_sales = weak["MediaCost"].sum() / target["CostPerSale"]
            bullets.append(
                f"Moving the ${weak['MediaCost'].sum():,.0f} spent on {join_names(weak['MarketingChannel'].tolist())} into "
                f"{target['MarketingChannel']} (${target['CostPerSale']:,.0f} per sale) would fund about {funded_sales:,.0f} sales "
                f"versus {int(weak['Conversions'].sum()):,} today."
            )

    for _, leak in leaks.head(3).iterrows():
        bullets.append(
            f"{leak['MarketingChannel']} leaks at {leak['LeakStep']}: {leak['ChannelRate']:.0%} continue vs {leak['SiteRate']:.0%} site-wide, "
            f"costing about ${leak['LostRevenue']:,.0f}. Fix: {leak['Fix']}"
        )
    if not leaks.empty:
        bullets.append(f"Bringing every leaking channel up to the site rate at its leak step is worth about ${leaks['LostRevenue'].sum():,.0f}.")
    bullets.append("Media cost per visit is an assumption set in config/agents_config.yaml - replace it with actual spend for true ROAS.")
    return bullets


def channel_heatmap(channel_funnel: pd.DataFrame, channel_order: list[str]) -> alt.Chart:
    steps = channel_funnel.dropna(subset=["ContinueRate"]).copy()
    site = steps[steps["MarketingChannel"] == "All Channels"].set_index("Step")["ContinueRate"]
    steps["VsSite"] = steps["ContinueRate"] - steps["Step"].map(site)
    steps["Transition"] = steps["Step"].astype(str) + "->" + (steps["Step"] + 1).astype(str)
    base = alt.Chart(steps).encode(
        x=alt.X("Transition:N", title="Funnel transition (step -> step)", sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("MarketingChannel:N", sort=["All Channels"] + channel_order, title=None, axis=alt.Axis(labelLimit=200)),
    )
    cells = base.mark_rect().encode(
        color=alt.Color(
            "VsSite:Q",
            title="vs site rate",
            scale=alt.Scale(scheme="redblue", domain=[-0.3, 0.3]),
            legend=alt.Legend(format="+.0%"),
        ),
        tooltip=[
            "MarketingChannel",
            "StepName",
            alt.Tooltip("ContinueRate:Q", format=".0%", title="Continue rate"),
            alt.Tooltip("VsSite:Q", format="+.0%", title="vs site"),
            alt.Tooltip("VisitsReached:Q", format=",", title="Visits at step"),
        ],
    )
    labels = base.mark_text(fontSize=12).encode(
        text=alt.Text("ContinueRate:Q", format=".0%"),
        color=alt.condition("abs(datum.VsSite) > 0.18", alt.value("#ffffff"), alt.value("#17212b")),
    )
    return (cells + labels).properties(height=34 * (len(channel_order) + 1))


def channel_funnel_compare_chart(channel_funnel: pd.DataFrame, channel: str) -> alt.Chart:
    data = channel_funnel[channel_funnel["MarketingChannel"].isin(["All Channels", channel])].copy()
    data["Label"] = data["Step"].astype(str) + ". " + data["StepName"]
    return (
        alt.Chart(data)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Label:N", sort=None, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=160)),
            y=alt.Y("PctOfChannelVisits:Q", title="% of channel visits reaching step", axis=alt.Axis(format="%")),
            color=alt.Color(
                "MarketingChannel:N",
                title=None,
                scale=alt.Scale(domain=[channel, "All Channels"], range=["#006fd6", "#9aa9b6"]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=["MarketingChannel", "StepName", alt.Tooltip("PctOfChannelVisits:Q", format=".1%"), "VisitsReached"],
        )
        .properties(height=320)
    )


def channel_effectiveness_chart(scorecard: pd.DataFrame) -> alt.Chart:
    tier_colors = {"Scale": "#18c99a", "Optimize": "#006fd6", "Fix or Cut": "#d9534f", "Owned": "#9aa9b6"}
    base = alt.Chart(scorecard).encode(
        x=alt.X("ConversionRate:Q", title="Visit conversion rate", axis=alt.Axis(format="%")),
        y=alt.Y("RevenuePerVisit:Q", title="Revenue per visit", axis=alt.Axis(format="$,.0f")),
    )
    points = base.mark_circle(opacity=0.85).encode(
        size=alt.Size("Visits:Q", scale=alt.Scale(range=[150, 1400]), legend=None),
        color=alt.Color(
            "Tier:N",
            scale=alt.Scale(domain=list(tier_colors), range=list(tier_colors.values())),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[
            "MarketingChannel",
            "Tier",
            alt.Tooltip("Visits:Q", format=","),
            alt.Tooltip("ConversionRate:Q", format=".1%"),
            alt.Tooltip("Revenue:Q", format="$,.0f"),
            alt.Tooltip("ROAS:Q", format=".1f"),
        ],
    )
    labels = base.mark_text(align="left", dx=12, color="#062b5f").encode(text="MarketingChannel:N")
    return (points + labels).properties(height=320)


TAB_NAMES = [
    "Executive Summary",
    "Financial Lens",
    "Customer Journey",
    "Insights Agent",
    "Performance Agent",
    "Sales Campaign Agent",
    "Digital Conversion",
    "Digital Channel Insights",
    "Digital Sales Campaign",
]
ACTIVE_TAB_KEY = "active_tab"


def active_tab_index() -> int:
    current = st.session_state.get(ACTIVE_TAB_KEY, TAB_NAMES[0])
    return TAB_NAMES.index(current) if current in TAB_NAMES else 0


def step_tab(step: int) -> None:
    index = min(max(active_tab_index() + step, 0), len(TAB_NAMES) - 1)
    st.session_state[ACTIVE_TAB_KEY] = TAB_NAMES[index]


def show_tab_navigator() -> None:
    index = active_tab_index()
    prev_col, position_col, next_col = st.columns([1, 2, 1], vertical_alignment="center")
    prev_col.button("◀ Previous", on_click=step_tab, args=(-1,), disabled=index == 0, width="stretch")
    position_col.markdown(
        f"<div class='tab-position'>Tab {index + 1} of {len(TAB_NAMES)} - {TAB_NAMES[index]}</div>",
        unsafe_allow_html=True,
    )
    next_col.button("Next ▶", on_click=step_tab, args=(1,), disabled=index == len(TAB_NAMES) - 1, width="stretch")


def show_tab_header(title: str, description: str) -> None:
    title_col, logo_col = st.columns([0.82, 0.18], vertical_alignment="center")
    with title_col:
        st.subheader(title)
        st.markdown(f"<div class='agent-copy'>{description}</div>", unsafe_allow_html=True)
    with logo_col:
        show_logo(width=95)


config = load_config()
source_defaults = config.get("data_sources", {})
llm_status = InteractionLlmClient()

hero_col, logo_col = st.columns([0.78, 0.22], vertical_alignment="center")
with hero_col:
    st.title("Revenue Growth Intelligence Platform")
    st.markdown(
        "<div class='main-subtitle'>AI agents that convert customer conversations into CMO-ready growth, "
        "performance, and campaign activation signals.</div>",
        unsafe_allow_html=True,
    )
with logo_col:
    show_logo(width=120)

with st.sidebar:
    st.header("Data Sources")
    st.markdown(
        "Sample data covers chat, phone, Salesforce and Adobe digital. Upload local CSVs below to replace it for this session."
    )
    chats_file = st.file_uploader("Upload local chats CSV", type=["csv"])
    calls_file = st.file_uploader("Upload local phone calls CSV", type=["csv"])
    clients_file = st.file_uploader("Upload local Salesforce clients CSV", type=["csv"])
    digital_file = st.file_uploader("Upload local Adobe digital CSV (optional)", type=["csv"])
    st.divider()
    local_dir = source_defaults.get("local_data_dir", "data")
    has_local = bool(local_dir) and local_data_available(local_dir)
    use_local = st.toggle(f"Use local '{local_dir}' folder", value=has_local, disabled=not has_local)
    st.caption("Sample GitHub CSV URLs (used when the local folder is off)")
    source_urls = {
        "chats": st.text_input("Chats CSV raw URL", value=source_defaults.get("chats_url", "")),
        "phone_calls": st.text_input("Phone calls CSV raw URL", value=source_defaults.get("phone_calls_url", "")),
        "salesforce_clients": st.text_input("Salesforce clients CSV raw URL", value=source_defaults.get("salesforce_clients_url", "")),
        "adobe_digital": st.text_input("Adobe digital CSV raw URL", value=source_defaults.get("adobe_digital_url", "")),
    }
    max_interactions = st.slider("Interaction sample size", min_value=20, max_value=200, value=60, step=20)
    st.divider()
    st.header("Backend LLM")
    if llm_status.available:
        st.success(f"Enabled: {llm_status.model}")
    else:
        st.warning("Fallback mode: add OPENAI_API_KEY to .env to enable LLM insights.")
    refresh_clicked = st.button("Refresh agents", type="primary", width="stretch")

if refresh_clicked:
    load_github_data.clear()
    run_agents.clear()

try:
    with st.spinner("Reading CSVs and running AI agents..."):
        frames, data_source_label = load_input_data(
            chats_file,
            calls_file,
            clients_file,
            digital_file,
            use_local,
            local_dir,
            source_urls,
        )
        result = run_agents(
            frames["chats"].to_json(orient="split"),
            frames["phone_calls"].to_json(orient="split"),
            frames["salesforce_clients"].to_json(orient="split"),
            frames["adobe_digital"].to_json(orient="split"),
            config,
            max_interactions,
        )
except Exception as exc:
    st.error(f"Unable to run agents: {exc}")
    st.stop()

insights_df = result["insights"]
performance_df = result["performance"]
campaign_df = result["campaign"]
digital = result["digital"]
digital_campaign = result["digital_campaign"]
digital_channels = result["digital_channels"]
digital_visits = frames["adobe_digital"]
has_digital = not digital_visits.empty
totals = digital_totals(digital_visits)

metric_cols = st.columns(6)
metric_cols[0].metric("Interactions", f"{len(insights_df):,}")
metric_cols[1].metric("Digital Visits", f"{totals['visits']:,}")
metric_cols[2].metric("Digital Revenue", f"${totals['revenue'] / 1000:,.0f}K")
metric_cols[3].metric("Campaign Targets", f"{len(campaign_df) + len(digital_campaign.get('targets', [])):,}")
metric_cols[4].metric("Unresolved Cases", f"{(insights_df['Resolved'].astype(str).str.lower() != 'yes').sum():,}")
metric_cols[5].metric("High Risk", f"{(insights_df.get('RiskLevel', pd.Series(dtype=str)).astype(str) == 'High').sum():,}")

show_sample_note(data_source_label)
show_tab_navigator()

(
    tab_summary,
    tab_financial,
    tab_journey,
    tab_insights,
    tab_performance,
    tab_campaign,
    tab_digital,
    tab_channels,
    tab_digital_campaign,
) = st.tabs(TAB_NAMES, key=ACTIVE_TAB_KEY, on_change="rerun")

with tab_summary:
    show_tab_header(
        "Executive Summary",
        "A CMO-ready view of the most important growth, retention, and campaign signals from the loaded interaction data.",
    )
    show_llm_note()
    show_bullets(build_executive_summary_bullets(insights_df, performance_df, campaign_df))
    if has_digital:
        levers_df = build_revenue_levers(insights_df, digital, digital_campaign, digital_visits)
        coverage_df, multi_channel, all_channel = build_channel_coverage(frames)
        st.markdown("#### Omnichannel Revenue Growth")
        show_bullets(build_growth_bullets(levers_df, totals, multi_channel, all_channel, int(coverage_df["Clients"].iloc[0])))
    summary_cols = st.columns(3)
    if "Intent" in insights_df.columns:
        summary_cols[0].bar_chart(insights_df["Intent"].value_counts().head(8))
    if "RiskLevel" in insights_df.columns:
        summary_cols[1].bar_chart(insights_df["RiskLevel"].value_counts())
    if "RecommendedChannel" in campaign_df.columns:
        summary_cols[2].bar_chart(outreach_only(campaign_df)["RecommendedChannel"].value_counts())

with tab_financial:
    show_tab_header(
        "Financial Lens",
        "A directional revenue view that translates customer risk and campaign recommendations into an executive opportunity frame.",
    )
    show_llm_note()
    show_bullets(build_financial_bullets(insights_df, campaign_df))
    lifetime_value = pd.to_numeric(insights_df.get("LifetimeValue", pd.Series(dtype=float)), errors="coerce")
    finance_cols = st.columns(3)
    finance_cols[0].metric("Average Lifetime Value", f"${lifetime_value.dropna().mean():,.0f}" if not lifetime_value.dropna().empty else "N/A")
    finance_cols[1].metric("Actionable Campaign Records", f"{(campaign_df.get('RecommendedChannel', pd.Series(dtype=str)).astype(str) != 'None').sum():,}")
    finance_cols[2].metric("High Risk Records", f"{(insights_df.get('RiskLevel', pd.Series(dtype=str)).astype(str) == 'High').sum():,}")
    if has_digital:
        st.markdown("#### Revenue Growth Across Chat, Phone, Salesforce and Digital")
        growth_total = float(levers_df["Revenue"].sum())
        growth_cols = st.columns(4)
        growth_cols[0].metric("Digital Revenue Booked", f"${totals['revenue']:,.0f}")
        growth_cols[1].metric("Identified Growth", f"${growth_total:,.0f}", f"{pct(growth_total, totals['revenue'])} on digital")
        growth_cols[2].metric("Clients in 2+ Channels", f"{multi_channel:,}")
        growth_cols[3].metric("Clients in Chat + Phone + Digital", f"{all_channel:,}")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.caption("Incremental revenue by growth lever")
            st.altair_chart(value_bar_chart(levers_df, "Lever", "Revenue", "Incremental revenue"), width="stretch")
        with chart_cols[1]:
            st.caption("Monthly digital revenue (booked)")
            monthly = digital.get("monthly", pd.DataFrame())
            if not monthly.empty:
                st.line_chart(monthly.set_index("Month")[["Revenue"]], color="#006fd6", height=240)
        st.caption("Salesforce clients seen in each channel")
        st.altair_chart(
            alt.Chart(coverage_df)
            .mark_bar(color="#18c99a", cornerRadiusEnd=3)
            .encode(
                y=alt.Y("Channel:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
                x=alt.X("Clients:Q", title="Clients"),
                tooltip=["Channel", "Clients"],
            )
            .properties(height=180),
            width="stretch",
        )
        st.markdown("#### Service Campaign Records (chat & phone)")
    st.dataframe(
        campaign_df[["ClientID", "RecommendedChannel", "RecommendationScore", "Reason", "Sentiment", "Intent", "Resolved"]]
        if not campaign_df.empty
        else campaign_df,
        width="stretch",
        hide_index=True,
    )

with tab_journey:
    show_tab_header(
        "Customer Journey",
        "A drill-through view showing how one customer interaction becomes insight, risk, and a recommended activation path.",
    )
    show_llm_note()
    journey_options = insights_df["ClientID"].dropna().astype(str).tolist() if "ClientID" in insights_df.columns else []
    selected_client = st.selectbox("Select customer record", journey_options) if journey_options else None
    if selected_client:
        selected_row = insights_df[insights_df["ClientID"].astype(str) == selected_client].iloc[0]
        show_bullets(build_customer_journey_bullets(selected_row))
        st.text_area("Transcript", value=str(selected_row.get("Transcript", "")), height=180)
        st.dataframe(pd.DataFrame([selected_row]), width="stretch", hide_index=True)
        if has_digital:
            client_visits = digital_visits[digital_visits["ClientID"].astype(str) == selected_client]
            st.markdown("#### Digital Journey (Adobe)")
            if client_visits.empty:
                st.info("No authenticated digital visits for this customer.")
            else:
                deepest = int(client_visits["FunnelStepNumber"].max())
                converted = (client_visits["Converted"].astype(str).str.lower() == "yes").any()
                st.markdown(
                    f"<div class='agent-copy'>{len(client_visits)} digital visit(s); furthest step reached: "
                    f"<strong>{FUNNEL_STEPS[deepest - 1]}</strong> ({'converted online' if converted else 'not yet converted online'}).</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    client_visits[
                        ["VisitDate", "Device", "MarketingChannel", "FunnelStepReached", "PlanViewed", "QuoteValue", "Converted", "Revenue"]
                    ],
                    width="stretch",
                    hide_index=True,
                )

with tab_insights:
    show_tab_header(
        "Insights Agent",
        "Transforms raw customer transcripts into a boardroom-ready view of sentiment, intent, risk, and next best action.",
    )
    show_llm_note()
    show_bullets(build_insight_bullets(insights_df))
    chart_cols = st.columns(3)
    if "Sentiment" in insights_df.columns:
        chart_cols[0].bar_chart(insights_df["Sentiment"].value_counts())
    if "Intent" in insights_df.columns:
        chart_cols[1].bar_chart(insights_df["Intent"].value_counts())
    if "RiskLevel" in insights_df.columns:
        chart_cols[2].bar_chart(insights_df["RiskLevel"].value_counts())
    st.dataframe(insights_df, width="stretch", hide_index=True)
    st.download_button("Download insights CSV", dataframe_to_csv(insights_df), "output_insights.csv", "text/csv")

with tab_performance:
    show_tab_header(
        "Performance Agent",
        "Connects customer outcomes to agent execution so leadership can see where performance is creating or limiting growth.",
    )
    show_llm_note()
    show_bullets(build_performance_bullets(performance_df, insights_df))
    if not performance_df.empty:
        st.bar_chart(performance_df.set_index("AgentName")[["ResolutionRate", "AvgSentimentScore"]])
    st.dataframe(performance_df, width="stretch", hide_index=True)
    st.download_button(
        "Download performance CSV",
        dataframe_to_csv(performance_df),
        "output_agent_performance_report.csv",
        "text/csv",
    )

with tab_campaign:
    show_tab_header(
        "Sales Campaign Agent",
        "Turns unresolved and at-risk interactions into prioritized outreach actions for conversion, retention, and win-back teams.",
    )
    show_llm_note()
    show_bullets(build_campaign_bullets(campaign_df, insights_df))
    chart_cols = st.columns(2)
    if not campaign_df.empty and "RecommendedChannel" in campaign_df.columns:
        chart_cols[0].bar_chart(outreach_only(campaign_df)["RecommendedChannel"].value_counts())
    if not campaign_df.empty and "Sentiment" in campaign_df.columns:
        chart_cols[1].bar_chart(campaign_df["Sentiment"].value_counts())
    st.dataframe(campaign_df, width="stretch", hide_index=True)
    st.download_button(
        "Download campaign CSV",
        dataframe_to_csv(campaign_df),
        "output_sales_campaign_recommendations.csv",
        "text/csv",
    )

with tab_digital:
    show_tab_header(
        "Digital Conversion Agent",
        "Diagnoses the 8-step Adobe digital funnel, pinpoints where and why visitors drop out, and sizes each fix in revenue.",
    )
    if not has_digital or digital["funnel"].empty:
        st.info("Load the Adobe digital CSV to see funnel diagnostics.")
    else:
        opportunities_df = digital["opportunities"]
        show_bullets(build_digital_conversion_bullets(digital, totals))
        kpi_cols = st.columns(5)
        kpi_cols[0].metric("Digital Visits", f"{totals['visits']:,}")
        kpi_cols[1].metric("Conversion Rate", f"{totals['conversion_rate']:.1%}")
        kpi_cols[2].metric("Digital Revenue", f"${totals['revenue']:,.0f}")
        kpi_cols[3].metric("Average Order Value", f"${totals['aov']:,.0f}")
        kpi_cols[4].metric("Identified Uplift", f"${opportunities_df['IncrementalRevenue'].sum():,.0f}")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.caption("8-step digital funnel: visits reaching each step")
            st.altair_chart(funnel_chart(digital["funnel"]), width="stretch")
        with chart_cols[1]:
            st.caption("Step continue rate by device")
            st.altair_chart(device_chart(digital["device_funnel"]), width="stretch")

        st.markdown("#### Prioritized Conversion Opportunities")
        st.dataframe(
            opportunities_df[
                ["Rank", "Area", "Opportunity", "CurrentRate", "TargetRate", "IncrementalConversions", "IncrementalRevenue", "Diagnosis", "Recommendation"]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "CurrentRate": st.column_config.NumberColumn("Current", format="percent"),
                "TargetRate": st.column_config.NumberColumn("Target", format="percent"),
                "IncrementalConversions": st.column_config.NumberColumn("Extra Sales", format="%.1f"),
                "IncrementalRevenue": st.column_config.NumberColumn("Extra Revenue", format="dollar"),
                "Diagnosis": st.column_config.TextColumn(width="large"),
                "Recommendation": st.column_config.TextColumn(width="large"),
            },
        )

        st.markdown("#### Marketing Channel Performance")
        st.dataframe(
            digital["channels"],
            width="stretch",
            hide_index=True,
            column_config={
                "Revenue": st.column_config.NumberColumn(format="dollar"),
                "ConversionRate": st.column_config.NumberColumn("Conversion Rate", format="percent"),
                "RevenuePerVisit": st.column_config.NumberColumn("Revenue / Visit", format="dollar"),
            },
        )
        st.download_button(
            "Download digital opportunities CSV",
            dataframe_to_csv(opportunities_df),
            "output_digital_opportunities.csv",
            "text/csv",
        )

with tab_channels:
    show_tab_header(
        "Digital Channel Insights",
        "Shows where each marketing channel's visitors drop out of the 8-step funnel compared with the site, "
        "and which channels pay back their spend.",
    )
    scorecard_df = digital_channels.get("scorecard", pd.DataFrame())
    if not has_digital or scorecard_df.empty:
        st.info("Load the Adobe digital CSV (with a MarketingChannel column) to see channel insights.")
    else:
        leaks_df = digital_channels["leaks"]
        channel_funnel_df = digital_channels["channel_funnel"]
        channel_order = scorecard_df["MarketingChannel"].tolist()
        st.markdown("#### Digital Insights")
        show_bullets(build_channel_insight_bullets(digital_channels, totals))

        paid = scorecard_df.dropna(subset=["ROAS"])
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Marketing Channels", f"{len(scorecard_df):,}")
        kpi_cols[1].metric(
            "Scale-tier Revenue Share",
            pct(scorecard_df.loc[scorecard_df["Tier"] == "Scale", "Revenue"].sum(), scorecard_df["Revenue"].sum()),
        )
        kpi_cols[2].metric("Blended Paid ROAS", f"{paid['Revenue'].sum() / paid['MediaCost'].sum():.1f}x" if not paid.empty else "N/A")
        kpi_cols[3].metric("Revenue Lost to Channel Leaks", f"${leaks_df['LostRevenue'].sum() if not leaks_df.empty else 0:,.0f}")

        st.markdown("#### Where Each Channel Drops Off")
        st.caption(
            "Share of visitors who continue to the next step. Red = channel loses more visitors than the site average at that step; "
            "blue = it holds them better."
        )
        st.altair_chart(channel_heatmap(channel_funnel_df, channel_order), width="stretch")

        chart_cols = st.columns(2)
        with chart_cols[0]:
            selected_channel = st.selectbox("Compare a channel's funnel with the site", channel_order)
            st.altair_chart(channel_funnel_compare_chart(channel_funnel_df, selected_channel), width="stretch")
        with chart_cols[1]:
            st.caption("Channel effectiveness: conversion vs revenue per visit (bubble size = visits)")
            st.altair_chart(channel_effectiveness_chart(scorecard_df), width="stretch")

        st.markdown("#### Channel Scorecard")
        st.dataframe(
            scorecard_df[
                [
                    "MarketingChannel", "Tier", "Visits", "Conversions", "ConversionRate", "Revenue", "RevenuePerVisit",
                    "MediaCost", "ROAS", "CostPerSale", "BiggestLeak", "Action",
                ]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "MarketingChannel": st.column_config.TextColumn("Channel"),
                "ConversionRate": st.column_config.NumberColumn("Conversion", format="percent"),
                "Revenue": st.column_config.NumberColumn(format="dollar"),
                "RevenuePerVisit": st.column_config.NumberColumn("Revenue / Visit", format="dollar"),
                "MediaCost": st.column_config.NumberColumn("Media Cost (assumed)", format="dollar"),
                "ROAS": st.column_config.NumberColumn(format="%.1fx"),
                "CostPerSale": st.column_config.NumberColumn("Cost / Sale", format="dollar"),
                "BiggestLeak": st.column_config.TextColumn("Biggest Leak vs Site", width="medium"),
                "Action": st.column_config.TextColumn(width="large"),
            },
        )

        st.markdown("#### Channel Drop-off Leaks (ranked by lost revenue)")
        if leaks_df.empty:
            st.info("No channel trails the site rate by more than the configured threshold at any step.")
        else:
            st.dataframe(
                leaks_df[["MarketingChannel", "LeakStep", "VisitsReached", "ChannelRate", "SiteRate", "LostSales", "LostRevenue", "Fix"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "MarketingChannel": st.column_config.TextColumn("Channel"),
                    "LeakStep": st.column_config.TextColumn("Leak Step", width="medium"),
                    "VisitsReached": st.column_config.NumberColumn("Visits at Step"),
                    "ChannelRate": st.column_config.NumberColumn("Channel Continue", format="percent"),
                    "SiteRate": st.column_config.NumberColumn("Site Continue", format="percent"),
                    "LostSales": st.column_config.NumberColumn("Lost Sales", format="%.1f"),
                    "LostRevenue": st.column_config.NumberColumn("Lost Revenue", format="dollar"),
                    "Fix": st.column_config.TextColumn(width="large"),
                },
            )
        st.download_button(
            "Download channel scorecard CSV",
            dataframe_to_csv(scorecard_df),
            "output_digital_channel_scorecard.csv",
            "text/csv",
        )

with tab_digital_campaign:
    show_tab_header(
        "Digital Sales Campaign Agent",
        "Turns digital abandoners into a prioritized win-back list, using Salesforce value and chat/phone history to pick the right channel and offer.",
    )
    targets_df = digital_campaign.get("targets", pd.DataFrame())
    if not has_digital or targets_df.empty:
        st.info("Load the Adobe digital CSV to build the digital win-back campaign.")
    else:
        retargeting_df = digital_campaign["retargeting"]
        show_bullets(build_digital_campaign_bullets(digital_campaign))
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Known Abandoners", f"{len(targets_df):,}")
        kpi_cols[1].metric("High Priority", f"{(targets_df['Priority'] == 'High').sum():,}")
        kpi_cols[2].metric("Quoted Value at Stake", f"${targets_df['QuoteValue'].sum():,.0f}")
        kpi_cols[3].metric("Expected Recovery", f"${targets_df['ExpectedRevenue'].sum():,.0f}")

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.caption("Expected recovered revenue by outreach channel")
            by_channel = targets_df.groupby("RecommendedChannel", as_index=False)["ExpectedRevenue"].sum()
            st.altair_chart(value_bar_chart(by_channel, "RecommendedChannel", "ExpectedRevenue", "Expected revenue"), width="stretch")
        with chart_cols[1]:
            st.caption("Abandoners by funnel step")
            by_step = targets_df.groupby(["AbandonStep", "AbandonStepName"], as_index=False).size()
            by_step["Step"] = by_step["AbandonStep"].astype(str) + ". " + by_step["AbandonStepName"]
            st.altair_chart(
                alt.Chart(by_step)
                .mark_bar(color="#006fd6", cornerRadiusEnd=3)
                .encode(y=alt.Y("Step:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)), x=alt.X("size:Q", title="Contacts"), tooltip=["Step", "size"])
                .properties(height=240),
                width="stretch",
            )

        st.markdown("#### Win-back Target List")
        filter_cols = st.columns(2)
        priority_filter = filter_cols[0].multiselect("Priority", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        channel_options = sorted(targets_df["RecommendedChannel"].unique())
        channel_filter = filter_cols[1].multiselect("Recommended channel", channel_options, default=channel_options)
        filtered_targets = targets_df[targets_df["Priority"].isin(priority_filter) & targets_df["RecommendedChannel"].isin(channel_filter)]
        st.dataframe(
            filtered_targets[
                [
                    "Priority", "ContactKey", "CustomerType", "AbandonStepName", "Device", "PlanViewed", "QuoteValue",
                    "ClientStatus", "ServiceSentiment", "OpenServiceIssue", "RecommendedChannel", "Reason", "Offer",
                    "ExpectedRevenue",
                ]
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "ContactKey": st.column_config.TextColumn("Client / Lead"),
                "QuoteValue": st.column_config.NumberColumn("Quote Value", format="dollar"),
                "ExpectedRevenue": st.column_config.NumberColumn("Expected Revenue", format="dollar"),
                "Reason": st.column_config.TextColumn(width="large"),
                "Offer": st.column_config.TextColumn(width="large"),
            },
        )
        if not retargeting_df.empty:
            st.markdown("#### Anonymous Retargeting Audiences")
            st.dataframe(
                retargeting_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "AvgQuoteValue": st.column_config.NumberColumn("Avg Quote Value", format="dollar"),
                    "ExpectedRevenue": st.column_config.NumberColumn("Expected Revenue", format="dollar"),
                },
            )
        st.download_button(
            "Download digital campaign CSV",
            dataframe_to_csv(targets_df),
            "output_digital_sales_campaign.csv",
            "text/csv",
        )
