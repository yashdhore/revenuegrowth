"""Generate synthetic Adobe Analytics visit-level data for the digital sales funnel.

Output: data/input_adobe_digital.csv (one row per website/app visit).

Known customers are drawn from the existing Salesforce, chat and phone CSVs so the
digital data joins on ClientID. Friction points are deliberately baked in so the
conversion-opportunity story is visible:
  * Product Comparison -> Application Start is the biggest drop-off.
  * Identity Verification fails far more often on mobile (form errors, slow loads).
  * Payment & e-Sign loses late-stage, high-value quotes.
  * Each marketing channel leaks at a different step (see CHANNELS), e.g. Paid Social
    and Display bounce early, Affiliate traffic stalls at Personal Details, and Direct Mail
    responders drop at the quote because the mailed offer price does not match.

Run:  python scripts/generate_adobe_digital_data.py [--data-dir data]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ANONYMOUS_VISITS = 3000  # enough volume for per-channel funnels to be stable
START, END = pd.Timestamp("2024-05-01"), pd.Timestamp("2024-11-30")

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FUNNEL = [
    "Home Page",
    "Product Comparison",
    "Application Start",
    "Personal Details",
    "Identity Verification",
    "Plan Customization & Quote",
    "Review & e-Sign",
    "Payment & Confirmation",
]
# Maps the digital step reached to the stage names already used in chat/phone/Salesforce.
CRM_STAGE = {
    1: "application educate", 2: "application educate",
    3: "application start", 4: "application start", 5: "application start",
    6: "application submit", 7: "application submit",
    8: "application complete",
}
PAGE_URL = {
    1: "/", 2: "/plans/compare", 3: "/apply/start", 4: "/apply/details",
    5: "/apply/verify", 6: "/apply/quote", 7: "/apply/review", 8: "/apply/confirmation",
}

# Probability of moving from step k to k+1 (index k-1), before modifiers.
BASE_CONTINUE = [0.58, 0.42, 0.82, 0.80, 0.84, 0.76, 0.82]

CHANNELS = {
    # name: (share of anonymous traffic,
    #        multiplier on each step's continue rate for steps 1->2 ... 7->8,
    #        campaign code prefix, share of traffic for known customers relative to anonymous)
    "SEM": (0.22, [1.10, 1.05, 1.02, 1.00, 1.00, 1.00, 1.00], "SEM", 0.8),
    "SEO": (0.20, [0.85, 0.80, 1.05, 1.03, 1.00, 1.02, 1.00], None, 0.8),        # research traffic: browses, then commits
    "Email": (0.09, [1.20, 1.15, 1.05, 1.03, 1.02, 1.05, 1.03], "EM", 3.0),      # warm audience, strongest all the way
    "Direct Mail": (0.08, [1.15, 1.10, 1.03, 1.00, 1.00, 0.70, 1.00], "DM", 2.0),  # offer/quote price mismatch
    "Affiliate": (0.07, [1.05, 1.00, 0.68, 0.95, 1.00, 1.00, 0.95], "AFF", 0.3),  # incentive hunters stall at details
    "Paid Social": (0.14, [0.60, 0.80, 0.95, 0.95, 0.95, 0.95, 0.95], "SOC", 0.5),
    "Display": (0.08, [0.65, 0.85, 0.95, 0.95, 0.95, 0.92, 0.95], "DSP", 0.4),
    "Direct": (0.12, [1.10, 1.05, 1.03, 1.02, 1.00, 1.02, 1.02], None, 2.2),
}
PLANS = {"Basic": (1200, 2400), "Plus": (2400, 4200), "Premium": (4200, 6800), "Elite": (6800, 9800)}
REGIONS = ["North", "East", "South", "West"]

rng = np.random.default_rng(SEED)


def continue_prob(step: int, device: str, channel_mult: list[float], known: bool, returning: bool) -> float:
    p = BASE_CONTINUE[step - 1]
    if step == 5 and device == "Mobile":
        p -= 0.30  # mobile ID verification friction (document upload / OTP)
    if step == 5 and device == "Tablet":
        p -= 0.12
    if step == 7:
        p -= 0.04 if device == "Desktop" else 0.10  # payment / e-sign friction
    p *= channel_mult[step - 1]
    if known:
        p += 0.06
    if returning:
        p += 0.04
    return float(np.clip(p, 0.05, 0.97))


def simulate_visit(device: str, channel: str, known: bool, returning: bool, target_max: int | None = None) -> int:
    """Return the deepest funnel step reached (1-8)."""
    if target_max is not None:
        return target_max
    mult = CHANNELS[channel][1]
    step = 1
    while step < 8 and rng.random() < continue_prob(step, device, mult, known, returning):
        step += 1
    return step


def pick_device() -> str:
    return rng.choice(["Mobile", "Desktop", "Tablet"], p=[0.56, 0.36, 0.08])


def pick_channel(known: bool) -> str:
    names = list(CHANNELS)
    shares = np.array([CHANNELS[c][0] for c in names])
    if known:  # existing customers come back via email / direct / direct mail far more often
        shares = shares * np.array([CHANNELS[c][3] for c in names])
    return str(rng.choice(names, p=shares / shares.sum()))


def campaign_code(channel: str, date: pd.Timestamp, plan: str) -> str:
    prefix = CHANNELS[channel][2]
    if prefix is None:
        return ""
    quarter = f"Q{(date.month - 1) // 3 + 1}"
    theme = rng.choice(["BRAND", "PLANS", plan.upper(), "SWITCH", "RENEW"])
    return f"{prefix}_{theme}_{quarter}{date.year % 100}"


def random_date() -> pd.Timestamp:
    days = (END - START).days
    return START + pd.Timedelta(days=int(rng.integers(0, days + 1)), minutes=int(rng.integers(6 * 60, 23 * 60)))


def build_row(visit_no: int, client: dict | None, lead_id: str, ecid: str, date: pd.Timestamp,
              returning: bool, target_max: int | None = None) -> dict:
    known = client is not None
    device = pick_device()
    channel = pick_channel(known)
    plan = str(rng.choice(list(PLANS), p=[0.30, 0.34, 0.24, 0.12]))
    max_step = simulate_visit(device, channel, known, returning, target_max)
    converted = max_step == 8

    lo, hi = PLANS[plan]
    quote_value = round(float(rng.uniform(lo, hi)), 2) if max_step >= 6 else np.nan
    if known and max_step >= 6:
        quote_value = round(quote_value * rng.uniform(1.0, 1.15), 2)  # upsell for existing customers

    slow = device == "Mobile" and rng.random() < 0.45
    load_sec = round(float(rng.normal(4.6 if slow else 2.1, 0.7)), 1)
    form_errors = 0
    if max_step >= 4:
        lam = 0.6 + (1.8 if device == "Mobile" and max_step == 5 else 0) + (0.8 if not converted else 0)
        form_errors = int(rng.poisson(lam))

    page_views = int(max_step * rng.uniform(1.3, 2.4) + rng.integers(0, 3))
    time_on_site = int(max_step * rng.uniform(45, 110) + rng.integers(10, 60))
    chat_clicked = rng.random() < (0.04 + 0.05 * (max_step in (4, 5, 6, 7)) + 0.04 * (form_errors >= 2))

    return {
        "VisitID": f"AV{visit_no:05d}",
        "AdobeVisitorID": ecid,
        "ClientID": client["ClientID"] if known else "",
        "LeadID": lead_id if (not known and max_step >= 4) else "",
        "CustomerType": "Existing Customer" if known else ("New Lead" if max_step >= 4 else "Anonymous Prospect"),
        "VisitDate": f"{date.month}/{date.day}/{date.year}",  # same M/D/YYYY format as the other inputs
        "VisitStartTime": date.strftime("%H:%M"),
        "Device": device,
        "Browser": str(rng.choice(["Chrome", "Safari", "Edge", "Firefox", "Samsung Internet"],
                                  p=[0.46, 0.34, 0.11, 0.05, 0.04])),
        "NewOrReturning": "Returning" if returning else "New",
        "MarketingChannel": channel,
        "CampaignCode": campaign_code(channel, date, plan),
        "Region": client["Region"] if known else str(rng.choice(REGIONS)),
        "LandingPage": "/offer" if channel == "Direct Mail" else (PAGE_URL[1] if rng.random() < 0.6 or channel == "Direct" else PAGE_URL[2]),
        "FunnelStepNumber": max_step,
        "FunnelStepReached": FUNNEL[max_step - 1],
        "CRMFunnelStage": CRM_STAGE[max_step],
        "ExitPage": "" if converted else PAGE_URL[max_step],
        "Abandoned": "No" if converted else ("Yes" if max_step >= 3 else "Bounced/Browsed"),
        "PageViews": page_views,
        "TimeOnSiteSec": time_on_site,
        "AvgPageLoadSec": max(0.6, load_sec),
        "FormErrors": form_errors,
        "ChatWidgetClicked": "Yes" if chat_clicked else "No",
        "PlanViewed": plan,
        "QuoteValue": quote_value,
        "Converted": "Yes" if converted else "No",
        "OrderID": "",  # filled after sorting
        "Revenue": quote_value if converted else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    data_dir = parser.parse_args().data_dir

    sf = pd.read_csv(data_dir / "input_salesforce_clients.csv")
    sf.columns = sf.columns.str.strip()
    chat_ids = set(pd.read_csv(data_dir / "input_chats.csv")["ClientID"])
    call_ids = set(pd.read_csv(data_dir / "input_phone_calls.csv")["ClientID"])

    # ~70% of Salesforce clients show up digitally; everyone who both chatted AND called is included.
    both = sf[sf["ClientID"].isin(chat_ids & call_ids)]
    rest = sf[~sf["ClientID"].isin(both["ClientID"])].sample(frac=0.45, random_state=SEED)
    digital_clients = pd.concat([both, rest]).to_dict("records")

    rows: list[dict] = []
    visit_no = 1

    for client in digital_clients:
        ecid = rng.bytes(8).hex().upper()
        n_visits = int(rng.choice([1, 2, 3, 4], p=[0.35, 0.35, 0.2, 0.1]))
        dates = sorted(random_date() for _ in range(n_visits))
        for i, date in enumerate(dates):
            target = None
            last = i == n_visits - 1
            # Keep the latest digital visit consistent with where Salesforce says the client is.
            if last and client["FunnelStep"] == "application abandon":
                target = int(rng.choice([3, 4, 5, 5, 6, 7]))
            elif last and client["FunnelStep"] == "application complete" and client["AppStatus"] == "app_approved":
                target = 8 if rng.random() < 0.6 else None
            rows.append(build_row(visit_no, client, "", ecid, date, returning=i > 0, target_max=target))
            visit_no += 1

    lead_no = 1
    for _ in range(N_ANONYMOUS_VISITS):
        ecid = rng.bytes(8).hex().upper()
        returning = rng.random() < 0.28
        row = build_row(visit_no, None, f"L{lead_no:04d}", ecid, random_date(), returning)
        if row["LeadID"]:
            lead_no += 1
        rows.append(row)
        visit_no += 1

    df = pd.DataFrame(rows)
    df["_dt"] = pd.to_datetime(df["VisitDate"] + " " + df["VisitStartTime"])
    df = df.sort_values("_dt").drop(columns="_dt").reset_index(drop=True)
    df["VisitID"] = [f"AV{i + 1:05d}" for i in range(len(df))]
    leads = df["LeadID"] != ""
    df.loc[leads, "LeadID"] = [f"L{i + 1:04d}" for i in range(leads.sum())]
    conv = df["Converted"] == "Yes"
    df.loc[conv, "OrderID"] = [f"ORD{i + 1:05d}" for i in range(conv.sum())]

    out = data_dir / "input_adobe_digital.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} visits to {out}")


if __name__ == "__main__":
    main()
