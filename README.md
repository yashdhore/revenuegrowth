# KP Sales Agents Streamlit

Streamlit app that runs three CSV-backed sales agents without Cosmos DB.

## Architecture

```text
GitHub raw CSV URLs
        |
        v
Streamlit UI
        |
        v
GitHubCsvLoader
        |
        v
InsightsAgent -> PerformanceAgent
             \-> SalesCampaignAgent
             \-> DigitalSalesCampaignAgent <- DigitalConversionAgent (Adobe digital)
```

## Agents

- `InsightsAgent`: combines chat, phone call, and Salesforce client CSV data into interaction insights.
- `PerformanceAgent`: scores agent performance from the generated insights.
- `SalesCampaignAgent`: recommends the best follow-up channel for unresolved or at-risk customer interactions.
- `DigitalConversionAgent`: diagnoses the 8-step Adobe digital funnel and sizes conversion opportunities by step, device and marketing channel.
- `DigitalChannelAgent`: compares the digital funnel by marketing channel (SEM, SEO, Email, Direct Mail, Affiliate, Paid Social, Display, Direct), finds where each channel drops off versus the site, and scores channel ROAS using assumed media costs.
- `DigitalSalesCampaignAgent`: builds a prioritized win-back list of digital abandoners, using Salesforce and chat/phone history to choose channel and offer.

## Expected CSV Files

The app reads these files from the local `data/` folder first, then falls back to raw GitHub URLs:

- chats (`input_chats.csv`)
- phone calls (`input_phone_calls.csv`)
- Salesforce clients (`input_salesforce_clients.csv`)
- Adobe digital visits (`input_adobe_digital.csv`, optional; powers the digital tabs)

Regenerate all synthetic data with `python scripts/generate_all.py`.

Use raw file URLs, not normal GitHub blob page URLs. A raw URL usually looks like:

```text
https://raw.githubusercontent.com/<owner>/<repo>/<branch>/data/input_chats.csv
```

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Configuration

Default CSV URLs and agent settings live in `config/agents_config.yaml`.

Create a local `.env` file for backend LLM settings:

```text
OPENAI_API_KEY=<your key>
OPENAI_MODEL=gpt-4o-mini
LLM_ENABLED=true
```

The `.env` file is intentionally ignored by Git because it can contain secrets. Use `.env.example` as the template.

Cosmos DB is not used in this version.

## Dashboard Tabs

- `Insights`: converts raw customer conversations into sentiment, intent, resolution status, risk, and next-best-action signals.
- `Performance`: rolls up insight outcomes by agent to show resolution quality, customer sentiment, and coaching opportunities.
- `Sales Campaign`: recommends follow-up channels for customers who need additional outreach to reduce churn risk or improve conversion.
