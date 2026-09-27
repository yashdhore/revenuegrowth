from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from agents import (
    DigitalChannelAgent,
    DigitalConversionAgent,
    DigitalSalesCampaignAgent,
    InsightsAgent,
    PerformanceAgent,
    SalesCampaignAgent,
)
from agents.data_loader import GitHubCsvLoader, GitHubCsvSources, load_local_data, local_data_available


def load_config(path: str = "config/agents_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def find_agent_config(config: dict, name: str) -> dict:
    return next(agent for agent in config["agents"] if agent["name"] == name)


def sources_from_config(source_config: dict) -> GitHubCsvSources:
    return GitHubCsvSources(
        chats_url=source_config.get("chats_url", ""),
        phone_calls_url=source_config.get("phone_calls_url", ""),
        salesforce_clients_url=source_config.get("salesforce_clients_url", ""),
        adobe_digital_url=source_config.get("adobe_digital_url", ""),
    )


def load_default_frames(config: dict) -> dict[str, pd.DataFrame]:
    """Local data folder first, then the GitHub URLs from config."""
    source_config = config["data_sources"]
    local_dir = source_config.get("local_data_dir", "")
    if local_dir and local_data_available(local_dir):
        return load_local_data(local_dir)
    return GitHubCsvLoader(sources_from_config(source_config)).load_all()


def run_pipeline(frames: dict[str, pd.DataFrame], config: dict) -> dict:
    insights = InsightsAgent(find_agent_config(config, "InsightsAgent")).run(
        frames["chats"],
        frames["phone_calls"],
        frames["salesforce_clients"],
    )
    digital = frames.get("adobe_digital", pd.DataFrame())
    return {
        "insights": insights,
        "performance": PerformanceAgent(find_agent_config(config, "PerformanceAgent")).run(insights),
        "campaign": SalesCampaignAgent(find_agent_config(config, "SalesCampaignAgent")).run(insights, digital),
        "digital": DigitalConversionAgent(find_agent_config(config, "DigitalConversionAgent")).run(digital),
        "digital_channels": DigitalChannelAgent(
            find_agent_config(config, "DigitalChannelAgent"),
            step_playbook=find_agent_config(config, "DigitalConversionAgent").get("step_playbook", {}),
        ).run(digital),
        "digital_campaign": DigitalSalesCampaignAgent(find_agent_config(config, "DigitalSalesCampaignAgent")).run(
            digital, frames["salesforce_clients"], insights
        ),
    }


def run_all(config_path: str = "config/agents_config.yaml", sources: GitHubCsvSources | None = None) -> dict:
    config = load_config(config_path)
    frames = GitHubCsvLoader(sources).load_all() if sources else load_default_frames(config)
    return run_pipeline(frames, config)


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    result = run_all()
    for name, value in result.items():
        if isinstance(value, dict):
            for part, df in value.items():
                print(f"{name}.{part}: {len(df)} rows")
        else:
            print(f"{name}: {len(value)} rows")
