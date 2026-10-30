"""Slack integration client for AIOps Assistant.

Drafts Block Kit notifications and posts alerts to incident channels.
Supports dry-run mode where message payloads are generated without network requests.
"""

from __future__ import annotations

from typing import Any

import structlog

from aiops.config import get_settings
from aiops.detection.thresholds import AnomalyEvent
from aiops.reasoning.schemas import FixSuggestion

logger = structlog.get_logger()


class SlackClient:
    """Client for building and posting Slack alert messages."""

    def __init__(
        self,
        bot_token: str | None = None,
        channel: str | None = None,
        dry_run: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.bot_token = bot_token or (settings.slack_bot_token.get_secret_value() if settings.slack_bot_token else "")
        self.channel = channel or settings.slack_channel
        self.dry_run = dry_run if dry_run is not None else settings.dry_run

    def draft_alert(
        self,
        suggestion: FixSuggestion,
        anomaly: AnomalyEvent,
    ) -> dict[str, Any]:
        """Construct a rich Slack Block Kit message payload."""
        badge = ":red_circle: HIGH ANOMALY" if anomaly.band.value == "HIGH" else ":warning: LOW ANOMALY"
        refs = ", ".join(f"`{r}`" for r in suggestion.referenced_incident_ids) or "_None cited_"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{badge} Detected in {anomaly.service}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Summary:* {suggestion.summary}\n"
                        f"*Probable Cause:* {suggestion.probable_cause}\n"
                        f"*Score:* `{anomaly.anomaly_score:.4f}` | *Risk:* `{suggestion.risk_level.value}` | *Confidence:* `{suggestion.confidence*100:.0f}%`"
                    ),
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Precedent Incidents:* {refs}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Suggested Remediation:*\n"
                    + "\n".join(f"• {s}" for s in suggestion.suggested_steps[:4]),
                },
            },
            {"type": "divider"},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Event ID: `{anomaly.event_id}` | Time: `{anomaly.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}`",
                    }
                ],
            },
        ]

        return {
            "channel": self.channel,
            "text": f"AIOps Alert: {anomaly.service} anomaly detected",
            "blocks": blocks,
        }

    async def send_alert(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send the alert to Slack, or return the draft payload in dry_run mode."""
        if self.dry_run or not self.bot_token:
            logger.info("slack_alert_dry_run", channel=self.channel, dry_run=self.dry_run)
            return {
                "ok": True,
                "status": "drafted",
                "channel": self.channel,
                "payload": payload,
            }

        try:
            from slack_sdk.web.async_client import AsyncWebClient

            client = AsyncWebClient(token=self.bot_token)
            response = await client.chat_postMessage(
                channel=payload.get("channel", self.channel),
                text=payload.get("text", "AIOps Alert"),
                blocks=payload.get("blocks"),
            )
            logger.info("slack_alert_sent", ts=response.get("ts"))
            return {
                "ok": True,
                "status": "sent",
                "ts": response.get("ts"),
                "channel": self.channel,
            }
        except Exception as e:
            logger.error("slack_alert_failed", error=str(e))
            raise
