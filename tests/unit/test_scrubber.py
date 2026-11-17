"""Unit tests for the PII and secret scrubber."""

from __future__ import annotations

from aiops.reasoning.prompts import build_analysis_prompt
from aiops.reasoning.scrubber import Scrubber
from aiops.retrieval.store import RetrievalResult


def test_scrub_ip_addresses() -> None:
    scrubber = Scrubber()
    text = "Connection failed from 192.168.1.105:50010 to 10.0.0.1"
    scrubbed = scrubber.scrub(text)
    assert "192.168.1.105" not in scrubbed
    assert "10.0.0.1" not in scrubbed
    assert "[IP_REDACTED]" in scrubbed


def test_scrub_email() -> None:
    scrubber = Scrubber()
    text = "Alert dispatched to oncall-lead@company.internal for review"
    scrubbed = scrubber.scrub(text)
    assert "oncall-lead@company.internal" not in scrubbed
    assert "[EMAIL_REDACTED]" in scrubbed


def test_scrub_api_keys_and_tokens() -> None:
    scrubber = Scrubber()
    text = "Received auth token sk-ant-api03-abcdef1234567890 and slack xoxb-12345678-abcdef"
    scrubbed = scrubber.scrub(text)
    assert "sk-ant-api03" not in scrubbed
    assert "xoxb-12345678" not in scrubbed
    assert "[API_KEY_REDACTED]" in scrubbed or "[SECRET_REDACTED]" in scrubbed


def test_scrub_aws_access_key() -> None:
    scrubber = Scrubber()
    text = "Upload failed with AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
    scrubbed = scrubber.scrub(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed


def test_scrub_dict_recursive() -> None:
    scrubber = Scrubber()
    data = {
        "service": "api-gateway",
        "nested": {
            "client_ip": "172.16.0.45",
            "contact": "admin@service.io",
        },
        "count": 42,
    }
    scrubbed = scrubber.scrub_dict(data)
    assert scrubbed["nested"]["client_ip"] == "[IP_REDACTED]"
    assert scrubbed["nested"]["contact"] == "[EMAIL_REDACTED]"
    assert scrubbed["count"] == 42


def test_prompt_redacts_logs_incident_content_and_additional_context(sample_anomaly_event):
    sample_anomaly_event.raw_log = "Connection failed from 192.168.1.105"
    precedent = RetrievalResult(
        incident_id="INC-1001",
        chunk_type="symptoms",
        similarity_score=0.9,
        metadata={"title": "Reported by operator@example.com"},
        content="Authorization: Bearer example-secret-token",
    )
    prompt = build_analysis_prompt(sample_anomaly_event, [precedent], "Peer at 172.16.0.45")
    assert "INC-1001" in prompt
    for secret in ("192.168.1.105", "operator@example.com", "example-secret-token", "172.16.0.45"):
        assert secret not in prompt
