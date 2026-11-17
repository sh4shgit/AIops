"""Verify the dashboard recommendation contract without provider credentials."""

import json

import httpx
import pytest

from aiops.api.main import create_app
from aiops.api.routes.datasets import hdfs_sample
from aiops.reasoning.groq import GROQ_COMPLETIONS_URL


@pytest.mark.asyncio
async def test_keyless_analysis_does_not_enable_generation(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    request = {"logs": hdfs_sample(offset=77, limit=1)["logs"]}
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        analysis = await client.post("/analyze", json=request)
        response = await client.post("/analyze/recommendation", json=request)
    assert analysis.status_code == 200
    assert not analysis.json()["recommendation_available"]
    assert response.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("citation, expected_status", [("S1", 200), ("S999", 502)])
async def test_generation_retrieves_redacts_and_checks_citations(monkeypatch, citation, expected_status):
    monkeypatch.setenv("GROQ_API_KEY", "test-provider-key")
    monkeypatch.setenv("INCIDENT_CORPUS_PATH", "")
    warning = hdfs_sample(offset=77, limit=1)["logs"][0]
    warning["message"] += " password=\"secret with spaces\" contact=operator@example.com"
    original_send = httpx.AsyncClient.send

    async def send(client, request, **kwargs):
        if str(request.url) != GROQ_COMPLETIONS_URL:
            return await original_send(client, request, **kwargs)
        body = json.loads(request.content)
        evidence = json.loads(body["messages"][1]["content"])
        evidence_text = json.dumps(evidence)
        assert "secret with spaces" not in evidence_text
        assert "operator@example.com" not in evidence_text
        assert "10.251.30.85" not in evidence_text
        assert evidence["incidents"][0]["title"] == "Interrupted DataNode block read"
        output_schema = body["response_format"]["json_schema"]
        assert output_schema["strict"]
        assert output_schema["schema"]["properties"]["cited_sources"]["items"]["enum"] == ["S1"]
        assert request.headers["authorization"] == "Bearer test-provider-key"
        return httpx.Response(
            200,
            json={"choices": [{
                "finish_reason": "stop",
                "message": {"content": json.dumps({
                    "status": "recommended",
                    "summary": "Check the client-side read failure.",
                    "probable_cause": "The client may have interrupted the block read.",
                    "suggested_steps": ["Compare client and DataNode logs for the block."],
                    "cited_sources": [citation],
                })},
            }]},
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        response = await client.post("/analyze/recommendation", json={"logs": [warning]})

    assert response.status_code == expected_status
    if expected_status == 200:
        result = response.json()
        assert result["sources"][0]["incident_id"] == "HDFS-EXAMPLE-01"
        assert result["sources"][0]["cited"]
        assert result["sources"][0]["line_number"] == 1
    else:
        assert "unsupported citations" in response.json()["detail"]
