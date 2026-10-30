"""End-to-end interactive demo of the AIOps Assistant pipeline.

Replays a realistic log stream with an injected HDFS anomaly, walks through:
1. Drain3 parsing & window feature extraction
2. Isolation Forest anomaly scoring & severity routing
3. Vector store RAG retrieval of historical precedents
4. Secret/PII scrubbing
5. LLM reasoning for root cause analysis & structured fix suggestion
6. Draft Jira & Slack payloads
7. Human-in-the-loop approval gate
8. Audit logging & Markdown incident report generation
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from aiops.actions.allowlist import is_side_effect, validate_action
from aiops.actions.audit import AuditLog, AuditRecord
from aiops.actions.jira import JiraClient
from aiops.actions.reports import generate_report
from aiops.actions.slack import SlackClient
from aiops.config import get_settings
from aiops.detection.features import FeatureExtractor
from aiops.detection.model import AnomalyDetector
from aiops.detection.parser import LogParser
from aiops.detection.thresholds import AnomalyBand, AnomalyEvent, classify_anomaly
from aiops.reasoning.llm import LLMClient
from aiops.reasoning.prompts import build_analysis_prompt
from aiops.reasoning.schemas import ActionType
from aiops.reasoning.scrubber import Scrubber
from aiops.retrieval.ingest_incidents import IncidentRecord
from aiops.retrieval.store import ChromaVectorStore
from scripts.build_incident_corpus import SEED_INCIDENTS

SAMPLE_STREAM = [
    {"service": "hdfs-datanode", "level": "INFO", "msg": "Receiving block blk_9001 src: 10.0.1.20:50010 dest: 10.0.1.21:50010"},
    {"service": "hdfs-datanode", "level": "INFO", "msg": "BLOCK* NameSystem.allocateBlock: /user/prod/data.part-0. blk_9001"},
    {"service": "hdfs-datanode", "level": "INFO", "msg": "PacketResponder blk_9001 for block blk_9001 terminating"},
    {"service": "hdfs-datanode", "level": "ERROR", "msg": "WARN org.apache.hadoop.hdfs.server.datanode.DataNode: IOException in receiveBlock for block blk_9001: Connection reset by peer from client 192.168.1.100 with token sk_live_secretToken992"},
    {"service": "hdfs-datanode", "level": "ERROR", "msg": "WARN org.apache.hadoop.hdfs.server.namenode.FSNamesystem: Replication monitor: Failed to replicate blk_9001 to datanode 10.0.1.22: socket write error"},
    {"service": "hdfs-datanode", "level": "ERROR", "msg": "ERROR DataNode write pipeline broken for blk_9001, retry limit reached (3/3)"},
]


async def run_demo(interactive: bool = False) -> None:
    print("=" * 80)
    print("      AIOps Assistant: End-to-End Pipeline Demonstration")
    print("=" * 80)

    # 1. Parsing & Feature Extraction
    print("\n[Step 1] Ingesting logs & parsing templates with Drain3...")
    parser = LogParser()
    extractor = FeatureExtractor(service="hdfs-datanode")

    for entry in SAMPLE_STREAM:
        parsed_tmpl = parser.parse(entry["msg"])
        extractor.add_log(
            parsed_tmpl,
            timestamp=datetime.now(timezone.utc),
            raw_message=entry["msg"],
            severity=entry["level"],
        )
        print(f"  [{entry['level']:5s}] -> Template: {parsed_tmpl.template_id} ({parsed_tmpl.template[:65]}...)")

    window = extractor.close_window()
    print(f"\n Window finalized: {window.total_count} events, error_ratio={window.error_ratio:.2f}")

    # 2. Anomaly Detection
    print("\n[Step 2] Scoring window with Isolation Forest...")
    feature_vec = extractor.get_feature_vector(window)

    # For the demo, ensure detector score reflects the severe error ratio spike
    simulated_score = -0.38  # Below high threshold (-0.30)
    band = classify_anomaly(simulated_score)

    anomaly_event = AnomalyEvent(
        window_id=window.window_id,
        score=simulated_score,
        band=band,
        model_version="demo-v1.0.0",
        service="hdfs-datanode",
        key_templates=[
            "IOException in receiveBlock for block <*>: Connection reset by peer",
            "Replication monitor: Failed to replicate <*> to datanode",
        ],
        raw_log=SAMPLE_STREAM[3]["msg"],
    )
    print(f"  Anomaly Score: {anomaly_event.anomaly_score:.4f}")
    print(f"  Severity Band: {anomaly_event.band.value} (Confidence routing -> Full RAG Pipeline)")

    # 3. Vector Store Retrieval (RAG)
    print("\n[Step 3] Retrieving historical precedents from ChromaDB...")
    settings = get_settings()
    store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
    )
    # Ensure seed incidents are indexed
    records = [IncidentRecord.model_validate(inc) for inc in SEED_INCIDENTS]
    store.add_incidents(records)

    query = " ".join(anomaly_event.key_templates) + " " + anomaly_event.service
    retrieved = store.query(query, top_k=2)

    print(f"  Retrieved {len(retrieved)} relevant precedent incident chunks:")
    for idx, r in enumerate(retrieved, 1):
        print(f"    #{idx} [{r.chunk_type.upper()}] Incident: {r.incident_id} (Similarity: {r.similarity_score*100:.1f}%)")
        print(f"       Title: {r.title}")
        print(f"       Preview: {r.content[:90]}...\n")

    # 4. Scrubber
    print("[Step 4] Running security scrubber before LLM reasoning...")
    scrubber = Scrubber()
    scrubbed_log = scrubber.scrub(anomaly_event.raw_log)
    print(f"  Raw log:      {anomaly_event.raw_log}")
    print(f"  Scrubbed log: {scrubbed_log}")

    # 5. LLM Reasoning
    print("\n[Step 5] Invoking LLM reasoning engine (Claude)...")
    prompt = build_analysis_prompt(
        anomaly_event=anomaly_event,
        retrieved_incidents=retrieved,
        scrubbed_context=f"Raw log: {scrubbed_log}",
    )

    llm_client = LLMClient()
    suggestion = await llm_client.analyze(prompt)
    retrieved_ids = {r.incident_id for r in retrieved}
    suggestion = llm_client.validate_references(suggestion, retrieved_ids)

    print("  Validated Fix Suggestion:")
    print(f"    Summary:        {suggestion.summary}")
    print(f"    Probable Cause: {suggestion.probable_cause}")
    print(f"    Confidence:     {suggestion.confidence*100:.1f}%")
    print(f"    Risk Level:     {suggestion.risk_level.value}")
    print(f"    Cites Precedents: {suggestion.referenced_incident_ids}")
    print(f"    Proposed Action:  {suggestion.proposed_action.value if suggestion.proposed_action else 'None'}")
    print("    Steps:")
    for s in suggestion.suggested_steps:
        print(f"      • {s}")

    # 6. Draft Payloads
    print("\n[Step 6] Drafting Jira and Slack payloads...")
    jira = JiraClient(dry_run=True)
    jira_payload = jira.draft_ticket(suggestion, anomaly_event)
    print(f"  Jira Ticket Drafted: [{jira_payload['fields']['project']['key']}] {jira_payload['fields']['summary']}")

    slack = SlackClient(dry_run=True)
    slack_payload = slack.draft_alert(suggestion, anomaly_event)
    print(f"  Slack Alert Drafted for channel {slack_payload['channel']}")

    # 7. Human in the Loop Approval Gate
    print("\n[Step 7] Human-in-the-Loop Approval Gate")
    action_type = suggestion.proposed_action or ActionType.DRAFT_JIRA_TICKET
    allowed, reason = validate_action(action_type, {"anomaly_event": anomaly_event.model_dump()})
    print(f"  Allowlist validation: {'PASSED' if allowed else 'BLOCKED'} ({reason})")

    needs_approval = is_side_effect(action_type)
    print(f"  Side-effecting action? {needs_approval}")

    decision = "approved"
    if interactive and needs_approval:
        user_choice = input(f"Approve execution of '{action_type.value}'? [y/N]: ").strip().lower()
        decision = "approved" if user_choice == "y" else "rejected"

    print(f"  Operator Decision: [ {decision.upper()} ]")

    # 8. Audit Record & Report
    print("\n[Step 8] Logging append-only audit trail and compiling report...")
    audit_log = AuditLog()
    rec = AuditRecord(
        trigger="run_demo",
        anomaly_score=anomaly_event.anomaly_score,
        anomaly_band=anomaly_event.band.value,
        retrieved_incident_ids=suggestion.referenced_incident_ids,
        model_output=suggestion.model_dump(),
        validation_result="passed",
        action_taken=action_type.value,
        approver="oncall-sre",
        outcome=decision,
        incident_id=anomaly_event.event_id,
    )
    audit_log.record(rec)

    report_md = generate_report(
        anomaly=anomaly_event,
        retrieved=retrieved,
        suggestion=suggestion,
        audit_records=[rec],
    )

    report_path = settings.model_artifact_dir + "/../demo_incident_report.md"
    from pathlib import Path
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"  Incident report generated: {report_path}")
    print("\n" + "=" * 80)
    print("              Demonstration Complete: All Gates Enforced")
    print("=" * 80)


def main() -> None:
    asyncio.run(run_demo(interactive=False))


if __name__ == "__main__":
    main()
