"""Locust load test for AIOps Assistant log ingestion endpoint.

Simulates bursty log traffic from multiple microservices to evaluate ingestion
throughput, validation latency, and deduplication efficiency.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

from locust import HttpUser, between, task


class LogIngestionUser(HttpUser):
    """Simulates a microservice reporting log batches to /logs."""

    wait_time = between(0.05, 0.2)  # fast bursts

    SERVICES = [
        "hdfs-datanode",
        "api-gateway",
        "auth-service",
        "payment-worker",
        "postgres-db",
    ]

    TEMPLATES = [
        ("INFO", "Processed request {req_id} status 200 latency {lat}ms"),
        ("INFO", "Receiving block blk_{blk} src: 10.0.0.{ip}:50010 dest: 10.0.0.{ip}:50010"),
        ("WARN", "High memory usage detected: {pct}% utilized"),
        ("ERROR", "Connection reset by peer from downstream host 10.0.0.{ip}"),
        ("DEBUG", "Cache lookup for key user:{uid} hit=true"),
    ]

    @task(5)
    def ingest_valid_batch(self) -> None:
        """Send a standard batch of 10-25 log entries."""
        batch_size = random.randint(10, 25)
        service = random.choice(self.SERVICES)
        batch = []

        now = datetime.now(timezone.utc).isoformat()
        for i in range(batch_size):
            lvl, tmpl = random.choice(self.TEMPLATES)
            msg = tmpl.format(
                req_id=f"req-{random.randint(1000, 9999)}",
                lat=random.randint(5, 250),
                blk=random.randint(100000, 999999),
                ip=random.randint(1, 254),
                pct=random.randint(70, 95),
                uid=random.randint(1, 500),
            )
            batch.append({
                "timestamp": now,
                "service": service,
                "level": lvl,
                "message": msg,
            })

        self.client.post(
            "/logs",
            json=batch,
            headers={"X-API-Key": "test-secret-key"},
            name="/logs (valid batch)",
        )

    @task(1)
    def ingest_burst_with_duplicates(self) -> None:
        """Send duplicate logs to stress deduplication via Redis SET."""
        now = "2026-10-28T12:00:00Z"
        fixed_batch = [
            {
                "timestamp": now,
                "service": "api-gateway",
                "level": "INFO",
                "message": f"Fixed repeated log message {i}",
            }
            for i in range(5)
        ]
        self.client.post(
            "/logs",
            json=fixed_batch,
            headers={"X-API-Key": "test-secret-key"},
            name="/logs (duplicate burst)",
        )

    @task(1)
    def check_health(self) -> None:
        """Check system health during load."""
        self.client.get("/health", name="/health")
