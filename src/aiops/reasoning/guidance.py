"""Read-only investigation suggestions keyed to observed symptoms."""

import re

from pydantic import BaseModel


class InvestigationGuidance(BaseModel):
    why: str
    steps: list[str]
    line_number: int | None = None
    matched_text: str | None = None


_RULES = [
    (
        r"exception while serving|connection reset by peer|broken pipe|failed to replicate",
        "Client and peer logs help distinguish an interrupted transfer from a recurring DataNode problem.",
        "Compare the DataNode warning with the client and peer logs for the same block.",
        "Check whether the transfer retried successfully and whether other blocks or peers are affected.",
    ),
    (
        r"checksum(?:exception| error| mismatch)|corrupted block",
        "Repeated checksum failures on one replica or volume help narrow the problem to specific storage.",
        "Check whether the checksum error repeats for the same block, replica, or storage volume.",
        "Review replica health and disk error counters for the affected DataNode.",
    ),
    (
        r"timed out|\btimeout\b|\bheartbeat\b",
        "The timing and spread of a timeout help distinguish a single connection problem from a wider service slowdown.",
        "Compare timeouts with network latency, JVM pauses, and service saturation at the same time.",
        "Check whether the problem affects a single peer or multiple clients and DataNodes.",
    ),
    (
        r"token expired|clock skew|jwt verification failed",
        "Clock differences and issuer settings can cause services to reject otherwise valid tokens.",
        "Check clock synchronization and token issuer configuration on the affected services.",
    ),
    (
        r"\bmaxmemory\b|too many clients",
        "A fall in cache hits can increase database traffic and exhaust available connections.",
        "Review cache evictions, cache hit rate, and database connection use around the first error.",
    ),
]


def explain_checks(lines: list[tuple[int, str]]) -> list[InvestigationGuidance]:
    guidance: list[InvestigationGuidance] = []
    for pattern, why, *steps in _RULES:
        for number, message in lines:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                guidance.append(InvestigationGuidance(
                    why=why, steps=steps, line_number=number, matched_text=match.group()
                ))
                break
    return guidance or [InvestigationGuidance(
        why="The window crossed a severity threshold. No known symptom rule matched, so these checks help establish context.",
        steps=[
            "Read the full warning or error, including any stack trace and surrounding lines.",
            "Check whether the same pattern recurs for the block or service in a wider time range.",
        ],
    )]
