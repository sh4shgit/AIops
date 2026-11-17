"""Local historical incident search against known log signatures."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from aiops.retrieval.example_incidents import SEED_INCIDENTS
from aiops.retrieval.ingest_incidents import IncidentRecord

_INCIDENTS = TypeAdapter(list[IncidentRecord])
_WORDS = re.compile(r"[a-z0-9]+")


class IncidentMatch(BaseModel):
    incident: IncidentRecord
    matched_template: str
    line_number: int


def load_incidents(path: str) -> list[IncidentRecord]:
    """Load a local incident export or the bundled examples."""
    if not path:
        return _INCIDENTS.validate_python(SEED_INCIDENTS)
    return _INCIDENTS.validate_json(Path(path).read_bytes())


class IncidentSearch:
    """Find the strongest matching signature per active incident."""

    def __init__(self, incidents: list[IncidentRecord]) -> None:
        self.signatures: list[tuple[IncidentRecord, str, list[str], int]] = []
        for incident in incidents:
            if incident.status != "active":
                continue
            for template in incident.log_templates:
                parts = [_WORDS.findall(part.lower()) for part in template.split("<*>")]
                specificity = sum(map(len, parts))
                if specificity >= 2:
                    literals = [" " + " ".join(part) + " " for part in parts if part]
                    self.signatures.append((incident, template, literals, specificity))
        self.signatures.sort(key=lambda signature: signature[3], reverse=True)

    def find(self, lines: list[tuple[int, str]], limit: int = 3) -> list[IncidentMatch]:
        """Match literal segments in order, with gaps only where the template allows."""
        normalized = [(number, " " + " ".join(_WORDS.findall(line.lower())) + " ")
                      for number, line in lines]
        matches: dict[str, IncidentMatch] = {}
        for incident, template, literals, _ in self.signatures:
            if incident.id in matches:
                continue
            for line_number, line in normalized:
                if _matches(line, literals):
                    matches[incident.id] = IncidentMatch(
                        incident=incident, matched_template=template, line_number=line_number
                    )
                    break
            if len(matches) >= limit:
                break
        return list(matches.values())


def _matches(line: str, literals: list[str]) -> bool:
    position = 0
    for literal in literals:
        start = line.find(literal, position)
        if start < 0:
            return False
        position = start + len(literal) - 1
    return True
