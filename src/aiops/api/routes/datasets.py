"""Read a bounded slice of the bundled LogHub HDFS sample."""

from datetime import datetime
from functools import lru_cache
from pathlib import Path
import re

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["datasets"])
DATASET = Path(__file__).resolve().parents[4] / "data" / "samples" / "HDFS_2k.txt"
SOURCE = "https://github.com/logpai/loghub/blob/master/HDFS/HDFS_2k.log"
LINE = re.compile(r"^(\d{6})\s+(\d{6})\s+\d+\s+(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\s+(.*)$")


def parse_hdfs_line(line: str) -> dict[str, str]:
    """Parse LogHub's HDFS format while preserving the original line."""
    match = LINE.match(line.strip())
    if not match:
        return {"timestamp": "", "service": "hdfs", "level": "INFO", "message": line.strip(), "raw": line}
    date, clock, level, message = match.groups()
    try:
        timestamp = datetime.strptime(date + clock, "%y%m%d%H%M%S").isoformat()
    except ValueError:
        timestamp = ""
    return {"timestamp": timestamp, "service": "hdfs", "level": level, "message": message, "raw": line}


@lru_cache(maxsize=1)
def load_sample() -> list[str]:
    return DATASET.read_text(encoding="utf-8").splitlines()


@router.get("/datasets/hdfs")
def hdfs_sample(offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=500)) -> dict:
    try:
        lines = load_sample()
    except OSError as exc:
        raise HTTPException(status_code=503, detail="HDFS sample is unavailable") from exc
    return {"name": "LogHub HDFS 2k", "source": SOURCE, "total": len(lines),
            "offset": offset, "limit": limit,
            "logs": [parse_hdfs_line(line) for line in lines[offset:offset + limit]]}
