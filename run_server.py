#!/usr/bin/env python
"""Quick-start script: launch the AIOps web dashboard locally.

Usage:
    python run_server.py              # starts on http://localhost:8000
    python run_server.py --port 9000  # custom port
    python run_server.py --reload     # auto-reload on code changes (dev mode)

The web UI is available at:  http://localhost:8000/
API docs are at:             http://localhost:8000/docs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the src/ package root is importable when run from the repo root
_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import uvicorn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AIOps Assistant web server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()

    print(f"\n🛡  AIOps Assistant starting...")
    print(f"   Web UI  → http://localhost:{args.port}/")
    print(f"   API docs→ http://localhost:{args.port}/docs")
    print(f"   Health  → http://localhost:{args.port}/health\n")

    uvicorn.run(
        "aiops.api.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
