"""
cron_run.py — one-shot ingest trigger for Render Cron Job

Runs the same ingest logic as POST /run, but as a standalone script
suitable for a Render Cron Job (which runs a command, not an HTTP call).

Usage in Render Cron config:
    Command: python cron_run.py
    Schedule: 0 3 * * *   (daily at 03:00 UTC)

Reads the same env vars as main.py.

v0.7.1 · no changes needed here — the agent_md source added to
main.py's run_ingest_all() is picked up automatically.
"""
import asyncio
import json
import sys
from main import run_ingest_all, SERVICE_VER


async def _entry():
    print(f"[{SERVICE_VER}] chainstate-priors cron_run starting ...", flush=True)
    result = await run_ingest_all()
    print(json.dumps(result, indent=2), flush=True)
    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_entry())
