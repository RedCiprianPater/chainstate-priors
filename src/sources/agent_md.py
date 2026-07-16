"""
CHAINSTATE PRIORS · agent_md source family
==========================================
New source module — nightly ingest of agent.md from ecosystem HF Spaces.

Deploy: drop this file at src/sources/agent_md.py in your chainstate-priors repo.

Then in src/ingest.py, add:

    from sources.agent_md import ingest_agent_md
    ...
    if os.getenv("INGEST_AGENT_MD_SPACES"):
        results["agent_md"] = ingest_agent_md(kv_writer, encoder_client)

That's the only change to your existing ingester wiring.
"""

import os
import re
import time
import hashlib
import logging
from typing import List, Dict, Optional
import httpx


LOG = logging.getLogger("chainstate.priors.agent_md")

# Standard HF Space content URL pattern.
# For non-static Spaces the URL is https://<slug>.hf.space/<file>
# For static Spaces the URL is https://<slug>.static.hf.space/<file>
#
# BUT — the canonical machine-readable path for content stored in a Space's
# git repo is always:
#
#   https://huggingface.co/spaces/<owner>/<repo>/resolve/main/<path>
#
# This is what we hit. It works whether the Space is static or dynamic and
# it survives frontend rewrites.

HF_RESOLVE_TEMPLATE = "https://huggingface.co/spaces/{owner}/{repo}/resolve/main/{path}"

# Env var format:
#   INGEST_AGENT_MD_SPACES=CPater/nwo-agentic:agent.md,CPater/nwo.apocalypse:agent.md,...
# Each entry is "owner/repo:filename". filename defaults to agent.md if omitted.
DEFAULT_SPACES = [
    # (owner, repo, filename)
    ("CPater", "nwo-agentic", "agent.md"),
    ("CPater", "nwo.apocalypse", "agent.md"),
    ("CPater", "nwo-gateway", "agent.md"),
    ("CPater", "nwo-blackbox", "agent.md"),
    ("CPater", "nwo-anon", "agent.md"),
    ("CPater", "nwo-neuro", "agent.md"),
    ("CPater", "nwo-cardiac", "agent.md"),
    ("CPater", "nwo-asm", "agent.md"),
    ("CPater", "nwo-oracle", "agent.md"),
    ("CPater", "nwo-zeropoint", "agent.md"),
    ("CPater", "nwo-coanda", "agent.md"),
    ("CPater", "nwo-ubi", "agent.md"),
    ("CPater", "nwo-asi", "agent.md"),
    ("CPater", "metastate", "agent.md"),
    ("CPater", "imperium-romanum", "agent.md"),
    ("CPater", "nwo-capital", "agent.md"),
    ("CPater", "ornith-chainstate", "AGENT.md"),
    ("CPater", "nwo-rwa", "agent.md"),
    ("CPater", "nwo-mixed-reality", "agent.md"),
]

# Guards — same as FETCH opcode
MAX_BYTES = 500_000       # 500 KB body cap
TIMEOUT_S = 15
STRIP_MAX = 20_000        # post-strip 20 KB plaintext cap for encoder

# TTL — longer than Wikipedia/arXiv because agent.md changes rarely
AGENT_MD_TTL_SECONDS = 60 * 60 * 24 * 30   # 30 days


def _parse_spaces_env() -> List[tuple]:
    """
    Parse INGEST_AGENT_MD_SPACES env var. Falls back to DEFAULT_SPACES.

    Format: "owner1/repo1:file1,owner2/repo2:file2,..."
    file defaults to "agent.md" if not specified.
    """
    raw = os.getenv("INGEST_AGENT_MD_SPACES", "").strip()
    if not raw:
        return DEFAULT_SPACES

    result = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        # "owner/repo:file" or "owner/repo"
        if ":" in entry:
            path, fname = entry.rsplit(":", 1)
        else:
            path, fname = entry, "agent.md"
        if "/" not in path:
            LOG.warning("skipping malformed entry: %r", entry)
            continue
        owner, repo = path.split("/", 1)
        result.append((owner.strip(), repo.strip(), fname.strip()))
    return result or DEFAULT_SPACES


def _fetch_one(client: httpx.Client, owner: str, repo: str, path: str) -> Optional[str]:
    """
    Fetch one agent.md from HuggingFace with guards.
    Returns text (post-strip, capped) or None on any error.
    """
    url = HF_RESOLVE_TEMPLATE.format(owner=owner, repo=repo, path=path)
    try:
        resp = client.get(url, timeout=TIMEOUT_S, follow_redirects=True)
    except httpx.HTTPError as e:
        LOG.info("agent_md fetch failed %s: %s", url, e)
        return None

    if resp.status_code != 200:
        LOG.info("agent_md missing %s (%d)", url, resp.status_code)
        return None

    # Byte cap
    body = resp.content[:MAX_BYTES]
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return None

    # Post-strip cap for encoder — no HTML stripping needed for markdown
    return text[:STRIP_MAX]


def _slug(owner: str, repo: str) -> str:
    """
    Priors key slug. Lowercase, no dots, no slashes.
    Example: cpater-nwo-apocalypse
    """
    s = f"{owner}-{repo}".lower().replace("/", "-").replace(".", "-")
    s = re.sub(r"[^a-z0-9_-]", "-", s)
    return s.strip("-")


def _summary_preview(text: str, max_len: int = 280) -> str:
    """
    First non-empty content paragraph after any YAML frontmatter, capped.
    """
    # Strip YAML frontmatter if present
    if text.lstrip().startswith("---"):
        m = re.search(r"^---\s*\n.*?\n---\s*\n", text, re.DOTALL | re.MULTILINE)
        if m:
            text = text[m.end():]
    # Strip markdown headings
    lines = [ln for ln in text.split("\n") if ln.strip() and not ln.lstrip().startswith("#")]
    # Take first non-empty prose line and stitch until we hit the cap
    out = " ".join(lines[:6])
    out = re.sub(r"\s+", " ", out).strip()
    return out[:max_len]


def ingest_agent_md(kv_writer, encoder_client) -> Dict:
    """
    Nightly ingest of agent.md from ecosystem HF Spaces.

    Parameters
    ----------
    kv_writer : function or object with .put(key, value, expiration_ttl) — your
                existing KV wrapper. Must handle both JSON (for `prior:*`) and
                float-array bytes (for `vec:*`).
    encoder_client : object with .embed(text) → List[float] of dim 384.

    Returns
    -------
    dict with count, updated_slugs, errors.
    """
    spaces = _parse_spaces_env()
    LOG.info("ingest_agent_md: %d spaces configured", len(spaces))

    updated = []
    errors = []
    now = int(time.time())

    with httpx.Client(headers={"user-agent": "chainstate-priors/0.7.1"}) as client:
        for owner, repo, path in spaces:
            slug = _slug(owner, repo)
            try:
                text = _fetch_one(client, owner, repo, path)
                if text is None:
                    errors.append({"space": f"{owner}/{repo}", "reason": "fetch failed or not present"})
                    continue

                # Embed
                vec = encoder_client.embed(text)
                if not vec or len(vec) != 384:
                    errors.append({"space": f"{owner}/{repo}", "reason": "encoder returned bad vector"})
                    continue

                # Priors payload
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                title = f"{owner}/{repo} · agent.md"
                url = HF_RESOLVE_TEMPLATE.format(owner=owner, repo=repo, path=path)
                payload = {
                    "source": "agent_md",
                    "slug": slug,
                    "title": title,
                    "url": url,
                    "space_url": f"https://{owner.lower()}-{repo.lower().replace('.','-')}.hf.space",
                    "summary_preview": _summary_preview(text),
                    "full_text_len": len(text),
                    "content_hash": content_hash,
                    "ingested_at": now,
                    "ttl_seconds": AGENT_MD_TTL_SECONDS
                }

                # Write to KV
                kv_writer.put(
                    f"prior:agent_md:{slug}",
                    payload,
                    expiration_ttl=AGENT_MD_TTL_SECONDS
                )
                kv_writer.put(
                    f"vec:agent_md:{slug}",
                    vec,
                    expiration_ttl=AGENT_MD_TTL_SECONDS
                )

                updated.append(slug)
                LOG.info("agent_md ingested: %s (%d chars, hash %s)",
                         slug, len(text), content_hash)

            except Exception as e:
                LOG.exception("agent_md ingest error for %s/%s: %s", owner, repo, e)
                errors.append({"space": f"{owner}/{repo}", "reason": str(e)})

    return {
        "source": "agent_md",
        "count": len(updated),
        "updated_slugs": updated,
        "errors": errors,
        "ran_at": now
    }


# =============================================================================
# Standalone smoke test:
#   python -m sources.agent_md
# Requires ENCODER_URL env var pointing at chainstate-encoder.
# =============================================================================
if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    class StubKV:
        def __init__(self): self.store = {}
        def put(self, k, v, expiration_ttl=None):
            self.store[k] = v
            print(f"  KV.put({k}, ttl={expiration_ttl}) → {'dict' if isinstance(v, dict) else 'vec[' + str(len(v)) + ']'}")

    class HttpEncoder:
        def __init__(self, url, token=None):
            self.url = url.rstrip("/")
            self.token = token
        def embed(self, text):
            headers = {"content-type": "application/json"}
            if self.token:
                headers["authorization"] = f"Bearer {self.token}"
            r = httpx.post(f"{self.url}/embed",
                           json={"text": text}, timeout=60, headers=headers)
            r.raise_for_status()
            return r.json().get("vector", [])

    enc_url = os.getenv("ENCODER_URL", "https://chainstate-encoder.onrender.com")
    enc = HttpEncoder(enc_url, os.getenv("ENCODER_TOKEN"))
    kv = StubKV()
    result = ingest_agent_md(kv, enc)
    json.dump(result, sys.stdout, indent=2)
    print()
