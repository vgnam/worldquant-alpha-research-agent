"""Shared OpenAI-compatible chat-completions transport with retry/backoff.

Both the planner and the refinement proposer call the same endpoint. A
transient 429/5xx (e.g. a provider's "brief maintenance" 503) used to fail the
call on the first attempt and silently drop to the deterministic fallback.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Dict, Optional

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
#: Some gateways (Cloudflare error 1010) reject the default ``Python-urllib``
#: agent outright, so identify the client explicitly.
USER_AGENT = "alpha-research-agent/1.0"
#: OpenCode Go rejects requests without a stable per-conversation session id
#: (used for routing and prompt caching); one id per agent process.
SESSION_ID = uuid.uuid4().hex


def _extra_headers(base_url: str) -> Dict[str, str]:
    if "opencode.ai" in base_url:
        return {"x-opencode-session": SESSION_ID}
    return {}


class LLMRequestError(RuntimeError):
    """Final failure after retries; ``status`` is the HTTP code when known."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def warn(message: str) -> None:
    """Visible, unbuffered warning; stdout is reserved for the JSON report."""
    print(f"[alpha-agent] {message}", file=sys.stderr, flush=True)


def post_chat_completion(
    *,
    base_url: str,
    body: Dict[str, Any],
    api_key: str,
    timeout: float,
    retries: int = 3,
    backoff: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    if reasoning_effort:
        body = {**body, "reasoning_effort": reasoning_effort}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    attempts = max(1, int(retries) + 1)
    last: LLMRequestError | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
                **_extra_headers(base_url),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                detail = ""
            last = LLMRequestError(f"HTTP {exc.code}: {detail}", status=exc.code)
            if exc.code not in RETRYABLE_STATUS:
                raise last from exc
        except (urllib.error.URLError, OSError) as exc:
            # OSError covers socket.timeout, which is not TimeoutError before 3.10.
            last = LLMRequestError(f"{type(exc).__name__}: {exc}")
        if attempt + 1 < attempts:
            delay = backoff * (2**attempt)
            warn(f"LLM request failed ({last}); retry {attempt + 1}/{attempts - 1} in {delay:.0f}s")
            sleep(delay)
    assert last is not None
    raise last
