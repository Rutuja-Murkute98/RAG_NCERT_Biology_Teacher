"""Shared retry-with-backoff helper for every Google API call in this project.

Every module here (captioning, embeddings, chat) goes through the same
google-genai SDK, but SEVERAL different things can go wrong on any single call:
  1. A rate limit - google-genai raises `google.genai.errors.ClientError`
     with `.code == 429`. Calling Gemini in a tight loop (one call per page,
     across 13 chapters) reliably hits the per-minute quota, not just under
     heavy load.
  2. Google's own servers being transiently overloaded - `ClientError`/
     `ServerError` with `.code` in the 500s (503 Service Unavailable, seen
     for real in production: a plain student question like "explain DNA"
     hit a 503 with no retry, surfacing as a generic "Something went wrong"
     error even though the SAME question worked moments later). This is
     Google's problem for a few seconds, not a real failure worth crashing
     the whole request over.
  3. A plain network blip during the actual API call - `httpx.TransportError`
     (connection reset/aborted/timed out).
  4. A network blip during OAuth TOKEN REFRESH specifically -
     `google.auth.exceptions.TransportError`. This uses a completely
     different HTTP stack (the `requests`/`urllib3` libraries, not `httpx`),
     so it doesn't get caught by #3's check even though it's the same kind
     of problem - found for real when a long unattended indexing run's
     machine went to sleep and woke up with DNS not yet available, crashing
     the run with an uncaught error type instead of retrying.
All four are worth automatically retrying; none is worth crashing the
whole run over. Written once here instead of duplicated in every caller.

"Exponential backoff" means each retry waits longer than the last (10s,
20s, 40s, ...) - a rate limit is a PER-MINUTE quota window, so retrying at a
fixed short interval risks hitting the same wall repeatedly; waiting longer
each time gives the window real time to clear.
"""

import time

import httpx
from google.auth.exceptions import TransportError as GoogleAuthTransportError
from google.genai.errors import APIError

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable_error(exc: Exception) -> bool:
    # APIError covers both ClientError (4xx) and ServerError (5xx) -- one
    # check for both, since 429 (our fault, too many requests) and 503
    # (Google's fault, temporarily overloaded) are both worth the same
    # retry-with-backoff treatment.
    if isinstance(exc, APIError) and getattr(exc, "code", None) in _RETRYABLE_STATUS_CODES:
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, GoogleAuthTransportError):
        return True
    return False


def call_with_retry(fn, *, max_retries: int = 5, base_delay_seconds: float = 10):
    """Call fn() (a zero-argument callable, e.g. a lambda) and retry with
    exponential backoff if it hits a retryable error (see
    _is_retryable_error). Anything else is NOT retried - it surfaces
    immediately, since retrying a genuine bug (bad input, auth failure, etc.)
    would just waste time repeating a call that was never going to succeed.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if not _is_retryable_error(exc) or attempt == max_retries - 1:
                raise
            wait_seconds = base_delay_seconds * (2**attempt)
            reason = type(exc).__name__
            print(f"    ({reason}, waiting {wait_seconds:.0f}s, retry {attempt + 1}/{max_retries})")
            time.sleep(wait_seconds)


if __name__ == "__main__":
    # Usage: uv run python -m rag_ncert_biology_teacher.retry_utils
    # Smoke test: a function that fails twice with a retryable error, then
    # succeeds - proves the retry loop actually recovers instead of just
    # re-raising immediately.
    attempts = {"count": 0}

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise httpx.ConnectError("simulated network blip")
        return "success"

    result = call_with_retry(flaky_call, base_delay_seconds=0.1)
    print(f"Result: {result!r} after {attempts['count']} attempts")
    assert result == "success" and attempts["count"] == 3
    print("PASS")
