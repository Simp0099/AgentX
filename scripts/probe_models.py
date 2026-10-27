"""Probe candidate free models for reliable OpenAI-style tool calling.

Each candidate gets ``TRIALS`` independent one-shot tool-calling requests: one
dummy tool ``get_number(n: integer)`` and the prompt "Call get_number with
n=7". A trial passes only when the response parses to a ``tool_calls`` entry
whose function name is exactly ``get_number`` and whose arguments are valid
JSON containing ``n == 7``.

Only text is sent -- no image ever leaves the machine, so the forensics model
and the dataset are untouched.

Quota care: every HTTP attempt counts against ``--call-cap``; a non-retryable
provider error (401/403/404/model-not-found) abandons the remaining trials for
that candidate instead of burning them; 429 responses back off and retry at
most twice per trial.

Credentials are read by environment variable NAME through ``agent.env`` and are
never printed, logged or written to the results file.

Usage:
    python scripts/probe_models.py
    python scripts/probe_models.py --only gemini_3_5_flash_lite
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.env import describe, get_secret, load_env  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_PATH = ROOT / "configs" / "agent_candidates.yaml"
RESULTS_PATH = ROOT / "results" / "model_probe.json"

TRIALS = 5
DEFAULT_CALL_CAP = 60
REQUEST_TIMEOUT = 20.0
MAX_429_RETRIES = 2
BACKOFF_SECONDS = (3.0, 6.0)

PROMPT = "Call get_number with n=7"

# The dummy tool: a single integer argument, no side effects, no image.
GET_NUMBER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_number",
        "description": "Return the supplied integer.",
        "parameters": {
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "the integer to return"}},
            "required": ["n"],
            "additionalProperties": False,
        },
    },
}


def load_candidates(only: list[str] | None = None) -> list[dict[str, Any]]:
    """Load the candidate list from ``configs/agent_candidates.yaml``."""
    with open(CANDIDATES_PATH, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    candidates = list(data.get("candidates") or [])
    if only:
        wanted = set(only)
        candidates = [c for c in candidates if c.get("name") in wanted]
    return candidates


def _extract_call(message: Any) -> tuple[bool, str]:
    """Return ``(passed, detail)`` for one assistant message.

    A pass requires the exact function name and JSON arguments containing an
    integer ``n`` equal to 7.
    """
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return False, "no tool_calls in response"
    for call in tool_calls:
        function = getattr(call, "function", None)
        if function is None:
            continue
        name = getattr(function, "name", "") or ""
        if name != "get_number":
            return False, f"wrong function name: {name!r}"
        raw_arguments = getattr(function, "arguments", "") or ""
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError):
            return False, "arguments are not valid JSON"
        if not isinstance(arguments, dict):
            return False, "arguments are not a JSON object"
        if arguments.get("n") != 7 or isinstance(arguments.get("n"), bool):
            return False, f"wrong n: {arguments.get('n')!r}"
        return True, "tool_calls[0] get_number(n=7)"
    return False, "tool_calls entries had no function"


def _classify_status(status: int) -> str:
    """Map an HTTP status to a coarse, reportable category."""
    if status == 429:
        return "rate_limited"
    if status in (401, 403):
        return "auth_error"
    if status == 404:
        return "not_found"
    if status == 400:
        return "bad_request"
    if status >= 500:
        return "server_error"
    return "http_error"


def probe_candidate(
    candidate: dict[str, Any],
    call_budget: list[int],
) -> dict[str, Any]:
    """Run the tool-calling trials for one candidate.

    ``call_budget`` is a one-element list holding the remaining global call
    allowance, so every HTTP attempt in every candidate draws it down.
    """
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            OpenAI,
            RateLimitError,
        )
    except ImportError as exc:
        raise SystemExit(
            "the 'openai' package is required: pip install 'openai>=1.40'"
        ) from exc

    result: dict[str, Any] = {
        "name": candidate["name"],
        "provider": candidate.get("provider", ""),
        "model": candidate.get("model", ""),
        "base_url": candidate.get("base_url", ""),
        "api_key_env": candidate.get("api_key_env", ""),
        "key_status": describe(candidate.get("api_key_env", "")),
        "passes": 0,
        "trials_attempted": 0,
        "trials_planned": TRIALS,
        "latency_ms": [],
        "http_errors": [],
        "rate_limit_messages": [],
        "status": "not_run",
        "detail": "",
        "upstream_verified": bool(candidate.get("upstream_verified", False)),
        "notes": candidate.get("notes", ""),
    }

    api_key = get_secret(candidate.get("api_key_env", ""))
    if api_key is None:
        result["status"] = "missing key"
        result["detail"] = f"{candidate.get('api_key_env')} is not set"
        return result

    client = OpenAI(
        api_key=api_key,
        base_url=candidate["base_url"],
        timeout=REQUEST_TIMEOUT,
        max_retries=0,  # retries are handled here so they are counted and capped
    )

    latencies: list[float] = []
    aborted = False

    for _trial in range(TRIALS):
        if aborted:
            break
        if call_budget[0] <= 0:
            result["detail"] = "global call cap reached"
            break

        attempt = 0
        while True:
            if call_budget[0] <= 0:
                break
            call_budget[0] -= 1
            started = time.monotonic()
            try:
                response = client.chat.completions.create(
                    model=candidate["model"],
                    messages=[{"role": "user", "content": PROMPT}],
                    tools=[GET_NUMBER_TOOL],
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=128,
                )
                latencies.append((time.monotonic() - started) * 1000)
                result["trials_attempted"] += 1
                passed, detail = _extract_call(response.choices[0].message)
                if passed:
                    result["passes"] += 1
                result["detail"] = detail
                break

            except RateLimitError as exc:
                message = str(exc)[:200]
                result["rate_limit_messages"].append(message)
                result["http_errors"].append("429 rate_limited")
                if attempt < MAX_429_RETRIES and call_budget[0] > 0:
                    time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    attempt += 1
                    continue
                # Quota exhausted: stop this candidate rather than spend more.
                result["detail"] = "rate limited (429) after retries"
                aborted = True
                break

            except APIStatusError as exc:
                category = _classify_status(exc.status_code)
                result["http_errors"].append(f"{exc.status_code} {category}")
                result["detail"] = f"{exc.status_code} {category}: {str(exc)[:160]}"
                if category in ("auth_error", "not_found", "bad_request"):
                    # Non-retryable: do not spend the remaining trials.
                    aborted = True
                break

            except (APIConnectionError, APITimeoutError) as exc:
                result["http_errors"].append(f"connection_error {type(exc).__name__}")
                result["detail"] = f"unreachable: {type(exc).__name__}"
                aborted = True
                break

            except Exception as exc:  # unexpected provider/client failure
                result["http_errors"].append(f"error {type(exc).__name__}")
                result["detail"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                aborted = True
                break

    result["latency_ms"] = [round(value, 1) for value in latencies]
    if latencies:
        result["median_latency_ms"] = round(statistics.median(latencies), 1)
    else:
        result["median_latency_ms"] = None

    result["status"] = "unreachable" if result["trials_attempted"] == 0 else "ok"
    return result


def verdict_for(result: dict[str, Any]) -> str:
    """Human-readable outcome for the results table."""
    if result["status"] == "missing key":
        return "missing key"
    if result["status"] == "unreachable":
        return "unreachable"
    if result["passes"] == 0:
        return "failed tool-calling"
    if result["passes"] == TRIALS:
        return "passed tool-calling"
    return f"partial tool-calling ({result['passes']}/{TRIALS})"


def select(results: list[dict[str, Any]]) -> tuple[dict | None, list[dict]]:
    """Pick the primary and fallback by pass count, then median latency."""
    ranked = [
        r
        for r in results
        if r["passes"] > 0 and r.get("median_latency_ms") is not None
    ]
    ranked.sort(key=lambda r: (-r["passes"], r["median_latency_ms"]))
    primary = ranked[0] if ranked else None
    fallbacks = ranked[1:]
    return primary, fallbacks


def markdown_table(results: list[dict[str, Any]]) -> str:
    """Render the candidate table for ``docs/model_selection.md``."""
    header = (
        "| # | candidate | provider | model | base URL | key var | key | "
        "pass/5 | median latency | HTTP errors | rate-limit msgs | outcome |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for index, item in enumerate(results, start=1):
        latency = item.get("median_latency_ms")
        latency_text = f"{latency:.0f} ms" if latency is not None else "n/a"
        errors = ", ".join(item["http_errors"]) or "none"
        limits = str(len(item["rate_limit_messages"])) if item["rate_limit_messages"] else "0"
        rows.append(
            f"| {index} | `{item['name']}` | {item['provider']} | `{item['model']}` | "
            f"`{item['base_url']}` | `{item['api_key_env']}` | {item['key_status']} | "
            f"{item['passes']}/{TRIALS} | {latency_text} | {errors} | {limits} | "
            f"{verdict_for(item)} |"
        )
    return header + "\n".join(rows) + "\n"


def main() -> int:
    """Probe every candidate and write the machine-readable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="probe only these candidate names",
    )
    parser.add_argument(
        "--call-cap",
        type=int,
        default=DEFAULT_CALL_CAP,
        help=f"maximum HTTP attempts across all candidates (default {DEFAULT_CALL_CAP})",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=RESULTS_PATH,
        help="where to write raw results",
    )
    args = parser.parse_args()

    loaded = load_env()
    print(f"env files loaded: {len(loaded)}")
    for path in loaded:
        print(f"  - {path}")

    candidates = load_candidates(args.only)
    if not candidates:
        raise SystemExit("no candidates selected")

    budget = [args.call_cap]
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        print(f"probing {candidate['name']} ({candidate['model']}) ...", flush=True)
        outcome = probe_candidate(candidate, budget)
        results.append(outcome)
        print(
            f"  -> {verdict_for(outcome)} | calls left {budget[0]} | "
            f"median {outcome.get('median_latency_ms')} ms"
        )

    primary, fallbacks = select(results)
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trials_per_candidate": TRIALS,
        "call_cap": args.call_cap,
        "calls_used": args.call_cap - budget[0],
        "results": results,
        "primary": primary["name"] if primary else None,
        "fallbacks": [f["name"] for f in fallbacks],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print()
    print(markdown_table(results))
    print(f"primary : {payload['primary']}")
    print(f"fallback: {payload['fallbacks']}")
    print(f"calls used: {payload['calls_used']} / {args.call_cap}")
    print(f"raw results: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
