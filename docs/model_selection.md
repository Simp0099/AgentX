# Model selection for the agent's reasoning engine (Phase 1)

This document records which free model the tool-using agent will drive, chosen by
measurement rather than by reputation. It covers every candidate carried over from the
companion desktop-assistant project's verified model registry plus the Gemini entry named in
the conversion checklist.

Re-run everything with:

```bash
export AGENT_ENV_FILE=/path/to/other/project/.env   # name only; never commit it
python scripts/probe_models.py
```

Raw results, including the exact HTTP error text, are in `results/model_probe.json`.
Probe date: **2026-10-27**. Calls used: **27 of a 60-call cap**.

## What was measured

Each candidate received five independent one-shot requests. Every request exposed exactly one
dummy tool and one instruction:

- tool: `get_number(n: integer)` with `additionalProperties: false`
- prompt: `Call get_number with n=7`
- request: `temperature=0`, `max_tokens=128`, `tool_choice="auto"`

A trial **passes** only when the response parses to a `tool_calls` entry whose function name is
exactly `get_number` and whose `arguments` are valid JSON containing the integer `n == 7`.
Prose answers, mangled argument JSON, wrong function names and empty `tool_calls` all count as
failures. Only text is sent; no image, no checkpoint, no dataset row leaves the machine.

Quota care: each HTTP attempt draws down a global cap; a non-retryable provider error
(401/403/404/unavailable) abandons that candidate's remaining trials instead of spending them;
a 429 backs off (3 s, then 6 s) and retries at most twice per trial before the candidate is
parked.

## Results

| # | candidate | provider | model | base URL | key var | key | pass/5 | median latency | HTTP errors | rate-limit msgs | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `gemini_2_5_flash` | gemini | `gemini-2.5-flash` | `https://generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY` | set (length 53) | 0/5 | n/a | 404 not_found | 0 | unreachable |
| 2 | `gemini_3_6_flash` | gemini | `gemini-3.6-flash` | `https://generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY` | set (length 53) | 5/5 | 2026 ms | none | 0 | passed tool-calling |
| 3 | `gemini_3_5_flash_lite` | gemini | `gemini-3.5-flash-lite` | `https://generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY` | set (length 53) | 5/5 | 6792 ms | none | 0 | passed tool-calling |
| 4 | `nemotron_3_5_lightning` | openrouter | `nvidia/nemotron-3.5-lightning:free` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | set (length 73) | 5/5 | 856 ms | none | 0 | passed tool-calling |
| 5 | `nemotron_3_ultra` | openrouter | `nvidia/nemotron-3-ultra-550b-a55b:free` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | set (length 73) | 5/5 | 1522 ms | none | 0 | passed tool-calling |
| 6 | `nemotron_3_super` | openrouter | `nvidia/nemotron-3-super-120b-a12b:free` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | set (length 73) | 0/5 | n/a | 404 not_found | 0 | unreachable |
| 7 | `inclusionai_ling_3_flash` | openrouter | `inclusionai/ling-3.0-flash-sante:free` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | set (length 73) | 0/5 | n/a | 404 not_found | 0 | unreachable |
| 8 | `space_bunny_free` | opencode_zen | `space-bunny-free` | `https://opencode.ai/zen/v1` | `OPENCODE_ZEN_API_KEY` | set (length 51) | 0/5 | n/a | 400 bad_request | 0 | unreachable |
| 9 | `big_pickle` | opencode_zen | `big-pickle` | `https://opencode.ai/zen/v1` | `OPENCODE_ZEN_API_KEY` | set (length 51) | 0/5 | n/a | 429 rate_limited ×3 | 3 | unreachable |

`key` reports presence and length only. No credential value appears in this file, the probe
output, or any commit.

## Why the five failures failed

| candidate | provider message (verbatim, trimmed) | reading |
|---|---|---|
| `gemini_2_5_flash` | `404 ... Error code: 404` | The model id named in the checklist does not exist on this key. The PDF's example is stale; `gemini-3.6-flash` is the working id. |
| `nemotron_3_super` | `404 ... This model is unavailable for free. The paid version is available now - use this slug instead: nvidia/nemotron-3-super-120b-a12b` | The `:free` slug is retired. Not adopted: a paid slug is not a free candidate. |
| `inclusionai_ling_3_flash` | `404 ... is no longer available on OpenRouter: no provider serves this model anymore` | Retired upstream. This matched the companion project's own `enabled: false` note. |
| `space_bunny_free` | `400 ... Upstream request failed: Model is unavailable.` | Downstream is dead even though the id is still catalogued. |
| `big_pickle` | `429 ... Free usage limit reached. Free models work in OpenCode (https://opencode.ai)` | Confirms the companion project's finding: the Zen free tier is restricted to the OpenCode client. It surfaces as 429 rather than the 403 that project recorded, so it is parked rather than retried. |

## Selection

Rule applied, per the checklist: **highest tool-call success rate wins; ties are broken by lower
median latency.** Four candidates tied at 5/5, so latency decided it.

| rank | candidate | model | pass/5 | median latency | role |
|---|---|---|---|---|---|
| 1 | `nemotron_3_5_lightning` | `nvidia/nemotron-3.5-lightning:free` | 5/5 | 856 ms | **PRIMARY** |
| 2 | `nemotron_3_ultra` | `nvidia/nemotron-3-ultra-550b-a55b:free` | 5/5 | 1522 ms | **FALLBACK 1** |
| 3 | `gemini_3_6_flash` | `gemini-3.6-flash` | 5/5 | 2026 ms | FALLBACK 2 |
| 4 | `gemini_3_5_flash_lite` | `gemini-3.5-flash-lite` | 5/5 | 6792 ms | FALLBACK 3 |

**PRIMARY: `nemotron_3_5_lightning` (`nvidia/nemotron-3.5-lightning:free`, OpenRouter, 856 ms)**
**FALLBACK: `nemotron_3_ultra` (`nvidia/nemotron-3-ultra-550b-a55b:free`, OpenRouter, 1522 ms)**

### Caveats worth knowing before Phase 2

- All four survivors are equally reliable at the tool call itself; they are separated only by
  latency measured on a single day and a single prompt. One bad trial would reorder ranks 2–3.
- `gemini_3_6_flash` is the most capable survivor (the companion project records reasoning,
  vision and structured output as verified) and is the only survivor on a different provider
  from the primary. That makes it the most useful *diversity* fallback even though it ranks
  third. It is deliberately not the primary because the checklist's rule is mechanical and it
  ranked third on it.
- Both Nemotron survivors are OpenRouter `:free` models on the same key. If that provider is
  rate-limited, both fail together; the Gemini entries are the hedge.
- `configs/agent.yaml` is intentionally untouched. Phase 2 writes the selection into it.

## Reproducing

```bash
python scripts/probe_models.py                      # all candidates, 60-call cap
python scripts/probe_models.py --only big_pickle    # one candidate
python scripts/probe_models.py --call-cap 10        # tighter quota
```

The candidate list lives in `configs/agent_candidates.yaml`. Credentials are resolved by
`agent/env.py` from the file named by `AGENT_ENV_FILE` and then the repo-local `.env`, never
overriding a value already exported in the shell.

## Tool-message compatibility (Phase 6)

The loop in `agent/loop.py` sends tool results back with **only** `role`, `tool_call_id` and
`content`. The optional `name` field on a `tool` message, and any extra assistant fields, are
left out deliberately: some OpenAI-compatible endpoints reject messages that carry fields they
do not model, and the three-field form is the intersection every provider in the table above
accepts.

This was chosen **preventatively**, not in response to an observed rejection — Phase 6 ran
entirely offline against a fake client, so no live multi-turn exchange has been recorded here.
If a real provider later rejects a message shape, record the exact error in this section rather
than widening the payload.
