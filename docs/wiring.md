# App wiring map

Entry point named by the Dockerfile and README: `app/app.py`. The single Gradio app construction has **Analysis** and **Chat** tabs. Analysis routes through `app.handler.analyze` to `agent.orchestrator.run_investigation`. Chat routes through `app.chat_handler.agent_chat` to `agent.loop.run_agent_full`; when the model is unavailable it invokes `run_investigation` and labels the result `fixed pipeline (no LLM)`. Both paths use the existing tools and localizer.

| Module | Imported by app.py? | Classification |
|---|---:|---|
| `agent/batch.py` | Yes | WIRED for Analysis ZIP processing |
| `agent/chat_handler.py` | Through app wrapper | WIRED for Chat, image state, and ELA fallback |
| `agent/config.py` | Yes, through orchestrator/tools | WIRED for thresholds and forensic settings |
| `agent/loop.py` | Through chat handler | WIRED for model-selected tool loop |
| `agent/orchestrator.py` | Through handlers | WIRED for Analysis and no-LLM fallback |
| `agent/agent_tools.py` | Through Chat handler | WIRED for model-facing tool dispatch |
| `agent/planner.py` | No | LIBRARY ONLY |
| `agent/policy.py` | Yes, through orchestrator | WIRED for deterministic conditional checks |
| `agent/rendering.py` | Yes | WIRED |
| `agent/reporting.py` | No | LIBRARY ONLY |
| `agent/schemas.py` | Yes | WIRED |
| `agent/summary.py` | No | LIBRARY ONLY |
| `agent/tool_registry.py` | Yes, through orchestrator | WIRED for deterministic checks |
| `agent/tools/common.py` | Yes, through tool implementations | WIRED for bounded tool execution |
| `agent/tools/compare_evidence.py` | Yes, through registry/model tools | WIRED |
| `agent/tools/generate_ela.py` | Yes, through registry/model tools | WIRED |
| `agent/tools/inspect_image.py` | Yes, through registry/model tools | WIRED |
| `agent/tools/inspect_metadata.py` | Yes, through registry/model tools | WIRED |
| `agent/tools/run_localizer.py` | Yes, through registry/model tools | WIRED |
| `agent/tools/validate_prediction.py` | Yes, through registry/model tools | WIRED |
| `agent/verdict.py` | Yes | WIRED |
| `scripts/download_data.py` | No | LIBRARY/CLI ONLY |
| `scripts/evaluate_agent.py` | No | LIBRARY/CLI ONLY |
| `scripts/make_plan_pdf.py` | No | LIBRARY/CLI ONLY |
