# App wiring map

Entry point named by the Dockerfile and README: `app/app.py`. Search found one Gradio app construction (`gr.Blocks`) and one `demo.launch()` in that file. The orchestrator module is imported by the app but not yet invoked by `analyze()`; therefore the app execution path is not yet agent-orchestrated.

| Module | Imported by app.py? | Classification |
|---|---:|---|
| `agent/batch.py` | No | LIBRARY ONLY |
| `agent/config.py` | Yes, transitively through orchestrator | Imported but unused by app inference |
| `agent/orchestrator.py` | Yes, module import | Wired import; not called |
| `agent/planner.py` | No | LIBRARY ONLY |
| `agent/policy.py` | Yes, transitively through orchestrator | Imported but unused by app inference |
| `agent/rendering.py` | Yes | WIRED |
| `agent/reporting.py` | No | LIBRARY ONLY |
| `agent/schemas.py` | Yes | WIRED |
| `agent/summary.py` | No | LIBRARY ONLY |
| `agent/tool_registry.py` | Yes, transitively through orchestrator | Imported but unused by app inference |
| `agent/tools/common.py` | Yes, transitively through registry | Imported but unused by app inference |
| `agent/tools/compare_evidence.py` | Yes, transitively through registry | Imported but unused by app inference |
| `agent/tools/generate_ela.py` | Yes, transitively through registry | Imported but unused by app inference |
| `agent/tools/inspect_image.py` | Yes, transitively through registry | Imported but unused by app inference |
| `agent/tools/inspect_metadata.py` | Yes, transitively through registry | Imported but unused by app inference |
| `agent/tools/run_localizer.py` | Yes, transitively through registry | Imported but unused by app inference |
| `agent/tools/validate_prediction.py` | Yes, transitively through registry | Imported but unused by app inference |
| `agent/verdict.py` | Yes | WIRED |
| `scripts/download_data.py` | No | LIBRARY/CLI ONLY |
| `scripts/evaluate_agent.py` | No | LIBRARY/CLI ONLY |
| `scripts/make_plan_pdf.py` | No | LIBRARY/CLI ONLY |
