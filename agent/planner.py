from __future__ import annotations

import os
from typing import Any

ProposedCall = dict[str, Any]

class PlannerProtocol:
    def propose(self, record: dict, allowed_specs: list[dict[str, Any]]) -> list[ProposedCall]:
        return []

class OllamaPlanner:
    def __init__(self):
        self.host = os.environ.get("OLLAMA_HOST")
        self.model = os.environ.get("OLLAMA_MODEL", "llama2")

    def propose(self, record: dict, allowed_specs: list[dict[str, Any]]) -> list[ProposedCall]:
        return []

def propose(record, allowed_specs, planner=None):
    if planner is None:
        return []
    if not hasattr(planner, "propose"):
        return []
    return planner.propose(record, allowed_specs)
