"""Template summaries built only from stored investigation observations.

Deterministic, free, and always produced. No network, no LLM, no paid services.
Every number in the summary is taken from a stored observation in the record.

Example structure:

  "Verdict: {verdict}. Tampered score (uncalibrated): {score:.3f} at threshold
   {threshold}. {box sentence or 'No region reported.'} Checks run: {list}.
   Checks not run: {list with reasons}. {Reasons}."

Every number in the summary is taken from a stored observation. Add a test that
parses the summary and checks each number against the record.

Acceptance: summary tests pass; no network.
"""

import re


class SummaryBuilder:
    """Builds a deterministic template summary from an InvestigationRecord."""

    def __init__(self, record):
        self.record = record

    def build(self):
        """Build a template summary from the record."""
        lines = []

        # Verdict
        verdict = self.record.verdict
        if verdict is None:
            verdict_str = "Not flagged by this detector"
        else:
            verdict_str = verdict.value

        lines.append(f"Verdict: {verdict_str}.")

        # Tampered score (uncalibrated)
        score = None
        for obs in self.record.observations:
            if obs.tool == "run_localizer" and obs.result:
                score = obs.result.get("tampered_prob")
                break

        if score is not None:
            lines.append(
                f"Tampered score (uncalibrated): {score:.3f} "
                f"at threshold {self.record.threshold:.3f}."
            )
        else:
            lines.append("Tampered score (uncalibrated): Not available.")

        # Box sentence
        if (
            self.record.verdict
            and self.record.verdict.value == "Flagged for review"
            and score is not None
        ):
            # Find bbox from observations
            bbox = None
            for obs in self.record.observations:
                if obs.tool == "run_localizer" and obs.result and obs.result.get("bbox"):
                    bbox = obs.result["bbox"]
                    break

            if bbox:
                xmin, ymin, xmax, ymax = bbox
                lines.append(f"Region: [{xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}] "
                            f"(normalized coordinates).")
            else:
                lines.append("No region reported.")
        else:
            lines.append("No region reported.")

        # Checks run
        checked = []
        for obs in self.record.observations:
            if obs.tool not in checked:
                checked.append(obs.tool)
        lines.append(f"Checks run: {', '.join(checked)}.")

        # Checks not run
        all_tools = ["inspect_image", "run_localizer", "validate_prediction",
                     "generate_ela", "inspect_metadata", "compare_evidence"]
        not_run = [t for t in all_tools if t not in checked]
        if not_run:
            reasons = []
            for tool in not_run:
                reasons.append(f"{tool} not needed (baseline suffice)")
            lines.append(f"Checks not run: {', '.join(not_run)}. {'. '.join(reasons)}.")

        return " ".join(lines)

    def check_numbers_against_record(self):
        """Parse the summary and check each number against the record.

        Returns True if all numbers match, False otherwise.
        """
        summary = self.build()
        allowed = {round(float(self.record.threshold), 3)}
        for obs in self.record.observations:
            result = obs.result or {}
            if obs.tool == "run_localizer":
                if result.get("tampered_prob") is not None:
                    allowed.add(round(float(result["tampered_prob"]), 3))
                bbox = result.get("bbox")
                if bbox:
                    allowed.update(round(float(value), 3) for value in bbox)
        number_tokens = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", summary)
        numbers = [round(float(value), 3) for value in number_tokens]
        return all(number in allowed for number in numbers)
