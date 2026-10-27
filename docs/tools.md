# Agent tool API

`agent/agent_tools.py` is the interface the language model sees. Every tool takes
only simple JSON types and returns a plain dict, so `json.dumps` always succeeds.
State the model must not control — the detector, the upload directory, and results
already produced in the same investigation — travels in a `ToolContext` injected by
trusted code:

```python
from agent.agent_tools import ToolContext, run_tool

ctx = ToolContext(detector=detector, upload_root="artifacts/uploads", threshold=0.5)
run_tool("inspect_image", {"image_path": "/abs/path/photo.png"}, ctx)
run_tool("run_localizer", {"image_path": "/abs/path/photo.png"}, ctx)
run_tool("validate_prediction", {"image_path": "/abs/path/photo.png"}, ctx)
```

`run_tool` **never raises**. Success returns the tool's payload; failure returns
`{"error": "...", "hint": "..."}`. An unknown name returns
`{"error": ..., "available": [...sorted tool names...]}`.

Paths must resolve inside `ctx.upload_root` when it is set; anything outside is
refused with an error dict.

## What this detector can and cannot do

The localizer is **ELA-based**, trained on **CASIA v2 splices**, and emits an
**uncalibrated score, not a probability**. It cannot detect fully AI-generated
images, and it is unreliable on heavily re-compressed screenshots. The tool
descriptions state these limits so the model does not over-claim them.

---

## `inspect_image`

**Purpose** — Validate an image and return its digest, dimensions and format. Run
this first: every later tool needs a readable image, and the SHA-256 identifies
the file in the record. It says nothing about tampering and does not modify the
file.

**Parameters**

| name | type | units | required |
|---|---|---|---|
| `image_path` | string | absolute path inside `ctx.upload_root` | yes |

**Success output**

```json
{
  "valid": true, "issues": [], "path": "...", "format": "PNG",
  "width": 16, "height": 16, "mode": "RGB", "byte_size": 812,
  "sha256": "…64 hex characters…"
}
```

**Error output** — `{"error": "image file not found: …", "hint": "check the path exists …"}`;
also for a path outside the upload root, an unreadable file, or one over the
configured byte/pixel limit.

**Example**

```python
run_tool("inspect_image", {"image_path": "/abs/uploads/photo.png"}, ctx)
```

---

## `run_localizer`

**Purpose** — Run the ELA-based tampering localizer and return its score and
region. Use it when you need to know whether the image looks spliced. Follow it
with `validate_prediction` before trusting the region.

**Parameters**

| name | type | units | required |
|---|---|---|---|
| `image_path` | string | absolute path inside `ctx.upload_root` | yes |
| `threshold` | number | 0–1, score at or above which the label is `tampered` | no (default: `ctx.threshold`) |

**Success output**

```json
{
  "tampered_prob": 0.4321, "label": "not_tampered",
  "bbox": [0.1, 0.12, 0.88, 0.9], "image_size": [16, 16],
  "model_version": "…", "threshold": 0.5
}
```

`tampered_prob` is **uncalibrated** and rounded to 4 decimals. `bbox` is
normalized `[xmin, ymin, xmax, ymax]`, or `null` when the model produced none.

**Error output** — a missing checkpoint (`"confirm checkpoints/ contains the
trained model"`), an unreadable image, a threshold outside 0–1, or a timeout.

**Example**

```python
run_tool("run_localizer", {"image_path": "/abs/uploads/photo.png", "threshold": 0.5}, ctx)
```

---

## `validate_prediction`

**Purpose** — Check the localizer result **already recorded** for this image: is
the score finite and in range, and is the box ordered, in range and plausibly
sized? Use it before presenting a region. It does not re-read the image and cannot
repair a missing or malformed box; `valid: false` means the region must not be
shown as the tampered area.

**Parameters**

| name | type | units | required |
|---|---|---|---|
| `image_path` | string | absolute path inside `ctx.upload_root` | yes |

The localizer score and box are read from `ctx`, never passed by the model.

**Success output**

```json
{"valid": true, "issues": []}
```

Issues are short strings such as `"box coordinates out of range or unordered"` or
`"box coverage above maximum"`.

**Error output** — `{"error": "no localizer result to validate", "hint": "call run_localizer for this image first"}`.

**Example**

```python
run_tool("validate_prediction", {"image_path": "/abs/uploads/photo.png"}, ctx)
```

---

## `generate_ela`

**Purpose** — Write an Error Level Analysis map next to the investigation and
return its path. Use it when the score is borderline or the box looks wrong and
you want supporting visual evidence. It cannot decide tampering: ELA also reacts
to honest re-saving and re-compression.

**Parameters**

| name | type | units | required |
|---|---|---|---|
| `image_path` | string | absolute path inside `ctx.upload_root` | yes |
| `quality` | integer | 1–100, JPEG quality for the ELA round trip | no (default: config `ela_quality`) |

**Success output**

```json
{"artifact": "artifacts/ela/<uuid>.png", "width": 16, "height": 16, "quality": 90}
```

**Error output** — an unwritable artifact directory, an unreadable image, or
`quality` out of range.

**Example**

```python
run_tool("generate_ela", {"image_path": "/abs/uploads/photo.png", "quality": 90}, ctx)
```

---

## `inspect_metadata`

**Purpose** — Read EXIF/metadata and report which configured fields are present.
Use it when the goal mentions origin, camera, editing history or source, since a
stripped or contradictory EXIF block is weak corroboration. Most images
legitimately have no EXIF: **absence alone is not evidence**, and deleted
metadata cannot be recovered.

**Parameters**

| name | type | units | required |
|---|---|---|---|
| `image_path` | string | absolute path inside `ctx.upload_root` | yes |

**Success output**

```json
{
  "format": "PNG", "width": 16, "height": 16, "mode": "RGB",
  "exif": {}, "present_fields": {}, "absent_fields": ["Make", "Model"],
  "parse_errors": [], "exif_status": "absent"
}
```

`exif_status` is one of `present`, `absent` or `parse_error`.

**Error output** — an unreadable image or a path outside the upload root.

**Example**

```python
run_tool("inspect_metadata", {"image_path": "/abs/uploads/photo.png"}, ctx)
```

---

## `compare_evidence`

**Purpose** — Compare the score, validation, ELA and metadata **already recorded**
in this investigation and list agreements and conflicts. Use it last, once at
least two other tools have run. It reads only recorded evidence and cannot gather
anything new; an empty conflict list means the recorded signals agree, not that
the image is authentic.

**Parameters** — none. The evidence comes from `ctx.results`.

**Success output**

```json
{
  "agreements": ["high score and valid localization"],
  "conflicts": ["high score with invalid localization"],
  "metadata_fields": ["exif", "format"]
}
```

**Error output** — `{"error": "no evidence recorded yet", "hint": "run run_localizer and validate_prediction first"}`.

**Example**

```python
run_tool("compare_evidence", {}, ctx)
```

---

## Schemas

`TOOL_SCHEMAS` holds the same six tools in OpenAI `tools` format:

```json
{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
```

`agent.orchestrator.build_tool_schemas()` returns this list. `tests/test_agent_tools_api.py`
checks every `parameters` block against the JSON Schema meta-schema, keeps `TOOLS`
and `TOOL_SCHEMAS` in sync, and asserts `json.dumps` succeeds for every tool.
