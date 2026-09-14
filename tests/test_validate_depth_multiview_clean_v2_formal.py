from __future__ import annotations

import json
from pathlib import Path
import sys

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import validate_depth_multiview_clean_v2_formal as validator


def test_v2_schema_is_valid_and_keeps_all_fallbacks_disabled():
    schema = json.loads(validator.SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    method = schema["properties"]["method_contract"]["const"]
    assert method["direct_depth_resize"] is False
    assert method["nearest_canvas_extrapolation"] is False
    assert method["single_plane_homography_fill"] is False
    assert method["generative_or_inpaint_fallback"] is False
    assert method["unsupported_policy"] == "KEEP_RAW_AND_FAIL_COVERAGE_GATE"


def test_full_admission_requires_same_immutable_upstreams():
    source = Path(validator.validate_full_admission.__code__.co_filename).read_text(encoding="utf-8")
    assert 'canary.get("inputs") != manifest.get("inputs")' in source
    assert 'canary.get("method_contract") != manifest.get("method_contract")' in source
