"""Build a draft session context and print review-frame evidence to stdout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from jsonschema import Draft202012Validator

from pipeline.raw_source import StrictRawResolver, read_local_regular_readonly
from pipeline.session_context import ProfilerConfig, build_session_context


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly profile one manifest-admitted RAW session; no visual producer is run."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--manifest-status", action="append", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--execution-mode",
        required=True,
        choices=("G2_CALIBRATION", "FORMAL_PRODUCTION"),
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--task-card", required=True)
    parser.add_argument("--task-card-sha256", required=True)
    parser.add_argument("--calibration-overlay")
    parser.add_argument("--calibration-overlay-sha256")
    parser.add_argument("--schema", required=True)
    parser.add_argument(
        "--product-line",
        required=True,
        choices=("004_CONTACT_GOLD", "EXACT78_R2_VISUAL_DOMAIN"),
    )
    parser.add_argument(
        "--diagnostic-track",
        required=True,
        choices=("HAND_ONLY_DIAGNOSTIC", "HAND_ARM_FINAL"),
    )
    parser.add_argument("--analysis-scale", required=True, type=float)
    parser.add_argument("--forearm-width-ratio", required=True, type=float)
    parser.add_argument("--contact-radius-ratio", required=True, type=float)
    parser.add_argument("--contact-projection-ratio", required=True, type=float)
    parser.add_argument("--count-min", required=True, type=int)
    parser.add_argument("--count-max", required=True, type=int)
    parser.add_argument("--target-count", required=True, type=int)
    parser.add_argument("--min-gap-seconds", required=True, type=float)
    parser.add_argument("--stable-contact-seconds", required=True, type=float)
    parser.add_argument("--pre-contact-seconds", required=True, type=float)
    parser.add_argument("--contact-on-threshold", required=True, type=float)
    parser.add_argument("--object6d-npz")
    parser.add_argument("--object6d-npz-sha256")
    parser.add_argument("--object-geometry")
    parser.add_argument("--object-geometry-sha256")
    parser.add_argument("--output-context")
    parser.add_argument("--emit", choices=("summary", "full"), default="summary")
    return parser


def _atomic_write_json(path: str, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    resolver = StrictRawResolver(
        manifest_path=args.manifest,
        expected_manifest_sha256=args.manifest_sha256,
        raw_root=args.raw_root,
        session_id=args.session_id,
        allowed_manifest_statuses=args.manifest_status,
    )
    config = ProfilerConfig(
        analysis_scale=args.analysis_scale,
        forearm_width_ratio=args.forearm_width_ratio,
        contact_radius_ratio=args.contact_radius_ratio,
        contact_projection_ratio=args.contact_projection_ratio,
        count_min=args.count_min,
        count_max=args.count_max,
        target_count=args.target_count,
        min_gap_seconds=args.min_gap_seconds,
        stable_contact_seconds=args.stable_contact_seconds,
        pre_contact_seconds=args.pre_contact_seconds,
        contact_on_threshold=args.contact_on_threshold,
    )
    result = build_session_context(
        resolver,
        execution_mode=args.execution_mode,
        product_line=args.product_line,
        diagnostic_track=args.diagnostic_track,
        profile_path=args.profile,
        expected_profile_sha256=args.profile_sha256,
        task_card_path=args.task_card,
        expected_task_card_sha256=args.task_card_sha256,
        calibration_overlay_path=args.calibration_overlay,
        expected_calibration_overlay_sha256=args.calibration_overlay_sha256,
        config=config,
        object6d_npz_path=args.object6d_npz,
        object6d_npz_sha256=args.object6d_npz_sha256,
        object_geometry_path=args.object_geometry,
        object_geometry_sha256=args.object_geometry_sha256,
    )
    schema = json.loads(read_local_regular_readonly(args.schema))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(result.context)
    if args.output_context:
        _atomic_write_json(args.output_context, result.context)

    evidence = {
        str(frame): {
            "roles": list(record["roles"]),
            "features": record["features"],
        }
        for frame, record in result.selection.evidence.items()
    }
    output = {
        "status": "DRAFT_CONTEXT_ONLY_NO_VISUAL_PRODUCER",
        "schema_validation": "PASS",
        "session_id": resolver.session_id,
        "suggested_task_card_parameters": {
            "source_manifest": {
                "path": resolver.manifest_path,
                "sha256": resolver.manifest_sha256,
                "admitted_status": resolver.manifest_status,
                "no_fallback": True,
            },
            "profile": {"path": args.profile, "sha256": args.profile_sha256},
            "profiler": result.selection.normalized_parameters
            | {
                "analysis_scale": config.analysis_scale,
                "forearm_width_ratio": config.forearm_width_ratio,
                "contact_radius_ratio": config.contact_radius_ratio,
                "contact_projection_ratio": config.contact_projection_ratio,
            },
        },
        "selected_frames": list(result.selection.frames),
        "covered_strata": list(result.selection.covered_strata),
        "selected_frame_evidence": evidence,
        "execution_allowed": result.context["execution_allowed"],
        "execution_blockers": result.context["execution_blockers"],
    }
    if args.emit == "full":
        output["session_context"] = result.context
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
