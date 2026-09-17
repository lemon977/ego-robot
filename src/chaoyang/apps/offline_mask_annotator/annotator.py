#!/usr/bin/env python3
"""Command-line entry point for the portable offline mask annotator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from mask_annotator.config import ConfigError, load_config
from mask_annotator.project import ProjectError, export_project, prepare_project
from mask_annotator.selftest import run_self_test
from mask_annotator.server import serve
from mask_annotator.validation import validate_project


ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Offline 9-class video mask annotator")
    value.add_argument("command", choices=("prepare", "serve", "run", "export", "validate", "self-test"))
    value.add_argument("--config", default=str(ROOT / "config.json"), help="path to config JSON")
    value.add_argument("--self-test-work-dir", default="", help="optional persistent directory for synthetic self-test")
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "self-test":
            result = run_self_test(Path(args.self_test_work_dir).resolve() if args.self_test_work_dir else None)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "PASS" else 1
        config = load_config(args.config, require_input=args.command in {"prepare", "run"})
        if args.command == "prepare":
            result = prepare_project(config)
            print(json.dumps({"status": "PREPARED", "output_dir": str(config.output_dir), "frames": len(result["frames"])}, ensure_ascii=False, indent=2))
        elif args.command == "run":
            prepare_project(config)
            serve(config)
        elif args.command == "serve":
            serve(config)
        elif args.command == "export":
            result = export_project(config)
            print(json.dumps({"status": result["status"], "output_dir": str(config.output_dir / "export"), "frames": result["frame_count"]}, ensure_ascii=False, indent=2))
        elif args.command == "validate":
            result = validate_project(config)
            print(json.dumps({"status": result["status"], "errors": result["errors"], "warnings": result["warnings"]}, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "PASS" else 1
        return 0
    except (ConfigError, ProjectError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
