"""One submission frontend with separate raw-auto and V1-compatible scopes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chaoyang submit inpainting-raw-to-robot",
        description="raw-auto: CPU raw admission; v1-compatible: finite adopted-V1 six-stage workflow.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--session", type=Path)
    selection.add_argument("--manifest", type=Path)
    parser.add_argument("--mode", choices=("raw-auto", "v1-compatible"), default="raw-auto",
        help="Explicit parent-V1 seeds/config are allowed only in v1-compatible mode")
    parser.add_argument("--preview", action="store_true", help="Read-only raw-byte audit; register/execute nothing")
    parser.add_argument("--repair-rounds", type=int, choices=(1, 2, 3), default=1,
        help="Optional bounded repair rounds in v1-compatible mode (default: 1)")
    parser.add_argument("--wall-seconds", type=int, default=None,
        help="New finite V1-compatible candidate Task time limit, at most 3600 seconds")
    parser.add_argument('--clean-reference-policy',choices=('exclude-uncertain-equipment',),default=None,
        help='Opt-in candidate: uncertain RGB equipment excluded from PP context, final deletion mask unchanged')
    parser.add_argument('--resume-from',type=Path,default=None,
        help='Explicit sealed-stage manifest for a fresh finite V1-compatible Task; never scan latest directories')
    parser.add_argument('--clean-write-policy',choices=('gradient-boundary','uncertain-boundary-veto'),default=None,
        help='Experimental fixed-E boundary policy; requires explicit context-exclusion candidate, never default adoption')
    args = parser.parse_args(argv)
    if args.repair_rounds != 1 and args.mode != "v1-compatible":
        parser.error("--repair-rounds 2 or 3 requires --mode v1-compatible")
    if args.clean_reference_policy is not None and (args.mode!='v1-compatible'or args.repair_rounds not in(2,3)):
        parser.error('--clean-reference-policy requires v1-compatible and --repair-rounds 2 or 3')
    if args.wall_seconds is not None and (args.mode != "v1-compatible" or not 0 < args.wall_seconds <= 3600):
        parser.error("--wall-seconds requires --mode v1-compatible and a value from 1 to 3600")
    if args.resume_from is not None and args.mode!='v1-compatible':
        parser.error('--resume-from requires --mode v1-compatible; raw-auto forbidden-derived-input contract is unchanged')
    if args.clean_write_policy is not None and (args.mode!='v1-compatible'or args.clean_reference_policy is None):
        parser.error('--clean-write-policy requires v1-compatible and --clean-reference-policy exclude-uncertain-equipment')
    from chaoyang.paths import get_paths
    from chaoyang.inpainting_candidate.entry import build_request
    root = get_paths().repo_root
    request = build_request(project_root=root, session=args.session, manifest=args.manifest, mode=args.mode,
        repair_rounds=args.repair_rounds)
    if args.mode == "v1-compatible":
        from chaoyang.governance.inpainting_candidate_registration import submit_v1_compatible_request
        options = {"wall_seconds": args.wall_seconds} if args.wall_seconds is not None else {}
        if args.clean_reference_policy is not None:options['clean_reference_policy']='EXCLUDE_UNCERTAIN_EQUIPMENT_RGB'
        if args.resume_from is not None:options['resume_manifest_path']=args.resume_from
        if args.clean_write_policy is not None:
            options['clean_write_policy']={'gradient-boundary':'POISSON_WORKING_GRADIENT_RAW_FIXED_BOUNDARY',
                'uncertain-boundary-veto':'POISSON_RGB_UNCERTAIN_BOUNDARY_VETO'}[args.clean_write_policy]
        result = submit_v1_compatible_request(request, repo_root=root, preview=args.preview, **options)
    else:
        from chaoyang.governance.inpainting_candidate_registration import submit_request
        result = submit_request(request, repo_root=root, preview=args.preview)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = ["main"]
