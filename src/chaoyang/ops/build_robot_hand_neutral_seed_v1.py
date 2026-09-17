#!/usr/bin/env python3
"""Create a same-session, URDF-neutral hand seed with no foreign trajectory."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso  # noqa: E402
from chaoyang.ops import run_robot_hand_fullsession_v2 as hand  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--hawor-result', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError('fresh neutral seed required')
    hawor = json.loads(args.hawor_result.read_text(encoding='utf-8'))
    if hawor.get('session_id') != args.session_id:
        raise RuntimeError('same-session HaWoR result required')
    assets = hand.old.load_pinned_robot_assets(ROOT)
    contracts = hand.handfit.model_contract(hand.old.official, hand.shared.wrist_adapter, assets)
    neutral = np.stack([np.asarray(contract['neutral'], np.float64) for contract in contracts])
    if neutral.shape != (2, 22) or not np.isfinite(neutral).all():
        raise RuntimeError('hand neutral q invalid')
    for side, contract in enumerate(contracts):
        if (neutral[side] < contract['lower'] - 1e-9).any() or (
            neutral[side] > contract['upper'] + 1e-9).any():
            raise RuntimeError(f'hand neutral side {side} violates URDF limits')
    args.output_root.mkdir(parents=True)
    path = args.output_root / 'TARGET_SESSION_URDF_NEUTRAL_HAND_SEED.npz'
    np.savez_compressed(path, q_hand=neutral[None])
    result = {
        'schema_version': 'rc1-target-session-hand-neutral-seed-v1', 'created_at': now_iso(),
        'status': 'PASSED_DEVELOPMENT_SEED', 'session_id': args.session_id,
        'inputs': {'hawor_result': artifact_ref(args.hawor_result),
                   'robot_asset_pin': artifact_ref(ROOT / 'assets/robot/ROBOT_ASSET_PIN.json'),
                   'code': artifact_ref(Path(__file__))},
        'output': artifact_ref(path),
        'seed_source': 'URDF_MODEL_CONTRACT_NEUTRAL_NOT_OTHER_SESSION',
        'training_eligible': False, 'control_ground_truth': False,
        'claim_limit': 'Numerical hand optimizer seed only; not measured mount, action or physical authority.',
    }
    atomic_json(args.output_root / 'RESULT.json', result)
    print(json.dumps({'status': result['status'], 'output': str(path)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
