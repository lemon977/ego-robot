# Handle Data Cleaning V3 — AI Handoff and Project-Cleanup Notes

Last updated: 2026-09-16 (Asia/Shanghai)

This document is the operational handoff for another AI or maintainer who may
clean up or reorganize `/mnt/workspace/code/chaoyang` while the handle-data V3
publication is running or after it completes. Read
`docs/HANDLE_DATA_CLEANING_V3.md` first for the algorithm, modality contracts,
audit results and output schema.

## Intentional project files — do not treat as residue

The following additive, currently untracked files are intentional deliverables:

- `src/chaoyang/ops/tactile_quality_gate_v1.py`
- `src/chaoyang/ops/convert_handle_egodex_v3.py`
- `src/chaoyang/ops/batch_clean_handle_content_v3.py`
- `src/chaoyang/ops/run_handle_cleaning_v3_queue.py`
- `docs/HANDLE_DATA_CLEANING_V3.md`
- `docs/HANDLE_DATA_CLEANING_V3_HANDOFF.md`

Do not delete, rename, autoformat or move these files while the queue is active.
The running Python processes import or execute them by their current absolute
paths. The older V2 tools were deliberately left unchanged because other work
may depend on them.

## Source and output boundaries

Immutable read-only sources:

- `/mnt/data/egodata/datasets/ego/chips_cards_handle_0911`
- `/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0914`
- `/mnt/data/egodata/datasets/ego/chips_cards_hands__0915`

Authorized publication roots:

- `/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0911`
- `/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_highview_0914`
- `/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915`

Never edit or delete source sessions. Do not place V3 datasets elsewhere. No
GPU is used. Production runs serially under `nice 15` and `ionice idle` so it
does not compete aggressively with training and simulation jobs.

## Queue and status authority

The queue order is fixed:

1. `chips_cards_handle_0911`
2. `chips_cards_handle_highview_0914`
3. `chips_cards_hands__0915`

The durable queue receipt is:

`/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0911/QUEUE_STATE.json`

Per-dataset live state is `<processed-root>/STATE.json`. Per-dataset terminal
state is `<processed-root>/DATASET_RESULT.json`. Logs are stored within the
processed roots as `RUN.log`; the foreground-to-background transition for 0911
is retained as `RUN_INITIAL.log`. The preflight decision for every session is
in `PREFLIGHT_AUDIT.json`.

The work is complete only when all of the following are true:

- `QUEUE_STATE.json.state == "COMMITTED"`;
- every dataset has `DATASET_RESULT.json.state == "COMMITTED"`;
- every dataset result has `failed == 0`;
- `completed == session_count` for each dataset;
- no V3 batch/converter process remains.

Do not infer completion from directory counts alone. Targets are published by
atomic rename, but the dataset-level terminal receipts are authoritative.

## Expected preflight totals

| Dataset | Sessions | Expected cleaned | Expected rejected |
| --- | ---: | ---: | ---: |
| 0911 | 241 | 139 | 102 |
| 0914 | 220 | 202 | 18 |
| 0915 | 220 | 220 | 0 |

For 0914, 17 rejections are tactile-quality failures and `playing_cards/088`
is rejected for `VISUAL_SENSOR_PROJECTION_MISMATCH`. For 0915, MANUS is
`ABSENT_NOT_CAPTURED`; PICO26 is preserved. MANUS absence must not be converted
into a rejection or filled with fabricated data.

## Monitoring without mutation

Use these read-only checks:

```bash
python - <<'PY'
import json
from pathlib import Path

processed = Path('/mnt/data/egodata/datasets/ego/processed')
for name in (
    'chips_cards_handle_0911',
    'chips_cards_handle_highview_0914',
    'chips_cards_hands__0915',
):
    root = processed / name
    for receipt in ('STATE.json', 'DATASET_RESULT.json'):
        path = root / receipt
        if path.is_file():
            value = json.loads(path.read_text())
            print(name, receipt, {key: value.get(key) for key in
                  ('state', 'completed', 'session_count', 'cleaned',
                   'rejected', 'failed')})
PY

pgrep -af '[b]atch_clean_handle_content_v3|[c]onvert_handle_egodex_v3|[r]un_handle_cleaning_v3_queue'
```

Before starting or restarting anything, confirm that no matching live process
and no live target lock already owns the same dataset. Never run a second writer
against an active processed root.

## Failure and resumption rules

The batch is receipt-resumable: already committed cleaned sessions and valid
rejection receipts are validated and adopted; they are not overwritten. A
converter staging directory is removed on failure, and a completed session is
published only by atomic rename.

If `QUEUE_STATE.json` says `FAILED`, inspect its `error`, the current dataset's
`RUN.log`, `STATE.json`, and process list first. Do not delete the output root or
start over. Resume only the failed dataset with the same command and paths
documented in `docs/HANDLE_DATA_CLEANING_V3.md`. The target lock prevents two
live batch owners but is not a substitute for checking the process list.

Server reboot is outside the queue's persistence guarantee. SSH or IDE
disconnection is safe because the batch and queue run in detached sessions with
parent PID 1. After a reboot, inspect receipts and resume rather than deleting
partial work.

## Cleanup rules after terminal success

Temporary test samples, transfer copies and local staging were already removed.
The active converter may use `/tmp/handle-stereo-*` and `/tmp/handle-mono-*`;
these are converter-owned and must not be removed while a conversion process is
live. They are deleted by normal converter completion/failure cleanup.

After all three terminal receipts are committed, retain the six intentional
project files listed above and all processed-root receipts/logs. They are audit
evidence, not residue. Any proposed removal of V3 tools or logs should happen
only after downstream consumers have adopted a stable release and should be a
separate, explicit decision.

## Claim and safety limits

- Tactile admission proves raw integer activity and integrity, not calibrated
  force or contact truth.
- A single inactive tactile side can be task-valid and is a warning only.
- Format compatibility never overrides tactile or visual/tracking rejection.
- Source archives are immutable and are never cleaned by deletion.
- No SSH passwords, private-key material or other credentials belong in project
  documentation or receipts.
