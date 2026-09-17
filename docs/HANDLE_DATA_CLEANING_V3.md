# Handle / Cards / Chips Data Cleaning V3

## Scope

This additive workflow cleans these immutable source datasets:

- `/mnt/data/egodata/datasets/ego/chips_cards_handle_0911`
- `/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0914`
- `/mnt/data/egodata/datasets/ego/chips_cards_hands__0915`

Published datasets live below `/mnt/data/egodata/datasets/ego/processed`. The
workflow does not edit or delete source sessions and does not modify the older
V2 converter or its completed outputs.

## Why V3 exists

The acquisition-aligned V2 batch checked file/HDF5 structure, finite arrays and
timeline validity, but did not gate tactile signal content. It also classified a
successful format conversion as `CLEANED` even when the converter reported
`SESSION_CONTENT_NOT_ADMISSIBLE`. In the ordinary acquisition-HDF5 path the
converter did not load the tactile arrays, so a structurally successful output
could omit tactile payloads.

V3 makes content admission fail closed, preserves tactile in every accepted
frame, and makes missing modalities explicit instead of fabricating them.

## Tactile quality policy

The raw `raw/tactile.jsonl` and `raw/tactile.meta.json` are authoritative. V3
validates metadata completion, hashes/byte counts, frame counts, stream identity,
strict sequence/QPC/wall-clock order, signed-int16 wire shape (369 values),
saturation and frozen streams.

Signal admission is session-scoped because these tasks can legitimately be
unimanual:

- at least 24 signal-bearing frames across both sides;
- at least 2 nonzero channels across side-distinct channel sets;
- peak absolute raw value at least 2;
- one all-zero side is a warning, not a rejection;
- both sides with no meaningful activity are rejected;
- a transient capture warning is retained as a warning when the capture is
  complete and `capture_valid=true`;
- missing, malformed, non-monotonic, saturated or frozen streams are rejected.

These checks establish raw integer activity and integrity only. They do not
claim calibrated force or contact truth.

## Modality contracts

### 0911 and 0914

The acquisition `dataset.hdf5` remains the alignment authority. MANUS25,
controllers, camera and tactile are present. V3 materializes both tactile sides
in every accepted `training_data.json` as:

- `wire_values_369`;
- active and valid wire masks;
- physical `finger_grid_5x4x8` plus active/valid masks;
- nearest-host-QPC source index, sequence and offset.

### 0915

MANUS was not captured. This is recorded as `ABSENT_NOT_CAPTURED` and is not a
quality failure. V3 aligns the actual raw VST, PICO head/controllers, PICO26 hand
joints and tactile streams on a 30 Hz host-QPC timeline. It preserves PICO26
with source/world/camera transforms and does not emit `manus25`, derived MANUS
wrist poses, or placeholder MANUS truth.

## Content admission and publication

An accepted acquisition session must have both:

- `PASSED_TACTILE_QUALITY`; and
- `SESSION_CONTENT_ADMISSIBLE`, or for 0915
  `SESSION_CONTENT_ADMISSIBLE_WITH_DECLARED_MANUS_ABSENCE`.

The established visual/tracking gate still rejects projection mismatch and
frozen tracking. Conversion uses a target-local staging directory and publishes
by atomic rename only after video frame counts, contiguous frame directories,
training records, tracking lines, controller lines and the immutable source
snapshot all validate.

Rejected sessions are not deleted. They receive a small `RESULT.json` receipt
below `rejected/<task>/<NNN>` with the exact gate result.

## Preflight audit results

| Dataset | Sessions | Admitted | Rejected | Notes |
| --- | ---: | ---: | ---: | --- |
| 0911 | 241 | 139 | 102 | All rejections are tactile-quality failures |
| 0914 | 220 | 202 | 18 | 17 tactile failures; cards/088 has visual projection mismatch |
| 0915 | 220 | 220 | 0 | MANUS declared absent; PICO26 preserved |

The authoritative per-session details are stored as `PREFLIGHT_AUDIT.json` in
each processed dataset root.

## Output layout

Each dataset root contains:

```text
PREFLIGHT_AUDIT.json
QUALITY_POLICY.json
STATE.json
DATASET_RESULT.json
cleaned/<task>/<canonical-session-id>/...
rejected/<task>/<NNN>/RESULT.json
```

Each cleaned session includes `CONVERSION_RESULT.json`, a clip manifest, a
HumanEgo manifest, mono/stereo videos, tracking/controller/SLAM sidecars and the
per-frame image/training-record tree. The manifests carry `modality_contract`
and `tactile_quality` objects.

## Tools and execution boundary

- `src/chaoyang/ops/tactile_quality_gate_v1.py`: pure raw tactile integrity/content audit.
- `src/chaoyang/ops/convert_handle_egodex_v3.py`: additive V3 converter wrapper.
- `src/chaoyang/ops/batch_clean_handle_content_v3.py`: serial audit/publication runner with
  target lock, resumable receipts and terminal dataset result.

Production conversion runs as one CPU process under `nice -n 15` and
`ionice -c 3`. It does not request a GPU. This deliberately trades elapsed time
for isolation from concurrent training and simulation work.

Example audit:

```bash
nice -n 15 ionice -c 3 \
  /cpfs_infra/user/chenxianchi/miniconda3/envs/egoforce/bin/python \
  src/chaoyang/ops/batch_clean_handle_content_v3.py \
  --mode audit \
  --task-source playing_cards=/absolute/cards/root \
  --task-source potato_chips=/absolute/chips/root \
  --date-tag MMDD --dataset-id DATASET_ID \
  --audit-report /temporary/audit.json
```

Publication uses the same arguments with `--mode convert` and
`--target-root /mnt/data/egodata/datasets/ego/processed/<dataset>`.
