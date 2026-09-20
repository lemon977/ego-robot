# B3 partial-window boundary failure audit

## Decision

`GO_ONE_BOUNDARY_ONLY_SUCCESSOR_ATTEMPT`.

The immutable B3 v1 run remains `FAILED_RUNTIME_FINAL`.  Its failure is a
deterministic partial-window API boundary mismatch, not a Hand/equipment Mask
quality result.  Exactly one v2 attempt is justified after normal CPU preflight
and V7.1 GPU lease acquisition.  It must use a new runner path, config SHA,
output root and receipt; the v1 runner, receipt and failed output history must
not be overwritten.

This decision does **not** pass Mask, Hand identity, equipment identity,
left/right identity, Object, Clean, Contact, training, control, or deployment.

## Exact diagnosis

The frozen first chunk is `[0,64)`, so the B3 runner passed
`max_frame_num_to_track=64`.

The pinned SAM3 code gives that argument two incompatible meanings:

1. `sam3_multiplex_tracking.py:295-297` computes
   `end = start + max_frame_num_to_track` and iterates
   `range(start, end + 1)`.  Starting at 0 with 64 therefore schedules
   **65 processing indices, 0 through 64**.
2. `sam3_multiplex_detector.py:728-742` computes the same value as
   `valid_frame_end`, but uses it as an **exclusive** chunk end.  At frame 64,
   the 16-aligned chunk is `[64,64)`, hence empty.
3. `_batch_find_inputs` then indexes `chunk_find_inputs[0]` at detector line
   140, producing the observed exact `IndexError: list index out of range`.

The failure receipt confirms the sequence: progress reached `64/65`, then the
empty batch failed at `_batch_find_inputs`.  All intended frames 0--63 had
already been processed.

Changing the argument from 64 to 63 is **not** a valid fix.  It merely makes
the detector's exclusive bound 63 while the tracker still schedules frame 63;
the last desired frame becomes the inconsistent boundary.

Full-sequence native calls do not expose this bug.  The known successful
224-frame route passes `max_frame_num_to_track=frame_count`; the tracker clamps
its inclusive end to `num_frames-1`, while the detector receives exclusive end
`num_frames`.  That is why the archived run produced 224/224 frames despite
using the same pinned vendor API.

## Minimal successor

Keep `max_frame_num_to_track=end_exclusive-start` unchanged and consume exactly
the promised half-open frame range.  After receiving `end_exclusive-start`
strictly sequential outputs, close the generator instead of requesting the
poison boundary.  Fail closed if the stream ends early or emits a non-sequential
frame.

The proposed v2 changes only iterator consumption.  It does not change:

- model, checkpoint, official code commit, adapter or GPU lease policy;
- text prompts (`hand`, `wrist-worn device`);
- threshold 0.5, geometry gates, 64-frame windows or candidate caps;
- independent state per role/chunk;
- UNKNOWN, overlap rejection, identity, or consumer semantics.

Prepared artifacts:

- `run_b3_sam31_independent_hand_equipment_canary_v2.py`
- `B3_MASK_SUCCESSOR_CANARY_V2.json`
- `0001_take_exact_half_open_sam_window.patch`
- `test_b3_boundary_successor_v2.py`

The v2 config uses new immutable destinations:

- deep output: `.../packages/B3_MASK_SUCCESSOR_CANARY_V2`
- GPU receipt: `.../packages/B3_MASK_SUCCESSOR_GPU_COMMAND_RECEIPT_V2.json`
- shallow output: `docs/current/visuals/0915_ROBOT_RECOVERY_15H_V2/B3_MASK_SUCCESSOR_V2`

Only one GPU attempt is recommended.  A runtime success would establish that
the frozen diagnostic completed, not that its Masks are correct.

## Cleanup and lease audit

Read-only verification after v1 failure found:

- final `B3_MASK_SUCCESSOR_CANARY_V1` directory: absent;
- matching `.B3_MASK_SUCCESSOR_CANARY_V1.staging-*`: absent;
- V7.1 lease status: `RELEASED`;
- release reason: `WORKER_RC_1`;
- wrapper PID 97219 and B3 worker PID 97222: no longer running;
- only an unrelated external GPU process remained; it was not touched.

Thus the atomic abort and lease release paths behaved correctly.

## Test result

- 5/5 new boundary/integration fixtures passed.
- 13/13 existing B3 preparation regressions passed unchanged.
- v2 config validation, Python compilation and patch dry-run passed.
- No GPU model was loaded and no remote file was modified by this audit.

## Frozen evidence

- v1 failure receipt SHA256:
  `20f5e0da0483d95991d1fb4ba8637157540837ade75e2a57bf2e7553014227dc`
- v1 runner SHA256:
  `8431c8a31ad3365ab59c49092d8e1557d499a67c0f7804d3e38f2793a8d5f6ce`
- pinned tracker SHA256:
  `65e1cea27ee243dad228fb05967a8be92c90ef304665ada76930b899aa034d24`
- pinned detector SHA256:
  `af0feb4f9a3d405103cdc70a90fcba2a9a1234d6fcfe50c563bbfae2c15922f8`
- successful 224-frame native result SHA256:
  `7db3bd40b1aa682a4e37b6a0af373a69fefda8e30a083e9448ebec4ff1df65f5`
