# HaWoR project-local environment migration

Status: `PASS_LOCAL_ENVIRONMENT_PROMOTED` (2026-09-03).

The V1 migration completed through the frozen POST_FULL59 contract.  The
published environment tree SHA-256 is
`5aafda3194a560f4c0b7f288f137571f1add309037470301652a40e7de7a265e`;
the authoritative lock and rollback contract are
`systems/hawor/local_environment_lock.json` and
`systems/hawor/local_environment_rollback.json`.  The procedure below remains
the recovery/audit runbook and is not permission to overwrite either result.

The completed full59 batch was a historical exception.  A future recovery may
not create links or launch a replay unless its owner again confirms the
terminal evidence and source-retirement gates.  A missing process alone is not
sufficient.

The source, vendored code/model/MANO pin, destination, launcher, and promotion
gates are separate authorities:

- historical environment provenance:
  `third_party/HaWoR/CHA0YANG_ENVIRONMENT_LOCK.txt` (immutable);
- historical vendored runtime pin:
  `third_party/HaWoR/CHA0YANG_RUNTIME_PIN.json` (immutable);
- new environment authority: `systems/hawor/environment_authority.json`;
- destination: `assets/environments/hawor-py310-v1`;
- activation-free launcher: `tools/hawor_python.sh`.

## 1. Fresh terminal-state preflight

From the repository root:

```bash
python tools/no_clobber_hardlink_snapshot.py preflight \
  --source /mnt/workspace/miniconda3/envs/hawor \
  --target assets/environments/hawor-py310-v1
```

The preflight is read-only.  It must have no fatal blocker, no stale staging
tree, the destination must be absent, and both roots must remain on one device.
At preparation time the only blocker was zero reported free blocks, with a
55,306,705-byte conservative metadata estimate and 6,394 new inodes.  Do not
infer usable metadata capacity from abundant free inodes: check the CPFS quota
or control plane.  The risk acknowledgement is permitted only after that
external capacity check; it is not a generic way to bypass `ENOSPC`.

## 2. Atomic no-clobber snapshot

When the fresh preflight is clear:

```bash
python tools/no_clobber_hardlink_snapshot.py create \
  --source /mnt/workspace/miniconda3/envs/hawor \
  --target assets/environments/hawor-py310-v1
```

If CPFS explicitly confirms metadata capacity while `statvfs` still reports
zero, append `--acknowledge-metadata-risk` and preserve that confirmation with
the run evidence.  The flag must not be used based only on this runbook.

The destination is published atomically only after all regular-file hashes,
relative links, directory metadata, source stability, hard-link counts, and
the manifest have passed.  Linux `renameat2(RENAME_NOREPLACE)` is preferred;
on CPFS/FUSE, which rejects that flag, the tool atomically reserves the absent
name with its own empty directory and replaces only that verified reservation.
A handled failure leaves the destination absent and retains a uniquely named staging tree with
`.chaoyang-hardlink-snapshot/failure.json`.

Verify the published payload and record the returned count, logical bytes,
hard-link identity, and tree content SHA:

```bash
python tools/no_clobber_hardlink_snapshot.py verify \
  --snapshot assets/environments/hawor-py310-v1 \
  --content
```

A hard-link tree is not immutable against in-place writes through the old
environment or its Conda cache aliases.  Retirement of that source and the
final content verification are therefore promotion gates.

## 3. Local Python/prefix/import probe

Choose a new immutable evidence directory below `tasks/control/runs/` and run:

```bash
python tools/verify_hawor_environment_migration.py probe \
  --environment assets/environments/hawor-py310-v1 \
  --output tasks/control/runs/<new_env_migration_run>/LOCAL_ENV_PROBE.json
```

The probe hides CUDA, disables user site and bytecode writes, clears inherited
Python/Conda paths, verifies the complete snapshot, and checks
`sys.executable`, `sys.prefix`, `sys.base_prefix`, and the origin of the exact
HaWoR imports.  It then checks payload metadata again so the imports cannot
silently mutate the snapshot.  Direct execution of copied Conda entry-point
scripts is not canonical; use `tools/hawor_python.sh -m <module>` or pass a
project runner to the launcher.

## 4. Three frozen replay points

Acquire a fresh, exact GPU lease only after full59 has released its lease.  Use
new absent stage roots and the frozen context runner.  The twelve frames in
each review sheet are a fixed visualization sample; the numeric authority is
the complete 36 target frames after 16 causal pre-roll frames.

```bash
tools/hawor_python.sh tools/run_hawor_context_canary_v2.py \
  --task-id chips \
  --session-id get_potato_chips_0901_016 \
  --raw-root /mnt/data/egodata/datasets/ego/chips_cards_tracker_0901/potato_chips/get_potato_chips_0901_016 \
  --stage-root tasks/chips/runs/pipeline/<new_env_migration_run>/fit_chips016 \
  --target-start 458 --target-end 494 \
  --lease-holder <exact_new_holder>

tools/hawor_python.sh tools/run_hawor_context_canary_v2.py \
  --task-id chips \
  --session-id get_potato_chips_0901_008 \
  --raw-root /mnt/data/egodata/datasets/ego/chips_cards_tracker_0901/potato_chips/get_potato_chips_0901_008 \
  --stage-root tasks/chips/runs/pipeline/<new_env_migration_run>/eval_chips008 \
  --target-start 308 --target-end 344 \
  --lease-holder <exact_new_holder>

tools/hawor_python.sh tools/run_hawor_context_canary_v2.py \
  --task-id poker \
  --session-id play_cards_0901_023 \
  --raw-root /mnt/data/egodata/datasets/ego/chips_cards_tracker_0901/playing_cards/play_cards_0901_023 \
  --stage-root tasks/poker/runs/pipeline/<new_env_migration_run>/eval_poker023 \
  --target-start 103 --target-end 139 \
  --lease-holder <exact_new_holder>
```

Compare against the SHA-pinned historical FIT/EVAL authorities:

```bash
tools/hawor_python.sh tools/verify_hawor_environment_migration.py compare \
  --fit-stage tasks/chips/runs/pipeline/<new_env_migration_run>/fit_chips016 \
  --same-task-stage tasks/chips/runs/pipeline/<new_env_migration_run>/eval_chips008 \
  --cross-task-stage tasks/poker/runs/pipeline/<new_env_migration_run>/eval_poker023 \
  --output tasks/control/runs/<new_env_migration_run>/REPLAY_COMPARISON.json
```

The comparison records whole-NPZ SHA equality, every array's shape/dtype and
bit equality, exact non-floating values, floating max/mean absolute error with
frozen `rtol=0, atol=1e-6`, exact NaN/Inf positions, bit-identical twelve-frame
PNG sheets, and semantic RESULT gates.  Any failing row keeps the authority at
`HOLD_ENV_PATH_MIGRATION`.

## 5. New lock without rewriting history

Only after the probe and all three comparisons pass:

```bash
python tools/verify_hawor_environment_migration.py write-lock \
  --probe tasks/control/runs/<new_env_migration_run>/LOCAL_ENV_PROBE.json \
  --comparison tasks/control/runs/<new_env_migration_run>/REPLAY_COMPARISON.json \
  --output systems/hawor/local_environment_lock.json
```

This does one final content verification and creates the new lock with
`O_EXCL`.  It records, but never edits, the historical environment lock and
runtime pin.  An existing local lock is a conflict and must not be overwritten.

## 6. Explicit rollback

Successful snapshots require the verified tree SHA printed by the tool.  Omit
`--execute` for a dry run:

```bash
python tools/no_clobber_hardlink_snapshot.py rollback \
  --snapshot assets/environments/hawor-py310-v1 \
  --confirm-tree-sha <exact_tree_sha256> \
  --execute
```

For a retained failed staging tree, use its failure evidence UUID:

```bash
python tools/no_clobber_hardlink_snapshot.py rollback-failed \
  --staging assets/environments/.hawor-py310-v1.hardlink-staging.<uuid> \
  --confirm-snapshot-id <exact_uuid> \
  --execute
```

Both flows reject mismatched confirmation values and special/unexpected
objects.  The successful rollback first atomically withdraws the target name;
an interrupted removal retains the quarantined tree as evidence.
