# Chips canonical raw intake — 2026-09-03

The 16-session formal PICO source is available through `tasks/chips/inputs/potato_chips`. Machine-readable sources are:

- `tasks/chips/inputs/raw_source_manifest.json` — canonical paths, historical IDs, immutable migration evidence, and consumption policy.
- `tasks/chips/inputs/canonical_intake_catalog.json` — all 16 sessions with source session/run, media dimensions/FPS/frame counts, stereo/SLAM/tracker/camera/preprocess presence, `all_data` count, intrinsics, PICO21 order, and task-local warnings.

Lightweight intake result: all required components exist and every `preprocess/all_data` directory count matches the declared video and HumanEgo frame count. All 16 `pico_humanego_manifest.json` files nevertheless report `finished_frames: 30`; because the actual `all_data` counts are complete, this is classified as `STALE_FINISHED_FRAMES_COUNTER`, not proof of truncation. Downstream code must not use `finished_frames` as completion authority.

Raw-source reference is authorized, but HaWoR/HumanEgo admission remains per-session and requires the numerical `SOURCE_GATE.json` checks for media decode, timestamps, K/c2w/world convention, PICO hand tracking, and initializer coverage. Mask, Clean, and Robot still require their own stage gates; canonicalization is not a quality-pass waiver.

Catalog SHA: `6cf610620ad0aac7409e8a219559ed21433201a68589e545681026729a192cda`.

