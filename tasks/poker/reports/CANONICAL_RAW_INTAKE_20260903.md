# Poker canonical raw intake — 2026-09-03

The 43-session formal PICO source is available through `tasks/poker/inputs/playing_cards`. It consists of the independent original `play_cards_0901_001..016` run plus legacy-misnamed poker clips `get_potato_chips_0901_017..043`, now canonically named `play_cards_0901_017..043` while retaining source-run provenance in the clip manifests.

Machine-readable sources are:

- `tasks/poker/inputs/raw_source_manifest.json` — canonical paths, historical IDs, immutable migration evidence, and consumption policy.
- `tasks/poker/inputs/canonical_intake_catalog.json` — all 43 sessions with source session/run, media dimensions/FPS/frame counts, stereo/SLAM/tracker/camera/preprocess presence, `all_data` count, intrinsics, PICO21 order, and task-local warnings.

Lightweight intake result: all required components exist and every `preprocess/all_data` directory count matches the declared video and HumanEgo frame count. All 43 `pico_humanego_manifest.json` files nevertheless report `finished_frames: 30`; this is a stale counter warning because the actual frame directories are complete. It must not be used as completion authority.

Raw-source reference is authorized, but numerical HaWoR/HumanEgo admission remains per-session through `SOURCE_GATE.json`. Poker tracker-label quality, same-session Clean donor availability, Object6D, and Robot contact/occlusion gates remain independent HOLD/PASS decisions.

Catalog SHA: `d90b8223b7d7a3a808f3d0a49860ee7747511097183d5c4bd81626dc75fbda6d`.

