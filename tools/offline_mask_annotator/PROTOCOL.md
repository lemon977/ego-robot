# Offline Mask Annotator protocol

Version: 1.0.0

## Scope and privacy

This directory is a self-contained offline annotation application.  It contains
code, static web assets and a synthetic self-test only.  It must never contain a
project MP4, extracted project frame, annotation from the main repository,
heldout/blind identifier, or absolute path copied from an existing dataset.

The application binds only to `127.0.0.1` by default and performs no network
requests.  Browser assets are shipped locally.

## User contract

For a normal CFR MP4 the user edits only these three values in `config.json`:

- `input_mp4`
- `output_dir`
- `session_id`

The default deterministic sample count is 24.  Advanced options may change the
count or point at a `timestamps.csv`.  CSV rows must contain a
`timestamp_seconds` column.  Each requested time resolves to the nearest unique
decoded video-frame PTS reported by ffprobe.

## Frozen sampling

1. ffprobe enumerates every video frame with best-effort timestamp and time.
2. Without a CSV, select rounded linspace indices over the decoded frame list,
   including the first and last frame.  Duplicate indices are forbidden.
3. With a CSV, resolve requested timestamps to nearest unique PTS; ties choose the
   lower decoded index.
4. ffmpeg extracts exactly those decoded indices.  The output PNG checksum, source
   MP4 checksum, decoded index and PTS are written before annotation begins.
5. Re-running `prepare` resumes only when the input/session/sampling identity is
   unchanged; otherwise it fails closed and asks for a new output directory.

This avoids assuming `frame_number / fps` for VFR or dropped-frame material.

## Label schema

The class-id PNG is exhaustive and mutually exclusive.  Visible pixels only are
annotated; no amodal filling behind occluders.

| id | symbol | meaning |
|---:|---|---|
| 0 | `B_BACKGROUND` | table/scene/background people/non-target pixels |
| 1 | `L_SKIN` | visible left fingers, palm, wrist and bare forearm |
| 2 | `R_SKIN` | visible right fingers, palm, wrist and bare forearm |
| 3 | `L_SLEEVE` | visible left sleeve and cuff |
| 4 | `R_SLEEVE` | visible right sleeve and cuff |
| 5 | `L_TRACKER` | visible left tracker, wristband and housing |
| 6 | `R_TRACKER` | visible right tracker, wristband and housing |
| 7 | `O_TASK_OBJECT` | visible manipulated task object of any shape |
| 8 | `U_UNCERTAIN` | genuinely ambiguous contact/blur/mixed pixels |

The raster starts as background.  Polygon and brush operations overwrite prior
class ids in order.  Painting background is therefore the eraser.  `complete`
is an explicit human assertion; the validator cannot infer a semantically missed
sleeve from pixels alone.

## Resumable annotation

Each frame stores an append-ordered vector operation list in JSON and an atomic
draft class-id PNG.  Polygon
vertices, brush centreline points and brush radius are in original image pixels.
Undo/redo is operation-based.  Atomic server writes and monotonically increasing
revision numbers prevent a stale browser tab from silently overwriting newer work.

## Export

For every sampled frame export:

- one 8-bit class-id PNG;
- nine 0/255 binary masks;
- one RGB overlay;
- the resumable annotation JSON and its draft class-id PNG;
- frame/source identities, PTS, dimensions and SHA-256 checksums.

`validate` checks manifest identity, dimensions, checksums, completion, operation
bounds, known class ids, binary values, one-hot/exhaustive partition, class-id
agreement and empty classes across the project.  It cannot certify semantic
correctness; independent human review remains required.

## Planned portable layout

```text
offline_mask_annotator/
  annotator.py
  config.json
  requirements-lock.txt
  start_linux.sh
  start_windows.bat
  README_ZH.md
  mask_annotator/
    config.py media.py project.py raster.py server.py validation.py selftest.py
  web/
    index.html app.js styles.css
  tests/
    test_core.py
```
