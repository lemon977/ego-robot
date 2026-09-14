# Poker input views

These are read-only reference views. No video is copied. Exact provenance,
allowed uses, and forbidden uses are authoritative in `../task.manifest.json`.

| View | Role | SHA-256 |
|---|---|---|
| `action_phone_dev.mov` | poker action, PHONE_DEV | `65e9372e2dcc4593d96866879c7c1345d8c616adbfebe5bfda55c3f8c695db7c` |
| `object_reference_phone_dev.mp4` | cards/rack reference, PHONE_DEV | `d041d3ce3b76b32c45b8ace7437e741567c297e4d6611e54d746b7c33f4013ec` |
| `empty_static_phone_dev.mp4` | cross-camera diagnostic only | `40299466ffde64ddbf327f3e3898cdd12aa22f10b12751229c25dbea5dfcc9ec` |
| `empty_sweep_phone_dev.mp4` | cross-camera diagnostic only | `5e667de01c67454378f34786f4edd8bcc5ef7d1dd05a0c85ebcab686896b6e9d` |
| `playing_cards/` | canonical formal PICO batch, 43 sessions | mapping `c3365566772c5dabb72de896128bfc2c2ce739f1bc07a002ae01348c6249595c` |

`raw_source_manifest.json` is the canonical path/evidence authority and
`canonical_intake_catalog.json` is the 43-session lightweight inventory.
Formal stage admission still requires each session's `SOURCE_GATE.json`.

The empty-table views are not formal Clean donors because their camera/layout
does not match the action recording.
