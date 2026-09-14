# Chips input views

These are read-only reference views. No video is copied. Exact provenance,
allowed uses, and forbidden uses are authoritative in `../task.manifest.json`.

| View | Role | SHA-256 |
|---|---|---|
| `action_phone_dev.mov` | chips action, PHONE_DEV | `6ec1040488e3eeebf68298e187d767285f61871455559f1e7d13914dcf0e20e5` |
| `object_reference_phone_dev.mp4` | chips/bowl reference, PHONE_DEV | `645340e3e85e3d46b8c3ee58432fc1ab26798b67e6e31082db67a5da315b34b0` |
| `empty_static_phone_dev.mp4` | cross-camera diagnostic only | `40299466ffde64ddbf327f3e3898cdd12aa22f10b12751229c25dbea5dfcc9ec` |
| `empty_sweep_phone_dev.mp4` | cross-camera diagnostic only | `5e667de01c67454378f34786f4edd8bcc5ef7d1dd05a0c85ebcab686896b6e9d` |
| `potato_chips/` | canonical formal PICO batch, 16 sessions | mapping `c3365566772c5dabb72de896128bfc2c2ce739f1bc07a002ae01348c6249595c` |

`raw_source_manifest.json` is the canonical path/evidence authority and
`canonical_intake_catalog.json` is the 16-session lightweight inventory.
Formal stage admission still requires each session's `SOURCE_GATE.json`.

The empty-table views are not formal Clean donors because their camera/layout
does not match the action recording.
