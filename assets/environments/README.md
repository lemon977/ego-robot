# Project-local execution environments

Generated execution environments live here and are not source-controlled.  The
promoted HaWoR environment root is `hawor-py310-v1/`; its canonical authority
is `systems/hawor/environment_authority.json`.  The commands below document a
future fresh recovery and must not overwrite the published V1 tree:

```bash
python tools/no_clobber_hardlink_snapshot.py preflight \
  --source /mnt/workspace/miniconda3/envs/hawor \
  --target assets/environments/hawor-py310-v1

python tools/no_clobber_hardlink_snapshot.py create \
  --source /mnt/workspace/miniconda3/envs/hawor \
  --target assets/environments/hawor-py310-v1
```

Do not add the metadata-risk acknowledgement merely because `df` rounds free
space to zero.  First confirm full59 is terminal, rerun the preflight, and check
the CPFS quota/control plane.  A published tree contains its compressed
manifest at `.chaoyang-hardlink-snapshot/manifest.json.gz`.  Failed staging
trees are retained next to the requested target and require the tool's explicit
`rollback-failed` confirmation flow.

The historical environment lock and vendored runtime pin in
`third_party/HaWoR/` are provenance records.  They are not rewritten when the
project-local runtime is promoted.
