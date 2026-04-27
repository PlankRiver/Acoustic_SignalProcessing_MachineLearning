# Dataset Sync Policy

## Current sync status
- Local branch tracks `origin/main`.
- Only Git-tracked files are synced to GitHub.
- Files matched by `.gitignore` are local-only unless already tracked historically.

## Upload only a subset
- Keep full dataset local (outside tracked sample directories).
- Keep only samples in one of these tracked paths:
  - `data/samples/`
  - `datasets/samples/`
  - `sample_data/`

## Commit limits (enabled by pre-commit)
- Max data files per commit: `500`
- Max total data size per commit: `150 MB`
- Max single data file: `15 MB`

You can override defaults for one commit:

```bash
DATASET_MAX_FILES=200 DATASET_MAX_BYTES=52428800 git commit -m "..."
```

Emergency bypass (not recommended):

```bash
BYPASS_DATASET_GUARD=1 git commit -m "..."
```

## Enable hooks in this clone

```bash
bash z_scripts/setup_git_hooks.sh
```
