# AGENTS.md

## Purpose
This file provides guidance for automated coding agents working in this
repository. Keep changes small, focused, and aligned with the existing
research pipeline.

## Project layout
- Root-level notebooks drive the end-to-end research workflow.
- `function/` contains legacy helper modules used by notebooks.
- `src/landuse/` contains packaged modules used by pipeline scripts.
- `pipelines/global/` contains staged pipeline scripts.
- `cloud/gcp/` contains Dockerfiles and GCP batch tooling.
- `docs/` contains longer-form documentation.

## Environment setup
Preferred (conda):
```
conda env create -f geo.yml
conda activate geo
```

Alternative (pip):
```
python -m pip install -r requirements.txt
```

GCP container requirements live in `cloud/gcp/requirements.txt`.

## Data and outputs
- Large datasets and model artifacts are not stored in the repo.
- Notebooks often require local path updates before execution.
- Do not commit generated data, figures, or model binaries.

## Running workflow
- The main workflow is notebook-driven. Use Jupyter to run the relevant
  notebooks in sequence.
- Pipeline scripts can be run with Python when data/configs exist, e.g.:
  `python pipelines/global/stage1_align.py`

## Tests
No automated test suite is provided. Avoid running heavyweight notebooks
unless explicitly requested. If a quick check is needed, use lightweight
imports or `python -m py_compile` on touched modules.

## Coding conventions
- Keep edits localized; avoid reformatting unrelated code.
- Use existing module locations (`function/` or `src/landuse/`).
- Prefer ASCII-only comments/strings unless existing content requires
  non-ASCII text.
- Limit notebook edits to the minimal required cells.

## Documentation
Update README or docs when adding user-facing behavior or new steps.
