# Task-Owned Execution Notebook Contract

Read this reference only when the selected task explicitly adopts the execution
notebook profile. Existing analysis, visualization, Colab, generator, and
exploratory notebooks do not migrate implicitly.

## Tracked source notebook

Keep one short protocol Markdown cell and exactly three output-clean code cells:

- `openspec-params`: complete kwargs, `run_level=pilot|production`, inputs, and
  output roots
- `openspec-run`: one named EXEC_BLOCK plus `tqdm.auto` only for observability
- `openspec-outputs`: display or point to the Apply-owned run manifest and thin
  outputs

The tracked notebook stores no execution outputs or traceback. Each task's
TEST/Apply packet names the existing execution environment and command; the
Loop adds no notebook executor or Jupyter dependency.

## Apply writer contract

Apply owns execution writes: heavy outputs, an optional run-local executed
notebook, the run manifest, and task-authorized thin pointers. It never rewrites
tracked source-notebook outputs.

The manifest records at least:

- `schema`, `exec_block`, and `run_level`
- `source_notebook_sha`, `canonical_params`, and `params_sha`
- `started_at`, `ended_at`, and `status`
- total, completed, resumed, and failed unit counts
- output paths and hashes
- `resume_decision`

Resume skips a unit only when completion state, `params_sha`, and every required
output hash match. A matching path alone is not sufficient. Keep traceback only
in approved scratch or a run-local execution artifact; progress bars are not
acceptance evidence.

## Verify reader contract

Verify is strictly read-only for the notebook, manifest, pointers, and product.
It checks the three tags, output-clean tracked source, every required manifest
field, source/parameter hashes, unit counts, status, and required output hashes.
A path's existence alone is not completion. Verify never writes back source
notebook outputs, the manifest, or product data after PASS.
