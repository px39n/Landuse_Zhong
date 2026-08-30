## ADDED Requirements

### Requirement: One shared gate protects every California cloud entry point
Every CLI, Python runner, Notebook-rendered command, task planner, queue/resume path, RF helper, and direct California export function SHALL call one shared fail-closed gate before constructing or submitting a live task. The gate SHALL bind the OpenSpec fingerprint, active task ref, P2 and decision-family hashes, local-product acceptance, deployment identity, exact task manifest, fresh budget acceptance, current operation inventory, and expected output roots.

Dry-run SHALL execute the same validation and task-spec construction path while creating zero Earth Engine operations, zero GCS objects, zero Assets, and zero local production products. A flag, credential, old successful task, P2 file, or Notebook cell SHALL NOT bypass a missing gate.

#### Scenario: A live-capable route lacks one acceptance
- **WHEN** any entry point lacks, mismatches, or cannot recompute a required gate input
- **THEN** it SHALL return a machine-readable blocked state before creating an Earth Engine operation or external write

#### Scenario: A dry-run validates a task graph
- **WHEN** the exact canary or full manifest is dry-run
- **THEN** the system SHALL retain task-spec and operation-inventory evidence while proving zero external mutation

### Requirement: Commercial deployment identity is explicit and fresh
Live California work SHALL use compute project `project-c4d41f60-0112-4481-a18`, Billing Account `01C80D-B5781C-DF2518`, principal `pyzhong98@gmail.com` or an explicitly accepted replacement, storage project `trial-285112`, bucket `gs://pv_cropland`, read-only Asset/quota root `projects/project-c4d41f60-0112-4481-a18/assets/`, GCS run root `gs://pv_cropland/s2_abandon_detect/{run_id}/`, and local run root `D:\xarray\s2_abandon_detect\{run_id}\`.

A deployment acceptance SHALL verify the active principal, project registration and access, Billing linkage, remaining credit/balance, cross-project bucket existence/write permission, read-only Asset dependencies and capacity, local-root writability, and equality to the task manifest. Credentials and tokens SHALL remain memory-only and SHALL NOT be serialized.

#### Scenario: The identity is current and exact
- **WHEN** all project, principal, Billing, bucket, Asset, and local-root probes match the immutable manifest
- **THEN** the gate MAY retain a timestamped deployment acceptance containing response hashes and no credential material

#### Scenario: A generic default or stale identity is observed
- **WHEN** the runner would use `trial-285112` as compute project, another bucket/root, an unaccepted principal, or stale/missing probe evidence
- **THEN** it SHALL fail before task creation

### Requirement: The Earth Engine RF remains one annual crop/noncrop model family
The RF target SHALL remain binary annual crop/noncrop. The graph SHALL use the accepted CDL/stable-zone training contract, fixed predictor schema and order, frozen training-bundle identities, 300 trees, deterministic seeds, `smileRandomForest`, and `MULTIPROBABILITY`; `p_crop` SHALL be class index 1. `direct_abandonment_rf` SHALL be false and P2/reference/gallery fields SHALL be rejected from training inputs.

The full manifest SHALL enumerate exact state-intersecting model-unit/year work. Min4, min5, and min6 SHALL consume one frozen annual `p_crop` sequence/model-family fingerprint and differ only in temporal projection. A duration or local-grid variant SHALL NOT retrain the RF or change its threshold.

#### Scenario: Three duration rules consume one RF family
- **WHEN** min4, min5, and min6 are evaluated
- **THEN** their training-bundle, predictor, seed, tree-count, threshold, model-family, and annual-series hashes SHALL be identical

#### Scenario: P2 or a direct-abandonment target enters training
- **WHEN** the training graph receives any reference label, review field, gallery field, abandonment-duration target, or holdout-derived choice
- **THEN** graph validation SHALL reject the task before submission

### Requirement: Local products and P2 precede paid RF execution
The only valid forward dependency order SHALL be:

1. cutoff-safe API and notebook/runner parity;
2. pre-model P1 packet and `decision_contract_family_sha256`;
3. independent `P2_ACCEPTED` freeze;
4. complete accepted five-feature detector/merged inventories, complete-CONUS CSVs, and deterministic California subset identities;
5. no-submit RF manifest and one admitted minimal GEE-to-GCS-to-D-drive canary;
6. conditional full California RF training;
7. frozen holdout evaluation and post-freeze gallery;
8. whole-change acceptance.

No later state SHALL be inferred from a successful command or existing file. Any bound source, parameter, sample, code, task, environment, cost, or output fingerprint change invalidates the dependent state and every later state.

#### Scenario: RF is requested before P2 or local acceptance
- **WHEN** P2, the decision family, any of the five required detector/merged inventory plus complete-CONUS CSV receipts, or the deterministic California subset identity is absent or invalid
- **THEN** the shared gate SHALL create zero paid operations and return the exact missing predecessor

#### Scenario: A predecessor changes after acceptance
- **WHEN** a bound predecessor fingerprint drifts
- **THEN** dependent canary, budget, full-run, evaluation, and gallery acceptances SHALL become invalid without deleting retained evidence

### Requirement: A checksum-verified minimal canary precedes full California training
The canary SHALL use the exact production RF code path, predictor/label schema, model-family contract, commercial identity, GCS/D-drive return path, and cost ledger on the smallest declared model-unit/year capable of measuring real training and inference EECU. It SHALL have an immutable task spec, one new submission, `max_inflight=1`, zero automatic retries, and a previously unused attempt prefix.

Canary acceptance SHALL require successful terminal OperationMetadata, actual EECU, exact GCS object inventory, generation/CRC32C/bytes, local bytes/SHA-256, schema and probability validation, and zero Asset creation. Its result SHALL be `rf_canary_checksum_verified`, `canary_only=true`, and `california_training_accepted=false`.

#### Scenario: The canary chain is complete
- **WHEN** the single admitted task and every returned object pass identity, schema, checksum, and actual-cost validation
- **THEN** its measurements MAY enter the conservative full-run budget model

#### Scenario: The canary fails or has incomplete cost/delivery evidence
- **WHEN** the task fails, an object or checksum differs, EECU is absent, or a required field is invalid
- **THEN** the stage SHALL stop without automatic retry or full-run admission

### Requirement: Full California RF training requires an all-in upper bound at or below USD 100
Full training SHALL require a budget acceptance observed no more than 60 minutes before submission. It SHALL bind the exact full task/attempt inventory; successful and failed canary EECU; conservative measured EECU-hours per unit; safety factor; fresh official compute price and source hash; actual spend to date; GCS operation, storage, download/egress, and every other paid-component upper bound; verified remaining funds and reserve; and `hard_cap_usd=100.00`.

The gate SHALL calculate:

```text
safe_total_upper_usd =
    actual_spend_to_date_usd
    + safety_factor * exact_remaining_work_units
      * conservative_measured_eecu_hours_per_unit
      * current_usd_per_eecu_hour
    + gcs_operations_upper_usd
    + storage_upper_usd
    + download_or_egress_upper_usd
    + other_paid_component_upper_usd
```

All terms SHALL be finite, current, traceable, and denominated in USD. Verified available funds SHALL cover `safe_total_upper_usd` plus the declared unavailable reserve. The full run MAY be admitted only when `safe_total_upper_usd <= 100.00`. Unknown quantities SHALL be blockers, not zero. Splitting work, changing prefixes, or omitting failed attempts SHALL NOT evade the cap.

#### Scenario: The full run fits the cap
- **WHEN** every prerequisite passes and the fresh accepted all-in upper bound is finite and at most USD 100.00
- **THEN** the user's confirmed conditional authorization permits the exact manifest to enter serial full-California execution

#### Scenario: The bound is high, stale, or incomplete
- **WHEN** the upper bound is greater than USD 100.00 or any price, EECU, task-count, balance, reserve, storage, operations, egress, or other paid term is missing or stale
- **THEN** the system SHALL emit `blocked_by_budget_or_admission`, prove zero new full-run operations, and SHALL NOT reinterpret the confirmation as unconditional spending authority

### Requirement: Full training is serial, attempt-complete, and stops at the cap
Full California RF training SHALL use `max_inflight=1` and zero automatic retries. Before every submission, the gate SHALL reconcile actual spend from all terminal attempts and recompute actual plus worst-case remaining cost. Every failed, cancelled, timed-out, checksum-invalid, or scientifically invalid attempt SHALL remain in the cost and object ledgers.

A later retry MAY occur only with a new attempt number, unused prefix, explicit retry reason, fresh admission, and proof that the complete remaining branch still fits the USD 100 cap. The runner SHALL stop before the next task whenever the cap, reserve, quota, capacity, identity, delivery, or predecessor invariant no longer passes.

#### Scenario: A full-run unit succeeds
- **WHEN** its operation, objects, local files, and actual EECU are accepted
- **THEN** the ledger SHALL close the attempt and recompute admission before the next unit

#### Scenario: A unit fails
- **WHEN** any terminal or downstream acceptance fails
- **THEN** the evidence and actual cost SHALL be retained, no retry SHALL be automatic, and no next unit SHALL start under the old budget acceptance

### Requirement: Conditional authorization has two truthful terminal branches
The RF task SHALL end in exactly one of these branches:

- `RF_CALIFORNIA_TRAINING_ACCEPTED`, only after the exact full inventory, outputs, checksums, actual costs, and model-family manifest pass; or
- `RF_CALIFORNIA_TRAINING_BLOCKED_BY_BUDGET_OR_ADMISSION`, when the full upper bound or another required gate cannot pass, with an unchanged full-run operation inventory after the block.

The blocked branch satisfies the safety requirement but SHALL NOT be reported as completed RF training. Holdout and gallery outputs SHALL mark RF unavailable rather than synthesize results. A canary, partial task set, or projected budget SHALL NOT be promoted to the accepted branch.

Either branch MAY satisfy the conditional-execution task itself after machine verification. Only the accepted branch satisfies RF-training completion; the blocked branch permits downstream RF-unavailable reporting and descriptive-gallery work without weakening or fabricating the missing RF result.

#### Scenario: Budget proof cannot be obtained
- **WHEN** canary evidence is retained but the all-in full-run bound remains unknown or over cap
- **THEN** final reporting SHALL preserve the blocker, any canary evidence, and `rf_result_available=false`

#### Scenario: Every full unit is accepted
- **WHEN** exact inventory completion, checksum delivery, actual cost, and model-family verification all pass
- **THEN** the system MAY emit the accepted branch without implying holdout or statewide scientific accuracy

### Requirement: Production evidence and scientific evidence remain separate
Task success, RF training acceptance, local matrix acceptance, P2 acceptance, holdout evaluation, and gallery completion SHALL have separate hashes and terminal states. Full training means the declared annual RF work and retained outputs exist; it does not by itself mean abandonment accuracy, statewide prevalence, area accuracy, or photovoltaic suitability is accepted.

#### Scenario: Cloud compute completes before evaluation
- **WHEN** RF training artifacts are accepted but the independent 36-core evaluation is absent or invalid
- **THEN** the system SHALL retain training acceptance and SHALL NOT emit scientific acceptance

#### Scenario: A gallery appears persuasive
- **WHEN** qualitative scenes agree with one model or variant
- **THEN** they SHALL not alter the frozen metric denominator, method choice, labels, or production claims

### Requirement: Local CONUS authority cannot expand into CONUS cloud execution
This change authorizes only the five declared local CONUS detection/product families. It SHALL NOT authorize CONUS Earth Engine tasks, nationwide RF training or inference, nationwide spending, CCDC/LandTrendr, classifier Asset creation, or an undeclared diagnostic. Every live cloud/RF task in this change remains limited to the exact California manifest. Broader cloud work requires a separately interviewed and sealed change with its own sources, task inventory, budget, outputs, and acceptance.

#### Scenario: A caller broadens the exact California cloud manifest
- **WHEN** a live task, queue, resume request, or cloud output prefix contains CONUS/nationwide scope, another undeclared geography, algorithm, Asset mutation, or diagnostic
- **THEN** the shared gate SHALL reject it before any operation or external write
