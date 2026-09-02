## ADDED Requirements

### Requirement: The public local runner returns and retains one verifiable receipt
The system SHALL expose `run_abandonment_detection(target_nc, window_year, start_year, current_end_year, init_cropland, extend_validation)` from the existing local abandonment module. A successful call SHALL return a structured receipt containing `status`, `feature`, `parameter_sha256`, input/source-mask-inventory and CONUS-CSV-AOI fingerprints, CSV path and SHA-256, abandonment and merged-chunk roots, expected and verified chunk counts, manifest paths, event and QA counts, and `accepted=true`. The call SHALL write `outputs/s0_us_abandon/{feature}/local_run_abandonment_detection_acceptance.json` only after reopening and validating every retained product. A normal return, log line, notebook output, contract test, or in-memory dataset SHALL NOT substitute for retained products.

#### Scenario: A local feature run materializes complete products
- **WHEN** the runner finishes one declared feature's source/mask chunk inventory and complete-CONUS CSV and every product, schema, hash, AOI, and cutoff assertion passes
- **THEN** it SHALL return `accepted=true` and retain an acceptance receipt whose paths and hashes recompute from the physical files

#### Scenario: The function returns but a product is absent
- **WHEN** any required CSV, chunk family, manifest, or acceptance field is missing, empty without an accepted zero-event explanation, unreadable, or hash-invalid
- **THEN** the run SHALL be rejected and SHALL NOT be reported as a completed local detection

### Requirement: The five local feature identifiers have exact outputs
The only feature identifiers in this amendment SHALL be `300m_min4`, `300m_min6`, `1km_mode_min4`, `1km_mode_min5`, and `1km_mode_min6`. For each `{feature}`, the system SHALL produce exactly these product families:

- `data/us_abandon_clean_{feature}.csv`;
- `D:\xarray\abandon_2_{feature}\chunk_{row}_{col}.nc`;
- `D:\xarray\merged_chunk_2_{feature}\chunk_{row}_{col}.nc`;
- `data/us_abandon_clean_{feature}.manifest.json`;
- `outputs/s0_us_abandon/{feature}/local_run_abandonment_detection_acceptance.json`.

The spelling `chunk` and `merged_chunk_2` is normative. Feature roots SHALL NOT reuse `D:\xarray\abandon_2`, `D:\xarray\merged_chunk_2`, a historical directory, or another feature root.

The large CSV and NetCDF products MAY remain ignored runtime data, but every compact per-feature manifest and acceptance receipt SHALL be Git-visible through the repository ignore contract so that retained evidence is reviewable.

#### Scenario: The full local matrix is accepted
- **WHEN** final local acceptance runs for this change
- **THEN** all five feature identifiers SHALL have one schema-valid CSV, complete abandonment and merged chunk inventories, a manifest, and an accepted receipt with no path collision

#### Scenario: A misspelled or shared root is proposed
- **WHEN** a product uses `chunck`, omits the feature token, or resolves to an existing baseline, historical, or other-feature root
- **THEN** validation SHALL fail before creating or replacing a file

#### Scenario: Compact evidence is ignored by Git
- **WHEN** a per-feature manifest or acceptance receipt is hidden by the repository ignore rules
- **THEN** final local-product acceptance SHALL fail until the compact artifact is explicitly retained

### Requirement: Local manifests bind cutoff-safe scientific content
Each per-feature manifest SHALL bind the six API parameters; feature token; source and eligibility-mask paths, bytes, SHA-256, variables, coordinates, years, CRS, transform, resolution, class and NoData mapping; exact source/mask-derived expected and actual chunk-key sets; exact CONUS CSV bounds, excluded FIPS set, state-shapefile component hashes, transformed geometry inventory, spatial-join predicate, and state membership counts; code fingerprint; every file's relative identity, bytes, and SHA-256; CSV schema and row count; NetCDF variables, dimensions, dtypes, nodata and encoding; and aggregate QA counts.

The manifest SHALL prove that event discovery and `abandonment_duration` stop at `current_end_year=2020`, that `event_start_year + window_year - 1 <= 2020`, that extension years never change 2020 event fields, and that every CSV row passed the declared extension rule. It SHALL record `validated_through_year=2022` when extension is enabled and 2020 otherwise.

#### Scenario: A 2018 event uses 2021--2022 to complete min5
- **WHEN** a candidate has only three consecutive noncrop years by 2020 but five by 2022
- **THEN** it SHALL not qualify for `min5`, SHALL not enter its CSV, and SHALL be counted in cutoff-failure QA rather than backfilled by extension

#### Scenario: Extension changes a main event field
- **WHEN** the same through-2020 sequence produces a different `abandonment_year`, cutoff duration, or `qualifies_at_cutoff` after only 2021--2022 values change
- **THEN** manifest validation SHALL fail

### Requirement: CONUS CSV membership uses the confirmed bbox and exact state polygons
Prediction publication SHALL first prefilter candidate point centers to inclusive bounds `lon_min=-125`, `lon_max=-65`, `lat_min=25`, and `lat_max=49`. It SHALL read `data/cb_2018_us_state_500k.shp`, transform its declared source CRS to EPSG:4326, normalize `STATEFP` to two digits, exclude exactly `02,15,60,66,69,72,78`, and retain candidates only through an inner point-to-state `sjoin(..., predicate="within")`. The retained CSV AOI SHALL therefore contain the contiguous 48 states plus District of Columbia; DC remains part of the AOI even when a feature has zero DC candidate rows.

The bbox SHALL be prefilter-only. Every CSV row SHALL have a point center strictly within one retained state polygon. The main CSV SHALL preserve the established prediction-table schema; acceptance SHALL deterministically rebuild the companion row-key-to-`state_fips`/`state_code`/`state_name` frame returned by `build_prediction_embedding`. Detector and merged NetCDF chunks SHALL instead preserve their separately fingerprinted source/mask-derived expected inventory and MAY contain valid cells outside CONUS; such cells SHALL never enter the CONUS CSV or its event/area counts. The manifest SHALL fingerprint every required shapefile component, CRS, bounds, exclusion set, join predicate, sorted 49-polygon inventory, the sorted membership-frame SHA-256 and state counts, and outside-bbox/outside-state/boundary exclusion QA.

#### Scenario: A bbox candidate is not within an accepted state polygon
- **WHEN** a candidate point is inside the numeric bbox but outside every retained state polygon, on a polygon boundary, or belongs to an excluded FIPS
- **THEN** it SHALL be excluded from the CONUS CSV and accepted event/area counts and SHALL be counted in mask QA

#### Scenario: A California consumer requests local predictions
- **WHEN** P2, RF validation, or the scene gallery consumes a local feature
- **THEN** it SHALL reconstruct the exact `STATEFP=06` subset from the accepted CONUS CSV with the frozen join and bind the parent manifest, membership-frame hash, row-key inventory, and subset SHA-256 rather than create or reinterpret a California-only `us_abandon_clean_{feature}.csv`

#### Scenario: Boundary components drift
- **WHEN** any shapefile component, CRS transformation, or geometry fingerprint differs from the accepted manifest
- **THEN** resume and product promotion SHALL fail closed

### Requirement: Native 300 m rows preserve their mapping to the feature grid
The 300 m features SHALL retain one row per native retained detection pixel and SHALL NOT aggregate those rows into a 1 km detection result. Environmental-feature attachment SHALL bind each row to the accepted `Feature_all` master grid using `feature_grid_row`, `feature_grid_col`, `feature_match_distance`, and `pixel_area`. The mapping method, tolerance, master-grid fingerprint, unmatched count, and duplicate master-cell distribution SHALL be retained. Model splits SHALL group or buffer on the master-cell identity so rows sharing a 1 km cell cannot cross train/holdout folds.

#### Scenario: Multiple native pixels map to one master cell
- **WHEN** two or more 300 m rows map to the same accepted 1 km feature cell
- **THEN** all rows SHALL retain their native identities and shared master-cell identity, and spatial leakage checks SHALL keep them in one fold group

#### Scenario: A native row cannot be mapped within tolerance
- **WHEN** no accepted master cell satisfies the declared coordinate and distance rule
- **THEN** the row SHALL be rejected or explicitly quarantined and SHALL NOT receive guessed environmental features

### Requirement: Local writes are atomic, resumable, and no-overwrite
Every chunk SHALL be written to a same-root temporary path, reopened and validated, and atomically promoted. Resume SHALL reuse a product only when source, mask, parameters, feature, chunk footprint, code, expected chunk key, bytes, and SHA-256 all match. A pre-existing different file or partial inventory SHALL block rather than overwrite, delete, or silently continue. Datasets SHALL be closed promptly and the implementation SHALL avoid loading the full source time cube into memory.

#### Scenario: An interrupted chunk is resumed
- **WHEN** a prior chunk and its manifest entry match all accepted fingerprints
- **THEN** the runner MAY reuse it without recomputation and SHALL record the reuse in the final inventory

#### Scenario: A destination contains different bytes
- **WHEN** any final CSV, NetCDF, manifest, or receipt path already exists with a different fingerprint
- **THEN** the runner SHALL stop without replacing or deleting the path

### Requirement: Every cloud attempt has immutable GEE, GCS, and D-drive identity
Every submitted RF canary or full-training unit SHALL bind `run_id`, stage, model-unit/year key, `task_spec_sha256`, attempt, Earth Engine task and operation IDs, and a previously unused GCS prefix. Every returned object SHALL be pinned by object name, generation, CRC32C, and bytes, downloaded to a temporary D-drive file, verified, atomically promoted, and recorded with local bytes and SHA-256. Unexpected, missing, duplicated, or mismatched objects block acceptance.

#### Scenario: A cloud unit is delivered intact
- **WHEN** the Earth Engine operation succeeds and the exact GCS object set returns with matching generations, CRC32C, bytes, and local SHA-256
- **THEN** the system SHALL retain a checksum-complete attempt manifest and only then permit the next dependent unit

#### Scenario: An attempt prefix or local path already exists
- **WHEN** the proposed prefix contains any object or the local destination contains nonmatching bytes
- **THEN** submission or promotion SHALL be refused without overwrite and a later attempt SHALL require new authorization

### Requirement: Full California RF training has explicit retained deliverables
When the conditional full-training branch is admitted, its retained manifest SHALL enumerate every state-intersecting model-unit/year, training-bundle and predictor-schema hash, random seed, tree count, sampling cap, annual source fingerprint, task and attempt identity, terminal state, actual EECU, returned object, and cost. It SHALL bind one frozen RF model-family identity and annual `p_crop` output family shared by min4/min5/min6. It SHALL assert `direct_abandonment_rf=false`, `p2_rows_in_training=0`, and `classifier_asset_created=false`.

#### Scenario: Full training completes under the accepted cap
- **WHEN** every declared unit and object is complete, checksum-valid, costed, and reconciled while the all-in actual plus remaining worst-case cost never exceeds USD 100.00
- **THEN** the system MAY emit `RF_CALIFORNIA_TRAINING_ACCEPTED` with the exact model-family, inventory, delivery, and cost hashes

#### Scenario: Only a canary or contract test exists
- **WHEN** the full state-intersecting inventory has not been executed and delivered
- **THEN** the terminal state SHALL remain canary-only or budget-blocked and SHALL NOT claim full California RF training

### Requirement: Empty, blocked, canary, complete, and scientifically accepted states are distinct
The system SHALL distinguish `local_product_missing`, `local_zero_event_accepted`, `rf_blocked_by_budget_or_admission`, `rf_canary_checksum_verified`, `rf_california_training_accepted`, and `california_holdout_evaluation_accepted`. A state SHALL be derived from machine-verifiable artifacts and SHALL NOT be inferred from a command exit code, task success, file existence alone, or narrative report.

#### Scenario: A safe budget refusal occurs
- **WHEN** the full RF upper bound is missing, stale, nonfinite, or greater than USD 100.00
- **THEN** the system SHALL retain the blocker and zero-new-full-run-operation proof without fabricating a training or scientific acceptance

#### Scenario: Products and holdout evaluation both pass
- **WHEN** all local products, any authorized RF products, the independent holdout evaluation, and the required manifests pass their separate contracts
- **THEN** final verification SHALL report each terminal state separately rather than collapsing them into one generic success
