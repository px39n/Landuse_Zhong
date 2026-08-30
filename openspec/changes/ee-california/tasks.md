# Active Task Registry

The operative registry is R5--R12. The former sealed R1--R4 registry and its inactive future appendix remain provenance under fingerprint `aeb5278f1439cb4667185348012205ed7327db13f6d22703a2e02c00169f233a`; they are superseded and are not executable under this revision.

## Local Cutoff-Safe Detector

- [x] 5.1 Implement the six-parameter cutoff-safe local detector in the existing module [#R5]
  - DEPENDS_ON: none
  - STATE: passed
  - ACCEPT: `function/cropland_abandonment.py` exposes `run_abandonment_detection(target_nc=Path(r"D:\xarray\reclass_lccs_1km.nc"), window_year=5, start_year=1992, current_end_year=2020, init_cropland=2, extend_validation=True)` and reuses existing writers/helpers rather than adding a duplicate detector module.
  - ACCEPT: `init_cropland` is a positive count applied to both the start-of-analysis crop eligibility and the crop prefix immediately before an exit; `window_year` controls every candidate mask, detector, QA, attribute, and manifest duration check with no hidden five-year constant.
  - ACCEPT: Event discovery, event completion, `abandonment_year`, `abandonment_duration`, `qualifies_at_cutoff`, and the 2020 current state use only `start_year..current_end_year`. With extension enabled, exactly `current_end_year+1` and `+2` must both be valid noncrop/nonbuilt and affect only persistence inclusion; missing, crop, built, NoData, or invalid extension values fail the pixel without changing its 2020 event fields.
  - ACCEPT: Golden tests distinguish the intentional correction from legacy full-axis behavior, including 2018 events that only reach min5 in 2022, isolated post-cutoff crop, built, NoData, missing extension years, recultivation, multiple exits, and `extend_validation=False`; all local paths create zero GEE operations, GCS objects, or Assets.
  - TEST: SCOPE: CLI; Run: python -m pytest tests/test_cropland_abandonment.py tests/test_abandonment_legacy_audit.py tests/test_mode2_runner_contract.py -q

- [ ] 6.1 Route the runner, generator, and final active Process cell through the public API [#R6]
  - DEPENDS_ON: R5
  - STATE: pending
  - ACCEPT: `tools/run_mode2_end_to_end.py`, the prediction/multiscale orchestration that owns the same path, and `tools/update_process_notebook_contract.py` call the public function instead of copying regex, Numba, cutoff, extension, chunk, or manifest logic.
  - ACCEPT: `build_prediction_embedding` and the owning prediction runner implement the confirmed local AOI exactly: inclusive bbox `[-125,-65] x [25,49]`, `data/cb_2018_us_state_500k.shp`, excluded `STATEFP=02,15,60,66,69,72,78`, and inner point-center `sjoin(..., predicate="within")`. The complete CSV preserves the established prediction schema; acceptance rebuilds the companion state-membership frame, binds its row-key/state hash and counts, and California consumers reconstruct a hash-bound `STATEFP=06` subset without publishing a California-only `us_abandon_clean_{feature}.csv`.
  - ACCEPT: The final active cell in `Process.ipynb` contains only imports, the six canonical parameters, the public call, and compact receipt inspection. Commented legacy cells remain byte-preserved, and notebook/generator parity tests prove regeneration produces the same active cells.
  - ACCEPT: Direct Python, CLI, and Notebook invocations with the canonical parameters produce the same parameter/AOI SHA-256, expected CONUS chunk inventory, output schema, and acceptance-receipt schema. Source datasets are opened lazily, sliced once, spatially chunked, closed promptly, atomically written, and resumed only on complete fingerprint equality.
  - TEST: SCOPE: CLI; Run: python -m pytest tests/test_process_notebook_contract.py tests/test_mode2_runner_contract.py tests/test_multiscale_embedding_contract.py tests/test_embedding_pipeline.py -q

## Pre-Model Reference and Independent P2 Holdout

- [ ] 7.1 Freeze cutoff-safe P1 evidence, 36 core keys, eight legacy keys, and the decision family [#R7]
  - DEPENDS_ON: R6
  - STATE: pending
  - ACCEPT: A new no-overwrite P1 attempt contains exactly 36 core candidates under the 12/9/6/3/6 diagnostic portfolio and all eight separate 2017/2018 legacy-boundary records, with source/alignment hashes, reserve order, blank review templates, and no local/RF/Landsat-model/post-2020 prediction leakage.
  - ACCEPT: `decision_contract_family_sha256` is frozen before review or prediction reveal and binds the five local feature IDs, input/grid fingerprints, confirmed CONUS bounds, excluded FIPS, state-shapefile identity, inner point-within join, deterministic `state_fips=06` subset rule, 1992/2020/2/two-year-extension semantics, valid class and NoData rules, inclusive min4/min5/min6 projections, RF training/model family, probability definition and threshold, and prohibition on per-window RF retraining.
  - ACCEPT: P1 creates zero paid Earth Engine operations, GCS writes, or Assets; every v1 and attempt-8 sample key remains immutable historical audit only.
  - TEST: SCOPE: CLI; Run: python -m pytest tests/test_s2_california_2020_sampling_attempt_manifest.py tests/test_s2_california_sampling.py tests/test_s2_california_review_evidence_2020.py -q

- [ ] 8.1 Complete and freeze the independent P2 holdout before any prediction reveal [#R8]
  - DEPENDS_ON: R7
  - STATE: pending
  - ACCEPT: P2 completes exactly 44 primary reviews (36 core plus 8 legacy), the deterministic 12-record blind-secondary subset (8 core plus 4 legacy), reviewer-independence checks, every disagreement, and every interval crossing the 2015/2016, 2016/2017, or 2017/2018 duration boundary; only unusable/still-unresolved core records use the frozen reserve, while legacy records are never replaced or dropped.
  - ACCEPT: Rule-neutral transition facts deterministically produce `qualifies_min4_2020`, `qualifies_min5_2020`, and `qualifies_min6_2020`; the 36 core keys are the sole primary metric denominator, the eight legacy keys remain a separate boundary appendix, and the secondary subset is reviewer-QA only.
  - ACCEPT: `P2_ACCEPTED.json` binds the decision-family, core, legacy, combined-freeze, review, adjudication, source, coordinate, and projection hashes. No P2 value trains or tunes RF, thresholds, window duration, resolution, features, candidate selection, or replacement selection, and no post-2020 observation mutates a 2020 field.
  - TEST: SCOPE: CLI; Run: python -m pytest tests/test_s2_california_review_evidence_2020.py tests/test_s2_california_2020_validation.py tests/test_spectral_abandonment.py -q

## Complete Detector Products and CONUS Prediction Tables

- [ ] 9.1 Materialize and verify all five detector/merged inventories and complete-CONUS CSVs [#R9]
  - DEPENDS_ON: R6,R8
  - STATE: pending
  - ACCEPT: A bounded canary first proves the canonical 300 m and 1 km sources, source/mask-derived chunk inventory, cutoff/extension behavior, atomic no-overwrite writes, native-to-master-grid mapping, the confirmed CONUS bbox and exact state-shapefile/exclusion/inner-point-within CSV rule, companion membership-frame hash, and acceptance-receipt validator on a small declared chunk set.
  - ACCEPT: Full local execution then materializes `300m_min4`, `300m_min6`, `1km_mode_min4`, `1km_mode_min5`, and `1km_mode_min6`; each has a nonmissing `data/us_abandon_clean_{feature}.csv`, `data/us_abandon_clean_{feature}.manifest.json`, `D:\xarray\abandon_2_{feature}\chunk_{row}_{col}.nc`, `D:\xarray\merged_chunk_2_{feature}\chunk_{row}_{col}.nc`, and `outputs/s0_us_abandon/{feature}/local_run_abandonment_detection_acceptance.json`.
  - ACCEPT: Every acceptance receipt is rebuilt from physical files and proves complete source/mask-derived expected/actual chunk-key equality, nonempty products or an explicitly schema-valid zero-event result, per-file bytes/SHA-256, CSV and NetCDF schema/encoding, exact CONUS CSV bounds, excluded-FIPS equality, 49-polygon AOI inventory, `within` membership for every CSV row, companion state-membership-frame hash/counts, outside-state and boundary QA, no feature/path collision, cutoff assertions, extension QA, 300 m native-row retention, and shared-master-cell leakage protection. NetCDF cells outside CONUS are permitted only under the accepted source/mask chunk contract and cannot enter the CSV. Compact manifests/receipts are explicitly Git-visible even when large CSV/NetCDF products remain ignored. A passing unit test without these physical products does not satisfy R9.
  - ACCEPT: Existing `abandon_2`, `merged_chunk_2`, history, P2, and prior feature roots remain byte-unchanged; no local run creates a cloud operation, GCS object, or Asset.
  - TEST: SCOPE: CLI+LOCAL-PRODUCT; Run: python -m pytest tests/test_cropland_abandonment.py tests/test_embedding_pipeline.py tests/test_multiscale_embedding_contract.py tests/test_mode2_runner_contract.py -q; python tools/run_prediction_mode_end_to_end.py --verify-products-only --scope conus --features 300m_min4 300m_min6 1km_mode_min4 1km_mode_min5 1km_mode_min6

## Conditional California RF Execution

- [ ] 10.1 Run the RF canary and conditionally execute full California training under the USD 100 cap [#R10]
  - DEPENDS_ON: R8,R9
  - STATE: pending
  - ACCEPT: The frozen RF remains the annual binary crop/noncrop 300-tree `MULTIPROBABILITY` graph with crop probability at class index 1, identical training/model/annual-series fingerprints for min4/min5/min6, `direct_abandonment_rf=false`, zero P2/gallery training rows, and zero classifier-Asset creation.
  - ACCEPT: Before full execution, all five detector/merged inventories and complete-CONUS CSV receipts pass and their parent-manifest/membership-frame-bound `STATEFP=06` subset identities are frozen; these prediction rows are validation inputs, not RF training labels. An exact California state-intersecting model-unit/year manifest then passes the shared no-submit route, fresh project/principal/Billing/credit/quota/Asset/bucket gates, and one minimal production-path RF unit completes the GEE-to-GCS-to-D-drive chain with terminal OperationMetadata/EECU and generation/CRC32C/bytes/local-SHA verification.
  - ACCEPT: A within-60-minute budget acceptance binds actual spend, exact remaining work and attempts, conservative measured EECU, safety factor, fresh official compute price, GCS operations/storage, download/egress, every other paid component, verified funds and reserve, and proves finite `safe_total_upper_usd <= 100.00`. Unknown, stale, mismatched, nonfinite, or over-cap evidence creates zero new full-run operations.
  - ACCEPT: If the budget and all gates pass, the exact full California training manifest executes serially with `max_inflight=1`, no automatic retry, immutable attempt prefixes, admission before every unit, and an actual-plus-worst-case-remaining cap check. If they cannot pass, R10 retains `RF_CALIFORNIA_TRAINING_BLOCKED_BY_BUDGET_OR_ADMISSION` and unchanged full-run operation inventory; it never claims training complete. Only complete inventory, object, cost, and model-family evidence may emit `RF_CALIFORNIA_TRAINING_ACCEPTED`.
  - ACCEPT: R10 may pass through either machine-verified terminal branch: accepted full training, or a budget/admission blocker that proves zero new full-run operations. The blocker branch unlocks only RF-unavailable reporting and the descriptive gallery in R11; it does not satisfy or imply RF training completion.
  - TEST: SCOPE: CLI+LIVE-CONDITIONAL; Run: python -m pytest tests/test_s2_rf_graph.py tests/test_s2_cdl_training.py tests/test_s2_validation_budget.py tests/test_s2_gee_deployment.py tests/test_s2_gee_output.py tests/test_s2_california_validation_contract.py -q

## Frozen Evaluation, Gallery, and Final Acceptance

- [ ] 11.1 Evaluate available frozen variants on P2 and build the post-freeze scene gallery [#R11]
  - DEPENDS_ON: R9,R10
  - STATE: pending
  - ACCEPT: Every local feature is first deterministically extracted as `STATEFP=06` from its accepted complete-CONUS CSV by rebuilding the frozen spatial join, binding parent manifest/CSV and membership-frame hashes, subset rule, row-key inventory, and subset SHA-256; it and every available accepted RF output are then joined by frozen sample identity and family hash to the exact 36-core holdout. Reports include numerators, denominators, confusion counts, defined Wilson intervals, `diagnostic_holdout=true`, `not_statewide_accuracy=true`, and `used_for_tuning=false`. Eight legacy records remain a separate boundary table and a budget-blocked RF is reported as unavailable rather than synthesized.
  - ACCEPT: A new append-only `post_freeze/{gallery_id}` binds `P2_ACCEPTED`, combined reference and decision-family hashes, sample keys, model/variant manifests, and media hashes. It covers all 36 core records by default and keeps eight legacy records in a labeled appendix; any smaller predeclared selection is hash-bound and marked nonrepresentative.
  - ACCEPT: Every scene includes true NDVI and McFeeters NDWI formulas/series, separately named NDMI if present, annual observation/QA/source identity, NAIP acquisition metadata/hash, official survey release/raw-slot/decoded evidence, frozen reference projections, and variant identities. All training, tuning, primary-metric, area, PV-mask, label-change, and scientific-acceptance eligibility flags are false, and P2 bytes/hashes remain unchanged.
  - TEST: SCOPE: CLI; Run: python -m pytest tests/test_s2_california_2020_validation.py tests/test_s2_california_review_evidence_2020.py tests/test_s2_california_landsat_review_curves.py tests/test_spectral_abandonment.py -q

- [ ] 12.1 Verify the whole change and publish exact terminal evidence [#R12]
  - DEPENDS_ON: R11
  - STATE: pending
  - ACCEPT: Whole-change verification distinguishes local product acceptance, P2 reference acceptance, RF canary, RF full-training accepted or budget/admission-blocked branch, holdout evaluation, and gallery completion; no command exit, contract-test pass, or cloud-task success substitutes for a required retained artifact.
  - ACCEPT: All active task refs, dependencies, ACCEPT/TEST hashes, complete-CONUS product manifests, exact bbox/exclusion/point-within AOI, 49-polygon inventory, feature paths, deterministic California subset identities, reference/model/cost fingerprints, forbidden-write assertions, and actual external-operation inventories are mutually consistent. Prior unfeatured roots, historical attempts, and accepted P2 artifacts remain unchanged.
  - ACCEPT: Final external-effects audit proves that CONUS authority was used only for local detection/product publication and that every GEE, GCS, paid RF, and live cloud operation remained limited to the exact California manifest.
  - ACCEPT: `openspec validate ee-california --strict`, Loop advisory planning/checking, the full declared final test profile, UTF-8/mojibake checks, and a final no-unaccounted-external-effects audit pass before whole-change acceptance.
  - TEST: SCOPE: CLI; Run: python -m pytest tests/test_cropland_abandonment.py tests/test_process_notebook_contract.py tests/test_mode2_runner_contract.py tests/test_multiscale_embedding_contract.py tests/test_s2_california_sampling.py tests/test_s2_california_review_evidence_2020.py tests/test_s2_california_2020_validation.py tests/test_s2_rf_graph.py tests/test_s2_cdl_training.py tests/test_s2_validation_budget.py tests/test_s2_gee_deployment.py tests/test_s2_gee_output.py tests/test_s2_california_landsat_review_curves.py tests/test_s2_california_validation_contract.py -q; openspec validate ee-california --strict

# Non-Operative Provenance

The previous R1--R4 active registry and its P3--P7 planning appendix are superseded by this confirmed R5--R12 registry. Their exact wording remains recoverable from the prior sealed fingerprint and repository history; it is intentionally not duplicated here and SHALL NOT be parsed into `feature_list.json`.
