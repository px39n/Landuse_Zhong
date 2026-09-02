## Context

The legacy Mode detector searches the complete 1992--2022 sequence. Its initial predicate is `startswith("11")`, its event predicate is equivalent to `1{2,}(0{5,})`, and its current flag rejects only a later two-year crop sequence or built class. Consequently 2021--2022 can create or extend an event later projected to 2020, a single later crop year can pass, and NoData can be coerced to noncrop. The S0 path stops at 2020 but uses a different six-year baseline and recultivation algorithm. This amendment preserves the two useful legacy crop-prefix predicates while deliberately correcting cutoff, persistence, NoData, parameter, and output semantics.

The local and RF lanes remain distinct. Local robustness compares CONUS LCCS trajectories at native 300 m and categorical-mode 1 km. The California Earth Engine RF remains a 30 m annual crop/noncrop probability model; abandonment windows are post-processing rules over one frozen annual series.

## Goals / Non-Goals

**Goals:**

- One six-parameter, memory-safe local API with a hard 2020 decision horizon and optional fixed 2021--2022 persistence validation.
- Physical, hash-verifiable source/mask-driven detector and merged chunks plus complete-CONUS prediction CSVs for five predeclared features, with deterministic California subsets for P2/RF evaluation.
- P2 retained as a pre-model independent holdout with deterministic 4/5/6-year projections.
- Full California annual RF training only after a measured canary and an all-in upper bound no greater than USD 100.00.
- An append-only post-freeze scene gallery that cannot become truth or tuning data.

**Non-Goals:**

- `window_year` does not change the temporal-majority smoother; it is minimum sustained noncrop duration.
- `us_abandon_clean_{feature}.csv` is not represented as the RF training table.
- The RF is not converted into a direct-abandonment classifier and is not retrained per duration.
- This change does not authorize CONUS Earth Engine/RF/paid cloud execution, CCDC/LandTrendr, classifier Asset creation, silent retries, or spending above USD 100.00; its CONUS authority is local-product-only.
- A contract/test pass, task success, or in-memory result is not evidence that retained products exist.

## Decisions

### 1. One public local API owns the scientific semantics

```python
def run_abandonment_detection(
    target_nc: str | Path = Path(r"D:\xarray\reclass_lccs_1km.nc"),
    window_year: int = 5,
    start_year: int = 1992,
    current_end_year: int = 2020,
    init_cropland: int = 2,
    extend_validation: bool = True,
) -> dict[str, object]:
    ...
```

The source must contain one unambiguous annual land-cover variable with unique, consecutive integer years. `window_year` and `init_cropland` are positive integers. Crop is class 1, built is class 7, and NoData/nonfinite/unmapped observations are invalid rather than noncrop.

`init_cropland=n` controls both the first `n` observations beginning at `start_year` and the consecutive crop prefix immediately before a candidate exit. The detector searches only through `current_end_year`, selects the latest valid event whose minimum noncrop window completes by that year, and requires continuous valid noncrop/nonbuilt status through the cutoff. `abandonment_duration` is inclusive through the cutoff.

With `extend_validation=True`, exactly years `current_end_year+1` and `+2` must both be valid noncrop/nonbuilt. They only set persistence inclusion; they cannot create or complete an event, move its start, extend its cutoff duration, or change its 2020 label. Missing extension years fail closed. `False` prevents post-cutoff reads. This is an intentional semantic correction, not byte equivalence with the legacy full-axis regex.

### 2. Computation stays in existing modules and the Notebook remains thin

`function/cropland_abandonment.py` owns detection and reuses its existing chunk writers, atomic replacement, encoding, QA, and manifest helpers. The implementation opens lazily, slices required years once, chunks spatially, avoids materializing the full source time cube, closes datasets promptly, and resumes only on complete fingerprint equality.

`tools/update_process_notebook_contract.py` remains generator authority. The final active `Process.ipynb` cell only imports the public function, declares the six defaults, calls it, and inspects the receipt. Commented legacy cells remain unchanged. Direct Python, runner, generator, and Notebook calls must share parameter and product fingerprints.

### 3. Chunk footprint, CONUS prediction scope, and California consumption are closed separately

Detector and merged NetCDF chunks retain the exact expected inventory derived from the fingerprinted target source and its declared eligibility mask; they are not redefined as state-clipped rectangular files. Prediction publication then applies `usa_bounds_main={lon_min:-125, lon_max:-65, lat_min:25, lat_max:49}` to candidate point centers. It reads `data/cb_2018_us_state_500k.shp`, transforms its source CRS to EPSG:4326, excludes `STATEFP` values `02,15,60,66,69,72,78`, and performs an inner point-to-state `sjoin(..., predicate="within")`. This retains the contiguous 48 states plus DC; the bbox is prefilter-only and every retained CSV row is authorized by its point center lying strictly within one retained state polygon. The manifest binds the source/mask chunk inventory separately from every shapefile component, source/target CRS transform, bounds, exclusion set, predicate, and resulting sorted state inventory.

California P1/P2, RF training/validation, holdout evaluation, and the post-freeze gallery remain limited to the exact `STATEFP=06` subset reconstructed from the accepted CONUS CSV by the same frozen spatial join. The national CSV is never redefined as a California CSV, and no CONUS cloud/RF execution is authorized.

| Feature | Input | Grid | Minimum years |
|---|---|---|---:|
| `300m_min4` | `D:\xarray\reclass_lccs_300m_esa_cci.nc` | native 300 m | 4 |
| `300m_min6` | same | native 300 m | 6 |
| `1km_mode_min4` | `D:\xarray\reclass_lccs_1km.nc` | categorical mode 1 km | 4 |
| `1km_mode_min5` | same | categorical mode 1 km baseline | 5 |
| `1km_mode_min6` | same | categorical mode 1 km | 6 |

All use 1992, 2020, initialization 2, and the fixed two-year extension. The main CSV preserves the established prediction schema. Acceptance deterministically rebuilds the companion row-key-to-`state_fips`/`state_code`/`state_name` frame returned by `build_prediction_embedding` and binds its hash and state counts in the manifest. Native 300 m rows remain native and bind `feature_grid_row`, `feature_grid_col`, `feature_match_distance`, and `pixel_area` when mapped to the 1 km `Feature_all` master grid. Shared master cells cannot cross spatial folds.

### 4. Local success requires actual retained artifacts

For every feature the authoritative families are:

```text
data/us_abandon_clean_{feature}.csv
D:\xarray\abandon_2_{feature}\chunk_{row}_{col}.nc
D:\xarray\merged_chunk_2_{feature}\chunk_{row}_{col}.nc
```

`chunk`/`merged_chunk_2` are normative spellings. Existing unfeatured, historical, P2, and prior feature roots are immutable. `us_abandon_clean_{feature}.csv` is the complete CONUS 2020 candidate/prediction table under the frozen bbox/state-polygon rule, not RF training data.

Each feature also retains `data/us_abandon_clean_{feature}.manifest.json` and `outputs/s0_us_abandon/{feature}/local_run_abandonment_detection_acceptance.json`. The latter is written only after reopening all products and validating parameters, source/AOI/code hashes, exact chunk keys, bytes/SHA-256, CSV and NetCDF schemas, cutoff/extension assertions, QA, and the function return receipt. A schema-valid zero-event result must still contain the complete chunk inventory and an explicit reason; missing products never mean zero events.

### 5. P2 is frozen before predictions and remains the sole independent holdout

P1 freezes 36 core records and a separate eight-record legacy appendix using only evidence no later than 2020 and no local/RF prediction. All 44 receive primary review; a deterministic 12-record subset receives blind secondary review. The 36 core records are the only primary metric denominator, the eight legacy records are boundary audit only, and the secondary subset is reviewer-QA only.

Before review or prediction reveal, `decision_contract_family_sha256` binds the five features, source/grid identities, the exact CONUS bbox/exclusion/polygon-within rule, deterministic `state_fips=06` extraction, cutoff/extension/class/NoData semantics, inclusive duration formulas, RF model/training identities, probability threshold, and min4/min5/min6 projections. P2 freezes rule-neutral prior-crop, exit-interval, continuous-noncrop, recultivation, evidence-quality, and adjudication fields. For `w` in `{4,5,6}`, a point qualifies only when prior crop and continuous noncrop are confirmed, recultivation is false, and `2020-exit_start_year_latest+1 >= w`. Intervals crossing 2015/2016, 2016/2017, or 2017/2018 boundaries require adjudication.

P2 labels, metrics, and gallery observations cannot influence training, threshold, duration, resolution, feature, candidate, or replacement selection. Post-2020 evidence cannot modify P2.

### 6. The RF remains one annual crop/noncrop family

The graph retains binary crop/noncrop training, the accepted CDL/stable-zone bundles, fixed predictor order, deterministic seeds, 300 trees, `smileRandomForest`, `MULTIPROBABILITY`, and crop probability at class index 1. `direct_abandonment_rf=false`; P2/reference/gallery fields are forbidden from training.

The full manifest enumerates every state-intersecting model-unit/year, training and source fingerprints, task/output identity, and permitted attempt. Min4/min5/min6 share the same RF model-family and annual `p_crop` hashes and differ only by temporal projection.

### 7. A measured canary precedes conditional full training

The minimal canary uses the production graph and the smallest declared California model-unit/year that can measure real training/inference EECU. Before submission it requires valid P2, all five accepted source/mask chunk inventories and complete-CONUS CSVs plus their deterministic California subset identities, exact deployment/Billing/credit/quota/Asset/bucket identity, a no-submit dry-run, an unused attempt prefix, and one-task authorization. Acceptance requires terminal OperationMetadata/EECU plus complete generation/CRC32C/bytes/local-SHA delivery. It remains `canary_only=true` and creates no classifier Asset.

### 8. USD 100 is an all-in hard cap, not an estimate target

Within 60 minutes before full submission, the gate computes:

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

Every term is finite, current, hash-bound, and includes successful and failed attempts. Verified funds cover the total plus the declared reserve. Only `safe_total_upper_usd <= 100.00` authorizes the exact full manifest. Missing/stale price, EECU, balance, capacity, quota, inventory, storage, egress, or delivery evidence blocks; unknown is never zero and work cannot be split to evade the cap.

Full training uses `max_inflight=1` and no automatic retry. Before every unit, actual spend plus worst-case remaining work is recomputed. A retry needs a new prefix, fresh admission, and proof the full remaining branch still fits. The two truthful terminal branches are `RF_CALIFORNIA_TRAINING_ACCEPTED` and `RF_CALIFORNIA_TRAINING_BLOCKED_BY_BUDGET_OR_ADMISSION`; the blocked branch creates zero new full-run operations and cannot claim training completion.

### 9. Delivery is immutable and product/scientific states remain separate

Every cloud unit binds run/stage/unit/task-spec/attempt/operation identity and an unused GCS prefix. Every returned object binds generation, CRC32C, bytes, local SHA-256, and an atomic D-drive destination. Missing, unexpected, mismatched, or pre-existing different bytes block without overwrite or deletion.

Local matrix acceptance, P2 acceptance, RF canary, RF full training, holdout evaluation, gallery completion, and scientific acceptance are distinct. Compute completion alone does not prove accuracy, prevalence, area, or PV suitability.

### 10. The gallery is a post-freeze descriptive derivative

The gallery lives in a new append-only `post_freeze/{gallery_id}/` sibling, never inside P2 `accepted/`. It preferably covers all 36 core records and keeps eight legacy records in a labeled appendix; a subset must have a predeclared selection hash and `nonrepresentative=true`.

Each scene binds P2/family/sample/model/variant/media hashes, frozen projections, true NDVI, McFeeters NDWI `(Green-NIR)/(Green+NIR)`, separately named NDMI when present, annual QA/source identities, NAIP acquisition metadata, and official survey raw/decoded evidence. Training, tuning, metric, area, PV-mask, label-change, and scientific-acceptance flags are false. Without actual field observations it is a scene-evidence gallery, not field validation.

### 11. Loop v3 authority is fingerprint-latched, zpy-direct, and ref-local

R5--R12 are the only active refs. Loop retention remains `thin`; ledger and scratch remain `test_cache/ee-california/...`; product, bundle, and GUI roots are `null`. Exact repo, D-drive, P2, gallery, and RF outputs are declared in specs/tasks. A matching active-registry fingerprint with no pending irreversible policy is sufficient for ordinary local Apply. Supervisor-direct work uses local role `zpy`, `agent=null`, synchronous execution, and an immediate supervisor join; it creates no dispatch envelope. Explicit `rose` remains predecessor/bootstrap provenance only.

Each active ref retains `max_apply_attempts=3` and dormant `max_unblock_runs=1`. These allowances are ref-local and are not replenished by narrative refresh, semantic restamp, a new run id, or Unblock. Deleted revision/change/hard-ceiling iteration counts are migration residue and SHALL NOT reappear as Apply authority; active-minute, breaker, and revision-count diagnostics remain finite. `sealed`, `confirmed_at`, headcount, and optional hard-ceiling fields are compatibility diagnostics, not ordinary start-work gates.

Loop autonomy never authorizes GEE exports, GCS writes, paid submissions, credentials, product `--commit`, Git push, PR operations, or writes to `master`. Those effects continue to require the separate Landuse gates and explicit user authority. Fingerprint or narrative refresh creates none of the declared products or external operations.

## Risks / Trade-offs

- Correct cutoff behavior intentionally differs from legacy full-axis output; golden cases must preserve the known predicates while proving the correction.
- Five complete detector/merged inventories and CONUS CSVs are IO-heavy; spatial streaming, atomic writes, exact resume fingerprints, and physical-output validation are mandatory.
- P2 is diagnostic, not prevalence-representative; report 36-core numerators/denominators and keep legacy separate.
- A canary may underpredict statewide complexity; use conservative measured distributions, safety factor, exact inventory, and all failed attempts.
- If the USD 100 upper bound cannot be proven, full RF training remains blocked without weakening the goal into an invented result.
- Scene selection can encourage post-hoc stories; prefer the full core panel or freeze and label a nonrepresentative subset.

## Migration Plan

1. Preserve the prior fingerprint, commented cells, P2 attempts, and unfeatured roots.
2. Implement semantic golden tests and the public function in the existing module.
3. Route runner/generator/notebook through that function.
4. Freeze P1, the decision family, and P2 before revealing predictions.
5. Run a local canary and materialize/verify all five complete source/mask chunk inventories and complete-CONUS CSVs, then freeze their deterministic California subset identities.
6. Run the admitted RF canary, calculate the all-in upper bound, and execute full training only if it is at most USD 100.00.
7. Evaluate available frozen outputs, build the post-freeze gallery, and run whole-change verification.

Rollback is fail-closed: retain immutable evidence, stop new work, and return to the last accepted predecessor without deleting or overwriting an attempt.

## Open Questions

None. The API, feature family, cutoff and extension semantics, exact CONUS bbox/exclusion/point-within AOI, California subset authority, product roots, P2 authority, RF identity, conditional USD 100 authorization, thin retention, zpy-direct Loop v3 routing, ref-local budgets, tests, and forbidden writes were confirmed through 2026-08-31.
