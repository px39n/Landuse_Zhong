## ADDED Requirements

### Requirement: P1 is a cutoff-safe pre-model evidence lane
P1 SHALL nominate and freeze review candidates without using any local robustness prediction, Landsat-derived model feature or trend, `p_crop`, RF result, Xie field, CCDC/LandTrendr result, post-2020 observation, or expected final label. It SHALL use only source evidence observed no later than 2020 and SHALL create zero paid Earth Engine operations, zero GCS writes, and zero Assets.

P1 SHALL bind final DWR releases and dynamically decoded numbered slots, exact CDL 2020, native-grid ESA-CCI 2020, FMMP 2020 context, and approved NAIP acquisitions no later than 2020. Every source SHALL record provider, dataset/release, role, observation/acquisition time, URI, retrieval time, bytes, SHA-256, CRS, resolution, and source-specific alignment. A single DWR, CDL, ESA-CCI, FMMP, NAIP, fallow, `other`, missing, or model signal SHALL NOT create authoritative abandonment truth.

#### Scenario: P1 builds a valid packet
- **WHEN** P1 ranks windows, selects candidates, or builds blind and reconciliation evidence
- **THEN** all ranking and visible evidence SHALL be no later than 2020, model-independent, source-fingerprinted, and free of authoritative labels

#### Scenario: A prediction or post-2020 observation leaks into P1
- **WHEN** a candidate score, packet, HTML page, CSV template, or selection decision contains a local feature result, RF result, Landsat model output, or 2021+ evidence
- **THEN** P1 SHALL reject the attempt and require a new no-overwrite packet and sample-key freeze

### Requirement: P1 freezes 36 core records and eight separate legacy records
The core packet SHALL contain exactly 36 unique grid-aligned records across Fresno (`06019`) and Madera (`06039`) and the three frozen diagnostic window roles. The portfolio SHALL retain 12 potential-abandonment, 9 active-crop-control, 6 recent-exit-boundary, 3 source-disagreement, and 6 QA/unresolved candidate strata. These are diagnostic retrieval strata, not final-label quotas, prevalence weights, or training weights.

P1 SHALL also retain exactly the eight distinct two-county `us_abandon_for_prediction0819.csv` records whose legacy exit begins in 2017 or 2018 and whose retrospective interval covers 2020. They SHALL form `legacy_retrospective_2017_2018`, remain separate from the 36 core records, and SHALL NOT be copied as truth. Their post-2020 duration is provenance of legacy hindsight only.

Every sample key SHALL bind packet kind, source-version hashes, window, EPSG:5070 grid row/column, and candidate stratum. Every prior v1 and attempt-8 key remains immutable `historical_audit_only` and cannot alias or replace a new key.

#### Scenario: The candidate inventory is frozen
- **WHEN** P1 emits `P1_READY_FOR_REVIEW`
- **THEN** it SHALL contain 36 core and 8 legacy unique keys, frozen reserve order for replaceable core records, complete evidence hashes, blank RFC 4180 review templates, and no model-facing field

#### Scenario: A legacy record enters the core quota
- **WHEN** an implementation counts, duplicates, replaces, or weights one of the eight source-bound legacy records as a core record
- **THEN** packet validation SHALL fail

### Requirement: The decision-contract family is frozen before any result reveal
Before P2 review or any local/RF prediction reveal, the system SHALL freeze `decision_contract_family_sha256`. It SHALL bind the exact local feature IDs, source/grid identities, confirmed CONUS bounds, excluded FIPS set, state-shapefile identity, inner point-within join semantics, deterministic `state_fips=06` subset rule, `start_year`, `current_end_year`, `init_cropland`, extension semantics, crop/noncrop/built/NoData mapping, inclusive duration formula, RF training/model-family fingerprints, probability definition and threshold, and deterministic min4/min5/min6 projection formulas. Any change SHALL invalidate dependent reviews and evaluations and require a new contract fingerprint and no-overwrite reference attempt.

#### Scenario: The family is frozen before review
- **WHEN** P1 becomes review-ready
- **THEN** the blank packet and P1 acceptance SHALL bind the same decision-family hash while exposing none of its predictions or expected labels to reviewers

#### Scenario: A duration, source, threshold, or model changes after freeze
- **WHEN** any bound decision input differs before evaluation
- **THEN** the existing P2 set SHALL not be used to tune or approve the changed family and dependent acceptance SHALL fail closed

### Requirement: P2 is one independent holdout with blinded reviewer QA
P2 SHALL obtain one primary decision for every 36 core and 8 legacy record, for exactly 44 unique primary reviews. A deterministic 12-record subset, stratified as eight core and four legacy records across counties/windows/risks, SHALL receive blind secondary review by a different annotator. Secondary reviewers SHALL not see primary decisions. Every disagreement, unusable evidence case, unresolved record, or transition interval crossing a duration boundary SHALL be adjudicated.

Only unusable or still-unresolved core records may be replaced, using the next same-window/same-stratum record in the pre-frozen reserve sequence. The eight legacy records are never replaced or dropped; an unresolved legacy record blocks P2. Exact agreement SHALL be at least 11/12 and Cohen's kappa at least 0.70.

The 36 core records SHALL be the only denominator for primary model/variant confusion metrics. The eight legacy records SHALL be reported in a separate boundary table. The 12 blind-secondary records SHALL measure review reliability and SHALL NOT add performance observations.

#### Scenario: P2 completes independently
- **WHEN** 44 primary reviews, 12 blind-secondary reviews, required adjudications, and any permitted core replacements pass
- **THEN** P2 SHALL freeze all actions, reviewer independence, sample identities, evidence hashes, and the exact 36-core holdout denominator

#### Scenario: P2 is used to choose a method
- **WHEN** any P2 label, score, metric, or gallery observation influences RF training, probability threshold, window duration, resolution, feature selection, sample replacement, or candidate selection
- **THEN** P2 acceptance and every derived scientific result SHALL be invalid

### Requirement: P2 freezes rule-neutral transition facts and three deterministic labels
For each accepted record, P2 SHALL freeze at least `prior_cropland_confirmed`, `exit_start_year_earliest`, `exit_start_year_latest`, `noncrop_continuous_through_2020`, `recultivated_by_2020`, evidence quality, adjudication state, and the hashes supporting those facts.

For each `w` in `{4,5,6}`, `qualifies_min{w}_2020` SHALL be true only when prior cropland is confirmed, noncrop is continuous through 2020, recultivation by 2020 is false, and `2020 - exit_start_year_latest + 1 >= w`. Duration bounds SHALL be `duration_min=2020-exit_start_year_latest+1` and `duration_max=2020-exit_start_year_earliest+1`. Intervals crossing 2015/2016, 2016/2017, or 2017/2018 decision boundaries SHALL be adjudicated before acceptance. `manual_category` MAY retain the five-year view for compatibility but SHALL NOT be copied as min4 or min6 truth.

Post-2020 evidence SHALL NOT modify any frozen fact or 2020 projection. Extension persistence is a model-output inclusion rule and SHALL be evaluated separately from the human 2020 reference.

#### Scenario: An exit starts in 2017 and persists through 2020
- **WHEN** prior crop and continuous noncrop are confirmed, there is no recultivation, and the latest exit year is 2017
- **THEN** `qualifies_min4_2020=true`, `qualifies_min5_2020=false`, and `qualifies_min6_2020=false`

#### Scenario: An exit interval crosses a duration boundary
- **WHEN** earliest/latest plausible exit years imply different labels for any of min4/min5/min6
- **THEN** the point SHALL be adjudicated or, for an unusable core point, replaced under the frozen reserve rule before P2 acceptance

#### Scenario: Later evidence changes
- **WHEN** 2021--2024 observations show persistence, recultivation, or construction
- **THEN** no 2020 transition fact, projection, freeze hash, holdout metric, or primary label SHALL change

### Requirement: P2 outputs are immutable, hash-complete, and non-authorizing by themselves
The accepted P2 root SHALL remain the no-overwrite path:

```text
D:\xarray\s2_abandon_detect\s2-ca-validation-20260827-window96-t1-v1\sources\landsat_c2_spectral_30m\intermediate\california_sampling\california_2020_reference_attempt_2\
```

It SHALL retain `accepted/authoritative_labels_2020.csv`, `accepted/s2_california_authoritative_samples_2020_v2.json`, `accepted/legacy_retrospective_boundary_audit_2020.csv`, `accepted/P2_ACCEPTED.json`, the P1 packet/submissions, and delivery manifests. `P2_ACCEPTED` SHALL bind the 36 core and 8 legacy point hashes, 12-record blind subset, adjudications, `decision_contract_family_sha256`, `core_reference_freeze_sha256`, `legacy_boundary_freeze_sha256`, and `combined_reference_freeze_sha256`.

P2 acceptance authorizes reference use only. Paid execution additionally requires the staged-execution budget, identity, canary, and task-manifest gates. Attempt 1, attempt 8, v1 packets, and accepted P2 files are never overwritten.

#### Scenario: P2 artifacts are complete
- **WHEN** every review, transition fact, projection, file inventory, and hash validates
- **THEN** the system MAY emit `P2_ACCEPTED` with `reference_accepted=true` and no claim that local products, RF training, or California scientific evaluation is complete

#### Scenario: A frozen item changes
- **WHEN** a source, coordinate, review action, label projection, family hash, path, bytes, or SHA-256 differs
- **THEN** P2 and every dependent run SHALL fail validation

### Requirement: Holdout evaluation reports frozen variants without tuning
After P2 freeze, the evaluator MAY compute per-feature and available RF confusion counts on the exact 36-core denominator. Each local variant SHALL be taken only from the deterministic `STATEFP=06` subset reconstructed from its accepted complete-CONUS CSV with the frozen spatial join and SHALL bind the parent CONUS manifest, parent CSV SHA-256, membership-frame hash, subset rule, subset row-key inventory, and subset SHA-256. Every result SHALL bind prediction and reference hashes, report numerator and denominator for each metric, include Wilson intervals where defined, and declare `diagnostic_holdout=true`, `not_statewide_accuracy=true`, and `used_for_tuning=false`. The eight legacy records SHALL have a separate boundary table and SHALL never enter the main confusion matrix.

#### Scenario: Frozen variants are evaluated
- **WHEN** a local feature or accepted RF output is joined to P2
- **THEN** a local feature SHALL first reproduce the accepted parent-CONUS-to-`state_fips=06` subset identity, and the final join SHALL use frozen sample identity and family hashes and report the exact 36-core denominator without changing any method choice

#### Scenario: An unavailable RF full run is evaluated
- **WHEN** RF full training is budget-blocked or lacks accepted retained outputs
- **THEN** the report SHALL state `rf_result_available=false` and SHALL NOT synthesize, extrapolate, or reuse another run's score

### Requirement: The post-freeze scene gallery is append-only and descriptive
The system MAY build `post_freeze/{gallery_id}/` only after valid P2 and decision-family freezes. The gallery SHOULD cover all 36 core records; the eight legacy records SHALL be a separately labeled appendix. If fewer core records are shown, the selection policy and selection SHA-256 SHALL be frozen before rendering and every page SHALL declare `nonrepresentative=true`.

The gallery SHALL bind `P2_ACCEPTED`, `combined_reference_freeze_sha256`, `decision_contract_family_sha256`, sample-key inventory, local/RF variant manifests, and every media hash. Each scene SHALL include frozen reference projections, prediction identities, exact coordinates/grid identity, annual observation and QA metadata, NDVI, true McFeeters NDWI `(Green-NIR)/(Green+NIR)`, separately named NDMI `(NIR-SWIR1)/(NIR+SWIR1)` when retained, NAIP acquisition identifiers/geometry/hash, official survey provider/release/year/raw slots/decoded semantics, deterministic display role, and a qualitative caption.

The gallery SHALL set `qualitative_only=true`, `training_eligible=false`, `threshold_selection_eligible=false`, `included_in_holdout_metrics=false`, `included_in_area_estimate=false`, `included_in_pv_mask=false`, `may_change_reference_label=false`, and `scientific_acceptance=false`. It SHALL create zero new GEE operations unless a separately admitted retained source task already exists; missing source evidence blocks the scene rather than triggering an export.

#### Scenario: A valid gallery is rendered
- **WHEN** all source, sample, model, variant, and media fingerprints match the frozen authorities
- **THEN** the gallery SHALL be written to a new append-only root and P2 bytes and hashes SHALL remain unchanged

#### Scenario: NDMI is labeled as NDWI or gallery evidence is reused
- **WHEN** the serialized formula/name is inconsistent or any gallery field is proposed for training, tuning, primary metrics, area, PV masking, or label revision
- **THEN** gallery validation SHALL fail closed
