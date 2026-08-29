# Evidence matrix

Use for literature/method exploration, design method selection, and unblock
hypothesis grading. Every cited source needs a stable identifier (DOI, arXiv
ID, PMID, URL, or local path). Never invent titles, authors, years, sample
sizes, or citations. Abstract-only reads must be labeled and down-weighted.

## Matrix fields

| Field | Content |
|---|---|
| question | claim or review question under test |
| method | method / model / pipeline family |
| data | dataset, geography, period |
| baseline | comparison point |
| metric | measurable outcome |
| finding | stated result in own words |
| limitation | known limits |
| venue / source | identifier + path |
| relevance | why it matters to this change |

## Strength grades

| Grade | Meaning |
|---|---|
| `replicated` | Independent sources converge |
| `single-study` | One credible primary source |
| `conflicting` | Sources disagree materially |
| `weak-indirect` | Indirect or secondary signal |
| `not-verified` | Claimed but not checked |

## Compact output shape

```text
STATUS: SYNTHESIZED | PARTIAL | BLOCKED
REVIEW QUESTION:
EVIDENCE LIMITS:
SEARCH STRATEGY:
EVIDENCE MATRIX:
SYNTHESIS: Themes / Agreements / Disagreements / Gaps
IMPLICATIONS:
UNVERIFIED:
```

Unblock hypotheses should tag supporting and contradicting evidence with one of
the five strength grades above. Low-confidence grades cannot authorize
destructive action.
