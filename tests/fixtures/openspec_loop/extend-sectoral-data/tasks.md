# Active Task Registry

- [x] 1.36 Repair direct-text acceptance and wrong-year fallback [#R36]
  - NO_DEP: baseline task.
  - ACCEPT: Wrong-year fallback is rejected and the accepted baseline is source-backed.
  - TEST: SCOPE: CLI; Run: python -m pytest tests/test_r36.py -q

- [ ] 1.37 Recover image-only cells with visual/OCR grid proof [#R37]
  - DEPENDS_ON: R36
  - STATE: MAXED
  - ACCEPT: Every recovered cell is generated from current visual/OCR evidence.
  - TEST: SCOPE: CLI; Run: python -m pytest tests/test_r37.py -q

- [ ] 1.38 Retry recovery with bounded localization handoff [#R38]
  - DEPENDS_ON: R36
  - SUPERSEDES: R37
  - ACCEPT: The retry uses current source-backed localization and never stale R37 values.
  - TEST: SCOPE: CLI; Run: python -m pytest tests/test_r38.py -q
