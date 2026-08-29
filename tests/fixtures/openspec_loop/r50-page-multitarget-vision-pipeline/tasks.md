# Active Task Registry

- [x] 2.1 Complete the Wenzhou remaining-manifest run [#R64]
  - NO_DEP: first active task in this fixture.
  - ACCEPT: The accepted manifest is complete and source-backed.
  - TEST: SCOPE: CLI; Run: python -m pytest tests/test_r64.py -q

- [ ] 2.2 Close Wenzhou at the strict all-years gate [#R65]
  - DEPENDS_ON: R64
  - ACCEPT: Every year reaches the declared evidence and compatibility counts.
  - TEST: SCOPE: CLI; Run: python -m pytest tests/test_r65.py -q

- [ ] 3.1 Atomically replace accepted Wenzhou product scope [#R66]
  - DEPENDS_ON: R65
  - ACCEPT: Exactly the accepted city scope is replaced and non-target rows remain unchanged.
  - TEST: SCOPE: CLI; Run: python -m pytest tests/test_r66.py -q

## Historical Task Ledger (Rollback-Only; Non-Operative)

- [ ] 5.1 Historical Wenzhou completion wording [#R36]
  - ACCEPT: Historical text is retained for review only.
  - TEST: SCOPE: CLI; Run: python -m pytest tests/test_historical.py -q
