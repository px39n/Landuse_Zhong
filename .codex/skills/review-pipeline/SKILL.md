---
name: review-pipeline
description: Answer one specialist risk question read-only for an OpenSpec change.
  Never implement, promote, or broaden into Apply.
license: MIT
metadata:
  author: openspec
  version: "3.0"
---

# review-pipeline

One bounded read-only review. Return a single risk answer with evidence refs.
Do not edit code, write the ledger, dispatch a review swarm, or award PASS.
Use the risk question supplied by the user or current task packet. Ask only
when that context cannot identify one bounded review question; pause the
affected review until the answer arrives.
