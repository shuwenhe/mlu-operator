# Phase 13.4 — Distributed MLA numerical evidence

The first architecture replay used the earlier rejected `native-dump-yarn` inputs. It failed at layer 1 even though diagnostic full-head and two-head native execution agreed. Corrected archived runs use the accepted `native-dump-yarn-scale` inputs and compare against the unmodified historical targets. All 27 layer gates passed on eight ranks. The standalone full generation replay produced the frozen greedy IDs and passed its logits and hidden-state checks for four prompt lengths and three decode steps.

The exporter reads the corrected closure logs. It verified 43,448 accuracy records and identical hashes across eight ranks for 880 replicated full-model O, hidden-state and logits boundaries. The historical failure log remains intact. Production `LLM.generate()` at TP=8 is NOT RUN.

[Corrected closure](PHASE13_4_FINAL_EVIDENCE_CLOSURE.md) · [Per-rank CSV](../artifacts/logs/phase13_4/accuracy.csv)
