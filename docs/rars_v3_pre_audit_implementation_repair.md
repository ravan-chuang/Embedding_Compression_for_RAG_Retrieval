# RARS-v3 Pre-Audit Implementation Repair

The first notebook execution used implementation commit
`00cc09426ce98fd60d0d800c80fab0c8fc890f03`. Its design phase completed the
registered calculations but failed while constructing `design_freeze.json`:
the evaluator referenced the expected audit-label manifest path before binding
that local path variable.

This repair adds only that missing path binding and a regression test. It does
not change the frozen protocol, query roles, candidate arrays, progressive
representation, registered comparators, byte budgets, metrics, thresholds, or
formal decisions. The failure occurred in the design phase; the audit phase
was never entered and audit labels were not materialized by the evaluator.

The partial output directory keyed by the old implementation commit is an
invalid execution record and must not be reused or overwritten. The repaired
notebook keys all local and durable outputs by the repaired implementation
commit, producing a separate clean run while preserving the failed attempt.
