# HC-RARS Phase 1 integration

This bundle adds the matched-budget `PCA64-RPQ-16B` numerical core and protocol.
It intentionally does **not** run the held-out test or add retrieval-aware codebook
training. Phase 1 first tests whether high-rank product coding preserves the rank-64
capacity advantage under exactly 16 bytes per document.

## Copy into the repository

From the repository root:

```bash
unzip hc_rars_phase1_bundle.zip -d /tmp/hc-rars-phase1
cp -R /tmp/hc-rars-phase1/protocols .
cp -R /tmp/hc-rars-phase1/scripts .
cp -R /tmp/hc-rars-phase1/tests .
cp -R /tmp/hc-rars-phase1/docs .
```

Ensure `scripts/__init__.py` exists. If the repository does not already have it:

```bash
touch scripts/__init__.py
```

Run:

```bash
python -m pytest -q \
  tests/test_hc_rars_phase1_core.py \
  tests/test_hc_rars_phase1_budget_contract.py \
  tests/test_hc_rars_phase1_protocol_contract.py
```

## Suggested branch and commit

```bash
git switch -c feat/hc-rars-phase1-rpq64-16b
git add protocols scripts tests docs
git commit -m "add matched-budget rank-64 residual PQ phase-1 core"
git push -u origin feat/hc-rars-phase1-rpq64-16b
```

## Deliberately excluded from this first patch

- Test-set evaluation driver
- Alpha selection driver
- Artifact serialization format
- Retrieval-aware codebook loss
- Harm constraint or CVaR objective

Those should be added only after the pure numerical core and exact budget contract
pass in the repository environment.
