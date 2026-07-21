# RARS-v7 pre-outcome Faiss DirectMap repair

## Failure boundary

The first V7 invocation at implementation commit `303c31b7...` stopped after
writing only `split_manifest.json`. It did not write `training_started.json`,
execute an epoch, update an adapter, or expose a selection metric.

The captured traceback was:

```text
RuntimeError: ... DirectMap::get ... direct map not initialized
```

The failure occurred while recomputing the already-verified V6 candidate-union
PQ scores, before pair support was emitted and before the adapter existed.

## Cause

Faiss `IndexIVFPQ.reconstruct_batch(row_ids)` requires an in-memory mapping
from corpus row IDs to inverted-list locations. The frozen serialized index
does not include that optional `DirectMap`. The verified V6 evaluator builds
one immediately before reconstruction, but the initial V7 trainer omitted the
equivalent call.

## Repair

V7 now calls:

```python
ivf.make_direct_map()
```

after validating the inverted lists and reproducing the original-query probe
IDs, and before any reconstruction. This creates a RAM-only lookup on the
already loaded index object. The code never serializes that object. The
registered index file is still SHA-256 checked before and after the run, so a
byte-level mutation remains a fatal error.

No protocol threshold, split, loss, pair definition, adapter parameter,
selection rule, or gate changed. The failed `303c31b7...` output directory is
retained as a pre-outcome failure artifact; the repaired implementation must
use a new commit-bound output directory.
