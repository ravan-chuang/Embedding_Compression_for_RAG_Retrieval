# RARS vs PCA Comparator Freeze

This directory records the validation-frozen PCA comparator under
`rars_pca_comparator_v1`.

## Selected PCA configuration

- Method: unweighted residual PCA
- Rank: 16
- Coefficients: int8
- Alpha: 0.75
- Top-B: 40
- Candidate pool: Top-100
- Final cutoff: Top-10
- Base index: IVF-PQ M32, nlist=512, nprobe=16

## Validation proxy result

- Corrected Top-10 overlap: 0.5700
- Overlap gain: +0.0298
- Maximum overlap gain: +0.0306
- Retained fraction of maximum gain: approximately 97.4%
- MSE reduction: 7.82%

The selected Top40 setting satisfies the preregistered rule requiring the
smallest correction depth retaining at least 90% of the maximum validation
overlap gain.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| PCA basis | `ffad7e9a65d87045edef7e8d96e5fb90f2a5cc2e213038db72392e65c9ed8fec` |
| PCA sample rows | `8a10e1f5abf4f62f3e8add94d7b0ee2b3449a6612594a3244aa11518705c2c07` |
| PCA scales | `7f2efa852dc11b6079e1d23e153de00e45119e4d3cfba40c97be0c72e3fc3bf6` |
| PCA int8 codes | `20bc05d1e5e645e85ca23ff07a420da8e732a311ecb2a2969025b21cde1ad41f` |
| Selected config | `019170f80a085e1d3c57311b7fb504450351312be4ea7bb41e7f8c1de60f44d4` |
| Validation table | `1c591ae216d1ea712f2b91b7196106c83511be5c451f2b64705acca4873a477a` |

The 16 MB document-code memmap is not committed. Its shape, dtype, byte size,
generation script, and SHA-256 are recorded in
`sidecars/codes_pca_rank16.metadata.json`.

No current held-out test result, 863-query sensitivity subset, or external qrels
were used for PCA selection.
