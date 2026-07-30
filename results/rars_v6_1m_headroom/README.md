# RARS-v6 1M headroom outcome ledger

`reported_outcome.json` records the stdout excerpt returned after the successful
Colab run at source commit `26a7717...`. It is deliberately **not** labelled a
canonical closure packet because the original `headroom_complete.json` and
registered `.npy` files are still on Google Drive rather than committed here.

Before V7 training, verify the durable directory:

```bash
python scripts/verify_rars_v6_1m_headroom_packet.py \
  --packet-root /path/to/rars-v6-1m-headroom/26a7717b964e
```

The verifier hashes every registered output, recomputes the recall means and
Recall@100 decomposition, and reproduces the formal V6 gate. V7 refuses to
train unless this verification passes.
