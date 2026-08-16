# CSPR P1 Source Acquisition Stop Report

Route: `STOP_DATA_SOURCE_UNAVAILABLE`.

The prospective P1 protocol was committed and pushed at `3767fca` while the registered raw source was absent. Its implementation passed all seven synthetic tests and all frozen artifact/reference hashes matched.

After that freeze, three independent transport paths were attempted: the frozen Hugging Face URL through the experiment machine's existing local HTTP proxy, the same revision and path through a public mirror, and the frozen URL without a proxy. Each path ended in a TLS connection reset before response-body transfer. The mirror only redirected to the frozen Hugging Face endpoint. No raw or partial file remains, no source row or candidate response was opened, and no P1 source-feasibility statistic was computed.

This is a source-availability stop, not a failed data-sufficiency result and not evidence against CSPR. The P1 protocol must remain unchanged. Work may resume only after the exact registered file is placed at the ignored raw path and its SHA256 equals `2e24b85dd599224eeb7379da297223bad50e11fea9fee6a7fdb4bd26aedec315`. Reconstructing the gzip from another representation, reusing the old 180-row packet, changing the revision, or substituting heuristic labels is not allowed.

No model gradient was computed. `optimizer_steps=0`, `training_authorized=false`, `core100_opened=false`, and `heldout_content_opened=false`.
