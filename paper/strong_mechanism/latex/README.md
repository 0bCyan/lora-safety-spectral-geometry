# Paper A LaTeX

This directory contains a complete, compilable, evidence-gated manuscript. It is intentionally venue-neutral; replace the document class only after the target venue is selected.

Build:

```bash
make -C paper/strong_mechanism/latex all
```

The build writes `latex/build/main.pdf`. Missing generated tables or figures appear as explicit framed placeholders. A release build is not allowed to contain any primary-result placeholder.

Current manuscript mode is `working draft`, not submission ready. Numeric results from new experiments must enter through generated table fragments or reviewed source edits tied to the evidence ledger.
