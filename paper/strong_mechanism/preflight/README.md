# M1 Formal Preflight

No formal Paper A training run may begin until an immutable `M1_preflight_lock.json` is created and committed.

Required sequence on the 4070 Ti machine:

```bash
make -C paper/strong_mechanism smoke
python paper/strong_mechanism/scripts/prepare_all_runtime_configs.py --offline
python paper/strong_mechanism/scripts/execute_dry_runs.py
# perform the separately named one-step smoke job and complete smoke_report.json
python paper/strong_mechanism/scripts/freeze_preflight.py \
  --smoke-report paper/strong_mechanism/data/runs/preflight/smoke_report.json
```

The lock records:

- repository commit and clean-worktree state;
- model repository and exact revision;
- environment-lock hash;
- 15 training commands and 45 evaluation/audit commands;
- all machine-local runtime-config hashes;
- all 60 dry-run outcomes;
- smoke-job result;
- claim, protocol, registry, config, and core-script hashes.

After the lock is committed, formal commands are run in the frozen order. The lock is never edited in place. A genuine protocol defect requires a versioned amendment and symmetric treatment of every affected arm.
