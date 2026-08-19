# FOACPO P0 Runtime Retry Record

The first P0 `run` invocation stopped inside `require_clean_start` before model
loading. The gate incorrectly required the local large checkpoint adapter and
its config to be Git-tracked, although these existing machine artifacts are
frozen by SHA256 and intentionally not tracked. No model forward, gradient,
optimizer mutation, output artifact, or parameter change occurred.

The retry changes only the tracked-file assertion: the two checkpoint files
remain mandatory and hash-verified, but are exempt from `git ls-files`.
Clean-worktree enforcement, every input hash, all data and selections, alpha
registry, gates, output rules, and the zero-step boundary remain unchanged.
