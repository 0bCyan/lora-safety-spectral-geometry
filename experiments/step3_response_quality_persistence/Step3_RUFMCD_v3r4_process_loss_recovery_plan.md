# Step3 RUF-MCD v3r4: Runtime Process-Loss Recovery

## Registered incident

The frozen v3r3 execution completed all 12 fit-unit gradient measurements and
entered the 96-facet CPU constraint-generation solver. After more than 50
minutes, the execution session no longer existed. A subsequent host check
found no Python process, no v3r3 `run/` directory, and a clean worktree.

No exception or solver result was captured. The event is therefore classified
as an infrastructure process loss before artifact write, not as geometric
infeasibility, certificate failure, or a method result.

## Frozen recovery

v3r4 permits exactly one complete restart from fit row zero in a new output
directory. It reuses the exact v3r3 config and the complete 24-round bound.
The runner change admits only the registered process-loss status and applies
the existing 24-round structural validation to it.

The recovery launch uses one process and repository-external stdout/stderr
logs. It must not be restarted while that process is alive. This changes only
runtime observability and process persistence.

The following remain unchanged:

- all 96 affine facets and seven certificate groups;
- the response covector and leave-one-out reflection definitions;
- the fixed angle, tangent budget, target floor, and target gain;
- the 24-round constraint-generation completeness bound;
- the active KKT, active gap, and full gap thresholds of `1e-7`;
- the fit, calibration, and diagnostic rows and all statistical gates;
- the zero-optimizer, no-training, no-generation boundary.

## Stop rule

If the single v3r4 process disappears before writing a complete frozen result,
stop as infrastructure-inconclusive. Do not start a fifth execution without a
new prospective record. A numerical stop emitted by the runner is retained as
the method result and must not be manually retried.

## Claim boundary

The fit rows have now been opened four times and the calibration/diagnostic
rows three times. Any v3r4 result is exploratory recovery evidence, not fresh
confirmation. A valid certificate applies only to the registered finite
96-facet polytope; population coverage remains unclaimed.

No optimizer, scheduler, checkpoint, candidate generation, Core100 access,
heldout access, or model training is authorized.
