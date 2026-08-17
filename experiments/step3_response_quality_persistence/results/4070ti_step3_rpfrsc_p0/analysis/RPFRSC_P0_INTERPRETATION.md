# RP-PFRSC Seed-4 Zero-Step P0 Interpretation

## Decision

The prospectively registered route is
`STOP_RPFRSC_P0_PROTECTED_NONINFERIORITY_FAILED`. Integrity, protected
finite-radius search, diagnostic common transfer, finite-step common descent,
target diagnostic advantage, and materiality all passed. The protected
diagnostic noninferiority conjunction failed, so RP-PFRSC does not authorize a
matched training protocol or any optimizer step.

The failure is localized to the bootstrap lower bound for finite-step TARC
retention. Mean TARC retention was `95.9848%`, and the worst unit retained
`87.0717%`; both clear their registered `95%` mean and `85%` single-unit
thresholds. The 95% bootstrap lower retention bound was `92.3630%`, however,
which does not satisfy the strict requirement to exceed `92.5%`. The miss is
`0.1370` percentage points and remains a failure under the frozen rule.

Independent `verify-run` reproduced the route and all six hashed artifacts.
No optimizer or scheduler was created, and `optimizer_steps=0`.

## Execution audit

An earlier invocation lost its controlling process before any registered run
artifact was written. Before the final invocation, no Python experiment
process remained, the prospective run directory did not exist, the worktree
was clean, and preflight again returned `PASS_RPFRSC_P0_PREFLIGHT` at frozen
HEAD `bee86a4753d06ae15768bec62aea777306a89cfe`. The command, seed, protocol,
and thresholds were unchanged. The completed invocation below is the only
evidentiary run; the interruption supplied no persisted statistic used for
selection.

The final run used seed 4 and eight construction optimizer units at the
unchanged Qwen3-4B LoRA initialization. Each unit evaluated all 127 subset
endpoints, screened exactly 16 endpoint directions, and searched 129
raw-anchored homotopy candidates using 239 exact optimizer-map evaluations.
The fixed eight-category diagnostic set was excluded from construction and
selection.

All 24 arm-by-unit finite-loss evaluations were transient direct parameter
applications followed by exact restoration. Initial and final trainable-state
hashes were identical:
`7410e4511f601b64819c045a6c946dc0e9d7a1a69a1c154bc59b3ce1fc8dbc76`.
Peak reserved memory was `10,215,227,392` bytes, leaving `20.68%` headroom on
the RTX 4070 Ti. Wall time was `1,995.79` seconds. No raw gradient tensor,
checkpoint, candidate response, Core100 row, or heldout row was written or
opened.

## Registered gates

| Gate | Result |
|---|---:|
| integrity | pass |
| protected finite-radius search | pass |
| diagnostic common transfer | pass |
| finite-step common descent | pass |
| protected diagnostic noninferiority | **fail** |
| target diagnostic advantage | pass |
| effect not weak | pass |

The failed protected conjunction determines the stop route. The later
component and mechanism analyses do not change that registered decision.

## Construction certificate

RP-PFRSC retained only candidates satisfying

\[
s_i(d)\ge 0.95s_i(d_{\mathrm{raw}}),
\qquad i\in\{R,U,T,C\},
\]

and maximized the exact E-TARC/branch bottleneck over the finite feasible
family. Since raw MCD was an explicit candidate, the protected family was
nonempty, each selected construction protected score retained at least 95%,
and the selected target bottleneck could not be below raw MCD within the
screened family.

The observed minimum construction retention was `0.953580`. Six units chose
nonzero homotopy fractions and two used the raw anchor:

| Fraction | Blocks |
|---:|---:|
| `0` | 2 |
| `0.125` | 2 |
| `0.25` | 3 |
| `0.50` | 1 |

The nonzero rate was exactly the registered `75%` floor. Construction target
gain had mean `0.010128`, 95% bootstrap interval
`[0.005234, 0.014962]`, and exact two-sided sign-test `p=0.03125` with six
positive units and two ties. The minimum selected target score was `0.073380`,
above the frozen `0.05` floor. Thus the finite-family theorem held and the
protected search gate passed, but its guarantee remained construction-unit
specific as stated in the protocol.

## Independent target transfer

The three optimized objectives transferred positively to the prompt-disjoint
diagnostic set. Their relative finite-loss changes versus raw MCD were:

| Target objective | Mean relative gain | 95% interval | Positive/tied units |
|---|---:|---:|---:|
| E-TARC | `+3.306%` | `[+1.575%, +5.179%]` | `6/2` |
| branch entropy | `+4.483%` | `[+2.222%, +7.035%]` | `6/2` |
| branch negative-gap | `+11.265%` | `[+5.600%, +17.450%]` | `6/2` |

The diagnostic target-bottleneck advantage over raw MCD had mean `0.007105`,
95% interval `[0.003715, 0.010641]`, and sign-test `p=0.03125`. The per-unit
macro relative finite-loss gain had mean `+6.351%`, interval
`[+3.199%, +9.669%]`, and the same sign-test. RP-PFRSC's mean diagnostic
target minimum was `0.075382`, above the frozen `0.05` materiality floor.
These results pass every registered target gate; they do not override the
protected failure.

## Protected noninferiority decomposition

The registered protected gate required, separately for exact score and actual
finite-loss reduction, mean retention at least 95%, bootstrap lower retention
strictly above 92.5%, and every unit at least 85%.

| Objective | Exact mean | Exact CI lower | Exact minimum | Actual mean | Actual CI lower | Actual minimum | Result |
|---|---:|---:|---:|---:|---:|---:|---:|
| risk | `99.407%` | `99.037%` | `98.445%` | `99.449%` | `99.127%` | `98.580%` | pass |
| uniform | `99.397%` | `99.099%` | `98.747%` | `99.472%` | `99.217%` | `98.997%` | pass |
| TARC | `96.865%` | `93.030%` | `87.944%` | `95.985%` | **`92.363%`** | `87.072%` | **fail** |
| continuation | `98.900%` | `98.205%` | `96.939%` | `99.140%` | `98.591%` | `97.621%` | pass |

Only the TARC actual bootstrap lower bound failed. TARC exact retention passed
all three thresholds, including a `93.030%` lower bound. This distinction
rules out a violation of the construction finite-family theorem: the failed
claim is independent-prompt finite-step noninferiority, which that theorem did
not assert.

## Post hoc mechanism audit

The following audit is explanatory and was not a registered gate. On the six
units with nonzero homotopy, mean TARC retention changed in two stages:

| Stage | Mean retention | Change from prior stage |
|---|---:|---:|
| construction exact score | `98.505%` | - |
| diagnostic exact score | `95.819%` | `-2.685` points |
| diagnostic actual reduction | `94.646%` | `-1.173` points |

Diagnostic exact TARC retention was below construction retention on all six
nonzero units. Actual TARC retention was then below diagnostic exact retention
on all six. Each directional count has a post hoc exact two-sided sign-test
`p=0.03125`; the two raw-anchor units are ties. The first gap is prompt
transfer, while the second is finite-step curvature/remainder. Their combined
mean erosion was `3.858` percentage points on active units.

Source composition gives a narrower, descriptive clue. Five selected nonzero
endpoints omitted TARC and all five reduced diagnostic actual TARC relative to
raw MCD. The sole selected endpoint containing TARC retained `105.457%` of raw
diagnostic actual reduction. This is a one-unit subgroup and cannot establish
a causal remedy; it is not grounds for another role scan. It does show why a
construction-only point constraint can be active yet fail to control support
shift.

## Formal implication for a distinct successor

The current constraint has no uncertainty margin. Let

\[
v_i(d)=\frac{\Phi(G_0+0.2d)-\rho\Phi(G_0+0.2d_0)}
{\lVert\Phi(G_0)\rVert_2},
\]

where `d0` is raw MCD, and let `u_i` and `\tilde u_i` be construction and
transferred normalized certificate gradients. If
`||\tilde u_i-u_i||_2 <= epsilon_i`, Cauchy-Schwarz gives

\[
\tilde u_i^Tv_i(d)
\ge u_i^Tv_i(d)-\epsilon_i\lVert v_i(d)\rVert_2.
\]

Therefore the robust margin condition

\[
u_i^Tv_i(d)\ge\epsilon_i\lVert v_i(d)\rVert_2
\]

is sufficient for `\tilde s_i(d) >= rho \tilde s_i(d0)` for every transferred
gradient in that Euclidean uncertainty ball. RP-PFRSC enforced only
`u_i^Tv_i(d) >= 0`; its theorem consequently cannot guarantee transferred
retention when `epsilon_i > 0`. The observed two-stage TARC erosion is
consistent with this exact limitation.

This inequality identifies what a mathematically distinct successor would
need: a prospectively calibrated, construction-only uncertainty radius or a
cross-fitted multi-support robust constraint. The present diagnostic set
cannot be used after the fact to choose `epsilon_i`, and this result does not
authorize such an experiment. Any test would require a fresh split and seed,
an independently committed protocol, and unchanged heldout boundaries.

## Route closure and evidence boundary

Per the registered stop boundary, failure closes the FRSC/PFRSC certificate-
role homotopy line. Another role reassignment, seed retry, threshold change,
or reuse of these diagnostics for selection would be outcome-dependent and is
not admissible. No matched training protocol is authorized.

This is exploratory mechanism evidence on one model and reused disclosed
development data. It supports exact finite-family protection on construction
units and consistent E-TARC/branch improvement, while rejecting pointwise
construction protection as sufficient evidence for prompt-disjoint finite-step
TARC noninferiority. It does not establish generated safety, utility, or
downstream response-quality improvement. `optimizer_objects=0`,
`scheduler_objects=0`, `optimizer_steps=0`, `checkpoint_writes=0`,
`training_authorized=false`, `candidate_outputs_generated=false`,
`core100_opened=false`, and `heldout_opened=false`.
