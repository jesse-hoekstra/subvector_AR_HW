# Calibrated EMW/GKM eigenvalue power bound

This is the method and operating guide for `alfd_eigval.py`. The publication
configuration discussed here is the weak-instrument design selected by
`--version 352515`, with $m_W=3$, $p=4$, $k=7$, $n=250$, and
$\alpha=0.05$.

The implementation combines three ideas that must be kept distinct:

1. EMW supplies the mathematical power upper bound for any null mixture.
2. GKM-style pooled ordinary importance sampling makes searching for a useful
   mixture on a common finite null grid computationally feasible.
3. Fresh direct iid simulation, order-statistic confidence rules, and an exact
   binomial confidence limit certify the reported Monte Carlo upper endpoint.

The pooled importance-sampling calculation is used for discovery and
diagnostics. It is deliberately not substituted into the iid confidence
argument.

## 1. Experiment and density

The Gaussian limit experiment observes

\[
\Xi\sim N_{k\times p}(M,I_k\otimes I_p),\qquad \Omega=M'M,
\]

and the statistic is the vector of ordered eigenvalues
$x_1\ge\cdots\ge x_p$ of $\Xi'\Xi$. For the real noncentral Wishart law,
the part of the eigenvalue density that depends on \(\Omega\) is

\[
\log \widetilde f(x\mid\Omega)=
\log{}_0F_1^{(2)}\!\left(k/2;\Omega/4,X\right)
-\tfrac12\operatorname{tr}\Omega.
\]

The omitted factor depends on the observation $x$, not on $\Omega$, and
cancels in every likelihood ratio. The real-zonal arguments passed to Koev's
routine are therefore `alpha=2`, `c=k/2`, `Omega/4`, and `X`.

For $p=4$, the null nuisance family is the unbounded ordered cone

\[
\Theta_0=\{(\nu_1,\nu_2,\nu_3,0):
\nu_1\ge\nu_2\ge\nu_3\ge0\}.
\]

`asymptotic_ncp_eigenvalues` maps each beta in the paper's DGP to the
full-rank point alternative \(\Omega_1\). At beta zero the fourth eigenvalue is
analytically zero; only that point uses the exact randomized-alpha shortcut.
Nearby beta values are not classified as null using a numerical tolerance.

## 2. Why any calibrated null mixture gives an upper bound

Let $g=f(\cdot\mid\Omega_1)$, and let

\[
h_\lambda(x)=\sum_{j=1}^H\lambda_j f_j(x),\qquad
\lambda_j\ge0,\quad\sum_j\lambda_j=1,
\]

be any probability mixture of null densities. If the likelihood-ratio rule
based on $g/h_\lambda$ is calibrated to size alpha under $h_\lambda$, its
power is at least the power of every test whose size is at most alpha at every
null point. This is the EMW Lemma 1 argument.

Consequences that are important in practice:

- The mixture does not have to be exactly least favorable for validity. A poor
  mixture produces a loose upper bound, not an invalid one.
- Fitting, grid choice, and support compression may be data dependent on a
  training bank, provided the final frozen mixture is calibrated on fresh iid
  mixture draws.
- Convergence and finite-grid diagnostics measure tightness. They are not the
  source of validity of the final EMW endpoint.

The old implementation used the scale left by the last EMW iteration as if it
were a calibrated critical value. A finite empirical EMW iteration can cycle,
so that shortcut could produce a sub-alpha mixture test and a purported upper
curve below valid feasible tests. The current workflow always normalizes the
mixture weights and performs a separate scalar calibration.

## 3. The common 68-point three-dimensional null grid

GKM's published numerical calculation has a one-dimensional null nuisance
parameter and uses 42 support candidates. For $m_W=3$, copying a one-
dimensional line would leave most of the ordered nuisance cone unsearched. The
code therefore uses a deterministic strength-by-shape design.

With the defaults, the common grid contains

\[
H=1+(\texttt{grid-shapes})(\texttt{grid-strengths})+N_{\rm anchors}
  =1+9\times7+4=68
\]

null rows. Its ray design consists of the origin and nine normalized shape
directions, each crossed with seven geometrically spaced largest-eigenvalue
strengths from 0.1 through 100. For the weak configuration, the directions are
the configured
$(35,25,15)$ shape; the three historical stress shapes represented by
$(50,25,5)$, $(15,10,5)$, and $(5,3,1)$; the rank-one, rank-two, and equal
rank-three boundaries; and the low/high third-eigenvalue-share shapes observed
along the complete beta path. The construction normalizes directions and
removes duplicates deterministically.

Geometric strengths generally do not land on the original values. The four
exact configured/historical rows are therefore appended as anchors after the
64-row origin-plus-rays design, giving 68 distinct null rows in total.

There is one ordered 68-row grid for the whole curve. It does not change from
beta to beta. For a particular non-null beta, the full-rank alternative is an
additional density row $g$, not a null support row. Thus a representative
benchmark evaluates 68 null rows plus one alternative row for every sampled
observation.

The reusable helper `common_null_grid_3d` retains its standalone default of six
shapes, or 43 rows, when `n_shapes` and `standard_points` are omitted. The
production CLI intentionally passes the configured stress rows and requests
nine shapes. A direct helper call with no CLI arguments is therefore not the
publication grid.

The grid controls are:

- `--grid-shapes`: number of nuisance-shape directions;
- `--grid-strengths`: number of strengths on each direction;
- `--grid-max-strength`: upper endpoint of the strength range;
- `--max-active-support`: maximum number of fitted mixture rows retained for
  the expensive direct calibration and power phases.

Increasing the first two controls increases shared null-table cost roughly as
$H^2$. Increasing only `grid-max-strength` does not add rows, but stronger
noncentralities can be substantially slower and can reduce importance-sampling
overlap. No finite grid certifies the full unbounded three-dimensional cone.
The old `--validation-grid-size` option has been retired and is rejected.

## 4. Pooled ordinary GKM importance sampling

For each common-grid law $f_s$, the code draws `n_fit` independent
observations. Equal stratum sizes make the proposal

\[
q(x)=\frac1H\sum_{s=1}^H f_s(x).
\]

All $H\,n_{\rm fit}$ observations are pooled, and all $H$ null density rows
are evaluated and cached once. For any rejection function
$\phi(x)\in[0,1]$, the rejection probability under null row $j$ is estimated
as

\[
\widehat{RP}_j=
\frac{1}{Hn_{\rm fit}}
\sum_{i=1}^{Hn_{\rm fit}}
\frac{f_j(X_i)}{q(X_i)}\phi(X_i).
\]

This is ordinary stratified importance sampling, matching the integral in GKM
equation (D.1). The implementation must not divide this expression by

\[
\frac{1}{Hn_{\rm fit}}\sum_i f_j(X_i)/q(X_i).
\]

That division would silently replace the estimator by self-normalized
importance sampling. Consequently, a finite-bank raw mass diagnostic need not
equal exactly one. The saved diagnostics report raw mass, the observed
importance-ratio maximum, and Kish effective sample sizes. Because
$q\ge f_j/H$, the exact ratio is bounded by $H$; a violation indicates an
implementation or numerical error.

The EMW update uses these pooled rejection-probability estimates. Sign changes
shrink coordinate step sizes to prevent an empirical staircase from cycling
forever. `n_iter` reuses the cached density table and adds no density calls.

The training bank is beta invariant. For each beta, only the alternative
density row on that bank must be evaluated again. This is the central GKM
computational reuse; rebuilding all null density rows for every beta would
discard most of the saving.

## 5. Active support and fresh iid certification

After fitting all $H$ weights, `compress_mixture` retains the largest
`max-active-support` weights with stable tie handling and renormalizes them.
The default cap is eight. The artifact records the retained indices, retained
weights, and discarded full-mixture mass for each beta.

Compression cannot invalidate the final EMW upper: the compressed object is
simply another legitimate null mixture, and all final cutoffs are recomputed
after it is frozen. It can make the bound looser. A large discarded mass,
unstable selected rows, or a material change when the cap is increased is a
tightness warning.

For each non-null beta, the complete sequence is:

1. Evaluate the alternative row on the shared training bank.
2. Fit all $H$ EMW weights using pooled ordinary importance sampling.
3. Compress to at most the requested active-support cap.
4. Draw a fresh iid sample directly from the frozen compressed mixture and
   recalibrate its likelihood-ratio cutoff.
5. Evaluate the frozen rule on an independent common-grid audit bank. That
   bank contains fresh direct iid draws within every null stratum and is never
   the training bank.
6. Draw a fresh iid alternative sample for point power and exact binomial
   confidence endpoints.

The training and audit null-density tables are each built only once and reused
across every beta. They have different seeds, roles, and authenticated cache
identifiers. The final mixture-calibration and alternative-power samples are
fresh for every beta.

The weighted pooled bank is used only for fitting and GKM-comparable point
diagnostics. The order-statistic calibration and binomial confidence routines
require direct iid observations; ordinary or self-normalized importance
weights cannot be passed to those arguments and retain the stated guarantee.

## 6. The two saved upper endpoints and the finite-grid lower endpoint

`bounds_point` is the paper-style Monte Carlo power of the independently
calibrated mixture likelihood-ratio rule. `bounds_point_se` is its standard
error from the final alternative sample. It does not include mixture-selection
or critical-value uncertainty.

`bounds` is the primary reported curve. It uses:

- a confidence-liberal order statistic on the fresh iid mixture-calibration
  sample, chosen so the population mixture rejection probability is at least
  alpha except on its allocated event; and
- a one-sided exact Clopper--Pearson upper endpoint for alternative power.

Conditional on correct density evaluations, union bounding these events over
the computed non-null beta values gives the requested simultaneous Monte Carlo
confidence level. It covers the saved beta points only, not every point on a
smooth interpolation through them.

The independent audit bank raises the cutoff until every one of the 68 checked
null rows satisfies the empirical or confidence-conservative size condition.
The resulting powers are finite-grid lower endpoints, and

\[
\epsilon_{\rm grid}=\text{EMW point upper}
                    -\text{finite-grid lower point}.
\]

This follows the tightening logic of GKM D.3.2. It is evidence about sharpness
on the searched grid, not a global size certificate between grid rows or beyond
strength 100. The mathematical EMW upper does not depend on this audit claim.
GKM reused-bank point/grid values are also saved as diagnostics; they are not
substitutes for the independent confidence endpoints.

For $B$ computed non-null beta points and curve confidence $C$, the driver
uses

\[
\delta_{\rm beta}=(1-C)/B,\qquad
\delta_{\rm upper,event}=(1-C)/(2B).
\]

The two primary-upper events are liberal mixture calibration and upper
alternative power. The finite-grid lower uses a separate two-event family,
with the same event allocation, for the common-grid audit and lower alternative
power. Thus the upper curve is simultaneous at level $C$, the lower curve is
separately simultaneous at level $C$, and their joint saved-grid bracket has
the union-bound level at least $\max(0,2C-1)$. This separation avoids loosening
the scientifically primary upper merely to report an optional lower diagnostic.
`n_fit` and `n_iter` receive no confidence allocation because the EMW claim
conditions on the frozen fitted mixture.

## 7. Adaptive matrix-hypergeometric truncation

`M_start` is a lower bound for numerical work, not one fixed truncation for a
configuration or curve. A separate adaptive check is performed for every
density pair $(\Omega,x)$. Therefore selected orders can differ across
observations, null or alternative density rows, phases, and beta values.

For each pair, `mhg_two_matrix_adaptive`:

1. chooses a trace-based starting degree no lower than `M_start`;
2. evaluates the positive coefficient series by total degree;
3. verifies that the peak is behind the final window, that the recent ratios
   contract smoothly, and that a geometric omitted-tail estimate is below
   `--mhg-rtol`;
4. retries only that pair in `--m-step` increments if the criterion fails; and
5. fails closed at `--m-max` or on a suspicious arithmetic coefficient
   collapse.

The practical controls are:

- `--m-start`: performance floor; raising it should not materially change an
  accepted value;
- `--m-step`: retry granularity;
- `--m-max`: hard safety cap, at which nonconvergence is an error rather than a
  silently accepted density;
- `--mhg-rtol`: relative coefficient-tail criterion.

Selected-order histograms, maximum order, estimated remainder ratios, and raw C
evaluation counts are logged. A raw C count above the logical density-pair
count means that some pairs retried. There is no need to rerun a curve at a
sequence of guessed fixed `M_trunc` values. For a new parameter regime, use the
default stringent tolerance and increase `--m-max` only if the run explicitly
reaches it.

The criterion is a carefully tested numerical tail diagnostic for this
positive series, not a symbolic interval proof. The confidence label is
therefore explicitly conditional on density accuracy. Regression tests cover
scalar values against SciPy, strong matrix probes, permutation/scaling
identities, arithmetic failure modes, and one increment beyond selected
orders.

## 8. What the simulation-count controls mean

The production profile currently uses:

| control | production | role |
|---|---:|---|
| `n_fit` | 2,000 per common-grid stratum | mixture discovery/tightness |
| `n_calibration` | 20,000 per non-null beta | fresh iid mixture cutoff |
| `n_validation` | 2,000 per grid stratum | independent finite-grid audit |
| `n_power` | 50,000 per non-null beta | fresh alternative power |
| `n_iter` | 600 | maximum cached-table EMW updates |

The reference profile uses 10,000, 100,000, 10,000, 100,000, and 600,
respectively. GKM specify $N_0=10{,}000$, $N_1=100{,}000$, and 600
iterations for their one-dimensional calculation. The extra independent
mixture calibration and audit separation here are deliberate requirements for
the finite-Monte-Carlo confidence claim, not numerical constants stated by
GKM for the $p=4$ problem.

How to choose the controls:

- `n_fit` controls the noise in ordinary-IS rejection probabilities and hence
  mixture quality. The familiar direct Bernoulli reference at alpha .05 is
  \(\sqrt{.05(.95)/n_{\rm fit}}\), but actual IS precision depends on raw
  masses, importance ratios, and effective sample size. Increase it when the
  fit residual, support, or weights are unstable, or when ESS is poor.
- `n_iter` costs almost no density time after the bank exists. Increase it only
  if residuals are still improving; it cannot cure a noisy or poorly
  overlapping training bank.
- `n_calibration` controls how liberal the confidence-protected mixture cutoff
  must be. Preflight prints the exact rejected rank and its inflation above
  alpha. Increase it if that inflation is scientifically too large.
- `n_power` controls both the point standard error and the one-sided exact
  binomial allowance. Preflight prints representative Clopper--Pearson margins
  near power .05 and .50 plus a distribution-free Hoeffding ceiling.
- `n_validation` controls only the usefulness of the finite-grid lower endpoint
  and epsilon. Its error budget is split over all $H$ rows, so small values
  can make the lower endpoint very conservative. Reducing it does not remove
  the validity of the primary EMW upper.
- `max-active-support` controls expensive per-beta density rows. A smaller cap
  remains valid after fresh calibration but may loosen the result. Use saved
  discarded mass and a prespecified cap sensitivity check to assess this.
- `beta-count` controls resolution, variable runtime, and confidence
  multiplicity. The default is 11 equally spaced points from -2 to 2:
  `-2, -1.6, ..., 0, ..., 1.6, 2`. Ten are non-null. Prefer an odd count when
  the exact beta-zero point must be included.
- `curve-confidence` is simultaneous over those computed non-null points. More
  points or higher confidence makes all protected endpoints more conservative.

There is no universal count that makes an estimate literally certain. Choose
counts prospectively from the preflight ranks and the scientific resolution
required, then retain all fit, IS, compression, audit, and numerical-density
diagnostics in the reported artifact. Do not run many random seeds and report
only the smallest curve without accounting for that selection.

## 9. Exact logical-pair cost and runtime benchmarking

Let $H$ be the common grid size, $A=\min(H,
\texttt{max-active-support})$, and $B$ the number of non-null beta points.
The conservative preflight count is

\[
\begin{aligned}
N_{\rm pairs}={}&H^2n_{\rm fit}+BHn_{\rm fit}
 +(A+1)Bn_{\rm calibration}\\
&+H^2n_{\rm validation}+BHn_{\rm validation}
 +(A+1)Bn_{\rm power}.
\end{aligned}
\]

The two $H^2$ null tables are one-time shared training/audit work. Each beta
adds one alternative density on each shared bank and $A$ active-null plus one
alternative row on its direct calibration and power observations. The current
deterministic top-$A$ compression retains exactly $A$ rows when $A<H$.
Adaptive retries add raw C evaluations but not logical pairs.

For the default 68-row, 11-beta production request, preflight reports 27,516,000
logical pairs over ten non-null betas. The phase counts are 9,248,000 for the
shared training null table, 1,360,000 for beta-specific training alternatives,
1,800,000 for direct mixture calibration, 9,248,000 for the shared audit null
table, 1,360,000 for beta-specific audit alternatives, and 4,500,000 for direct
power. The training bank contains 136,000 pooled observations. Preflight also
reports, for the default 99% confidence, the exact current calibration/audit
ranks and binomial allowances. These values change with counts, beta grid, and
confidence, so use the command output rather than copying constants from this
document. The phase counts describe a fresh run. If a compatible authenticated
training or audit bank already exists, its 9,248,000-pair table is reused rather
than recomputed; the statistical design and artifact signature do not change.

`--benchmark-preflight` measures the actual adaptive C evaluator on the target
machine. Every benchmark observation is evaluated under all 68 common null
rows plus the representative alternative, matching the row width of the
shared-bank phase. Roughly two thirds of its observations span the common null
grid, including boundary and stress sources, and one third come from a
representative alternative. This mirrors the fresh-run phase mix more closely
than an alternative-only benchmark. It uses a private deterministic RNG stream
and writes no artifact. The timing includes worker-pool startup; rare draws can
still be slower than the benchmark sample.

On a many-core machine, use at least two to four benchmark samples per worker
so every process receives steady-state work. The CLI permits up to 256 samples.
For the 48-physical-core Xeon host used for this project, compare 24 and 48
workers with 192 samples; 96 logical threads should be used only if that
measurement actually improves aggregate pairs per second.

The evaluator is CPU C code and currently does not use CUDA. An RTX 4090 or RTX
5000 therefore gives no direct speedup without a separate GPU port. Compare
worker counts on the intended Xeon using measured pairs per second; do not
assume that all logical threads scale linearly.

## 10. Exact commands to run

Run these from the repository root.

```bash
# One-time native build.
sh koev/mhg15/build.sh

# Fast end-to-end scalar plumbing regression; no p=4 artifact is written.
python3 alfd_eigval.py --version 352515 --profile smoke

# Statistical budget and exact logical-pair count; no simulations/artifact.
python3 alfd_eigval.py \
  --version 352515 \
  --profile production \
  --workers 16 \
  --preflight-only

# Real adaptive p=4 timing on this machine; no production artifact or RNG use.
python3 alfd_eigval.py \
  --version 352515 \
  --profile production \
  --workers 16 \
  --benchmark-preflight \
  --benchmark-samples 64

# Start or resume the exact production request after accepting both reports.
python3 alfd_eigval.py \
  --version 352515 \
  --profile production \
  --workers 16 \
  --acknowledge-expensive
```

Use exactly the same statistical, grid, adaptive-M, seed, and worker-independent
method flags in preflight, benchmark, and production. Worker count may be tuned
from benchmarks without changing the statistical estimand. Do not add `--force`
to a normal resume: compatible pooled-bank caches and the per-beta checkpoint
are loaded automatically. Use `--force` only when intentionally replacing an
incompatible completed artifact.

An exploratory request can reduce beta resolution and audit precision, but its
confidence scope must be described honestly. For example:

```bash
python3 alfd_eigval.py \
  --version 352515 \
  --profile production \
  --beta-count 5 \
  --curve-confidence .95 \
  --n-fit 1000 \
  --n-calibration 15000 \
  --n-validation 1000 \
  --n-power 30000 \
  --preflight-only
```

Replace `--preflight-only` first by `--benchmark-preflight` and only then by
`--acknowledge-expensive` if the printed statistical allowances and measured
runtime are acceptable. Such a five-point run is not a certified 11-point
curve.

Corrected results and caches live under
`<version>/adaptive/`. The completed result is
`alfd_eigval_<version>.npz`; a compatible `.partial.npz` file resumes completed
beta points after interruption. Authenticated `pooled_training_*.npz` and
`pooled_audit_*.npz` files hold the shared banks. Legacy `M_trunc_*` artifacts
are never loaded.

The finite-sample comparison uses a separate signed cache:

```bash
python3 new_power_comparison.py --version 352515 --preflight-only
python3 new_power_comparison.py --version 352515 --acknowledge-expensive
```

That comparison overlays a finite-$n$, estimated-covariance simulation with a
known-covariance Gaussian-limit EMW curve. Pointwise dominance across those two
different experiments is not a theorem. The same-limit-experiment
smallest-eigenvalue benchmark stored by `alfd_eigval.py` is the relevant
implementation invariant.

## 11. Live local and Weights & Biases monitoring

`watch_power_progress.py` is a read-only sidecar for a long production run. It
does not import W&B into `alfd_eigval.py`, alter a simulation seed, or write a
scientific checkpoint. This separation is intentional: the numerical driver
repeatedly creates large multiprocessing pools, whereas W&B owns network and
background-process state.

Before starting the bound, create or validate the canonical finite-sample DGP
cache. The watcher never launches these simulations implicitly:

```bash
python3 new_power_comparison.py \
  --version 352515 \
  --num-simulations 100000 \
  --workers 48 \
  --seed 20240101 \
  --chunk-size 5000 \
  --acknowledge-expensive
```

Run the bound in one `tmux` pane and the watcher in another:

```bash
# Numerical pane: use the exact settings accepted after preflight/benchmark.
python3 alfd_eigval.py \
  --version 352515 \
  --profile production \
  --workers 48 \
  --beta-count 11 \
  --curve-confidence .95 \
  --n-fit 2000 \
  --n-calibration 50000 \
  --n-validation 2000 \
  --n-power 100000 \
  --n-iter 600 \
  --grid-shapes 9 \
  --grid-strengths 7 \
  --grid-max-strength 100 \
  --max-active-support 8 \
  --acknowledge-expensive

# Monitoring pane (one-time setup: pip install wandb; wandb login).
python3 watch_power_progress.py \
  --version 352515 \
  --wandb-project subvector-ar-hw \
  --wandb-mode online
```

Start the numerical command first, then start the watcher after the driver has
printed its run signature and created the initial partial checkpoint. This is
especially important for an intentional `--force` rerun: starting the watcher
too early could show a still-present completed artifact from the preceding run.

The bound driver writes an authenticated initial partial checkpoint before it
starts the shared null banks, then replaces that file atomically after every
completed beta. The watcher validates the signed DGP cache and checkpoint,
plots all three `simulate_power_dgp` curves, and overlays only finite completed
bound points. It writes an atomic local PNG and a tidy long-format CSV with
every plotted value even when W&B uploading later fails. The authoritative
inputs remain the signed DGP and partial/final bound NPZ files; the CSV is a
human-readable joined view of their different beta grids. The watcher resumes
the W&B run from the statistical run signature; an intentionally new rerun can
instead be given a distinct `--wandb-run-id`.

The W&B path itself can be tested without a DGP cache, checkpoint, or any
matrix-hypergeometric calculation:

```bash
python3 watch_power_progress.py \
  --version 352515 \
  --demo \
  --wandb-project subvector-ar-hw \
  --wandb-mode online
```

This sends six progressive synthetic snapshots in a few seconds. The plot,
W&B run name/configuration, run ID, CSV rows, and output directory are all
marked `SYNTHETIC DEMO`; default files go only under
`352515/adaptive/demo/`. These values are a telemetry test, not a power bound.

The primary green curve is `bounds_confidence`, the simultaneous-Monte-Carlo
upper conditional on density accuracy. `bounds_point` is shown dashed and is
not the protected endpoint. `power_cp1` is highlighted because it is normally
the largest of the three feasible DGP curves. The displayed interpolated gap
to `power_cp1` is a cross-experiment diagnostic only: those DGP simulations
use finite $n=250$ and estimated residual covariance, whereas the bound is for
the known-covariance Gaussian limit experiment. Consequently the watcher does
not stop the run automatically. `invariant_benchmark_lower_confidence` is
computed in the same limit experiment; if it exceeded `bounds_confidence`,
that would be a genuine failure, and the numerical driver itself refuses such
a result before checkpointing it.

One timing limitation is important. Both beta-invariant shared null banks are
built before the beta loop, so the baseline plot appears immediately but the
first non-null bound point appears only after that shared startup work. On the
measured weak-configuration machine this can still take several days. Once the
loop begins, every atomically completed beta appears without waiting for the
rest of the curve. Use Ctrl-C for a manual stop; the compatible partial
checkpoint and bank caches are designed to resume without `--force`.

## 12. Publication checklist

Before treating a run as reportable, verify:

- the artifact signature and C-library/source hashes match the run;
- every density call converged below `m-max`, with plausible selected-order and
  retry diagnostics;
- pooled raw masses and ratio caps are plausible and ESS is not poor;
- fit complementarity residuals, active supports, and discarded masses do not
  indicate an obviously loose mixture;
- the iid mixture point calibration attains alpha;
- the primary confidence upper is at least alpha and dominates the one-sided
  lower confidence limit for the valid same-experiment chi-square test;
- the independent common-grid maximum rejection probabilities and epsilon are
  reported with `finite_grid_only` scope; and
- any sensitivity to grid design or active-support cap is disclosed as a
  tightness issue, not folded into the Monte Carlo confidence percentage.

## References

- Elliott, Müller and Watson (2015), *Nearly Optimal Tests When a Nuisance
  Parameter Is Present Under the Null Hypothesis*,
  `research_papers/Elliott-NEARLYOPTIMALTESTS-2015.pdf`: Lemma 1 and equation
  (9), PDF page 9; update equation (10), PDF page 13; numerical algorithm, PDF
  pages 37--38.
- Guggenberger, Kleibergen and Mavroeidis supplement,
  `research_papers/666-3194-1-SP.pdf`: section D.3.2 and equation (D.1), PDF
  pages 14--16. Its published nuisance grid is one dimensional; the 68-point
  strength-by-shape grid and independent certification split are explicit
  $p=4$ extensions in this codebase.
- Koev and Edelman (2006), *The Efficient Evaluation of the Hypergeometric
  Function of a Matrix Argument*.
