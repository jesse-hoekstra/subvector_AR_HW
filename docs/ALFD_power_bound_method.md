# Direct GKM/EMW eigenvalue power bound

This document describes the calculation performed by `alfd_eigval.py` for the
paper design selected by `--version 352515`: $m_W=3$, $p=m_W+1=4$, $k=7$,
$n=250$, and $\alpha=0.05$.

The implementation follows Guggenberger, Kleibergen, and Mavroeidis (2019,
hereafter GKM), Supplement Section D.3.2, as directly as possible. GKM perform
the calculation for $m_W=1$. We retain their pooled ordinary importance
sampler, fixed EMW update, two critical-value calculations, and Monte Carlo
power calculation, and replace their scalar null grid with an explicitly
documented sparse grid for the three-dimensional null nuisance parameter.

The reported curve is a GKM-style Monte Carlo point estimate. It is not a
finite-Monte-Carlo confidence endpoint and it has no shaded confidence band.

## 1. Experiment and density

The Gaussian limit experiment observes

\[
\Xi\sim N_{k\times p}(M,I_k\otimes I_p),\qquad \Omega=M'M,
\]

and retains the ordered eigenvalues $x_1\ge\cdots\ge x_p$ of $\Xi'\Xi$.
Under the rank-deficient null for $p=4$,

\[
\Omega_0=\operatorname{diag}(\nu_1,\nu_2,\nu_3,0),\qquad
\nu_1\ge\nu_2\ge\nu_3\ge0.
\]

At a non-null beta, `asymptotic_ncp_eigenvalues` supplies the four ordered
eigenvalues of the full-rank alternative $\Omega_1$. Beta zero is analytically
rank deficient and its curve value is exactly $\alpha$; nearby beta values are
not classified as null by a floating-point tolerance.

Only the following part of the real noncentral-Wishart eigenvalue density
depends on $\Omega$:

\[
\log \widetilde f(x\mid\Omega)=
\log{}_0F_1^{(2)}\!\left(k/2;\Omega/4,X\right)
-\tfrac12\operatorname{tr}\Omega.
\]

The omitted factor depends on $x$ but not on $\Omega$, so it cancels from all
likelihood ratios and importance ratios. The bound concerns tests measurable
with respect to the full ordered eigenvalue vector, the invariant class studied
by GKM.

## 2. Which GKM quantity is the plotted bound?

For one alternative density $g$ and null-grid densities $f_1,\ldots,f_H$,
let

\[
h_\lambda(x)=\sum_{j=1}^H\lambda_jf_j(x),\qquad
\lambda_j\ge0,\quad\sum_j\lambda_j=1.
\]

EMW Lemma 1 says that the power of the level-$\alpha$ Neyman--Pearson test of
the mixture null $h_\lambda$ against $g$ is an upper bound on the power of any
test that controls size at every null parameter. EMW then search for an
approximately least-favourable $\lambda$ so this upper bound is tight.

GKM compute two related powers:

- $\bar\pi$ is the Step-7 power obtained after Step 6 calibrates the likelihood
  ratio to have Monte Carlo size $\alpha$ under the fitted null mixture.
- $\widetilde\pi$ is the Step-9 power after Step 8 raises the cutoff until the
  maximum estimated rejection probability over the finite null grid is
  $\alpha$.
- $\epsilon=\bar\pi-\widetilde\pi$ measures the effect of this grid
  adjustment.

GKM Supplement page 16 explicitly states that $\widetilde\pi$ is the
point-optimal power bound used in their Figure 3. Accordingly, the green curve
and the primary saved `bounds` array in this repository are
$\widetilde\pi$; `bounds_se` contains its conditional binomial Monte Carlo
standard error. The code saves $\bar\pi$ as `mixture_power` (with
`mixture_power_se`) and $\epsilon$ as `epsilon_grid`; they are diagnostics, not
additional plotted curves.

This terminology follows GKM. Numerically, both values are Monte Carlo point
estimates based on a finite discretization. There is no claim of simultaneous
confidence coverage, and repeated seeds by themselves do not create a valid
confidence interval.

## 3. The common 68-point null grid

GKM use 42 points, equally spaced in log scale between zero and 100, for their
one-dimensional nuisance parameter. Their main paper explicitly restricts the
actual power-bound computation to $m_W=1$ because the computational burden for
larger $m_W$ is overwhelming. The papers therefore do not prescribe a number
or arrangement of grid points for $m_W=3$.

Our deterministic higher-dimensional extension separates nuisance strength
from nuisance shape. For the weak configuration it contains

\[
H=1+9\times7+4=68
\]

rows:

- the origin;
- nine normalized shape directions, each evaluated at seven geometrically
  spaced largest-eigenvalue strengths from 0.1 through 100; and
- four exact stress anchors: $(35,25,15)$, $(50,25,5)$, $(15,10,5)$, and
  $(5,3,1)$.

The shape directions include the configured and stress shapes, rank-one,
rank-two, and equal rank-three boundaries, and low/high third-eigenvalue-share
shapes encountered on a fixed 81-point design path over beta in $[-2,2]$.
That design path is independent of `--beta-count`, and duplicate directions
are removed deterministically.

The same ordered grid is used for every beta. This matches GKM's reuse of one
null grid across alternatives and permits the expensive null-density table to
be cached once. The fitted weights and the two cutoffs are nevertheless
alternative-specific: each non-null beta defines a different point-optimal
problem. The number 68 is our documented sparse-grid choice, not a number
claimed by GKM. A finite grid cannot by itself verify size throughout the
unbounded three-dimensional cone, so a publication robustness check should
expand the shape and strength grids and report whether the curve changes.

## 4. One pooled ordinary-importance-sampling bank

For each grid law $f_s$, draw $N_0$ independent observations. With equal
stratum sizes, the proposal is

\[
q(x)=\frac1H\sum_{s=1}^H f_s(x).
\]

All $HN_0$ observations are pooled. For a rejection rule
$\phi(x)\in[0,1]$, GKM equation (D.1) estimates rejection probability under
null row $j$ as

\[
\widehat{RP}_j(\phi)=
\frac{1}{HN_0}\sum_{s=1}^H\sum_{i=1}^{N_0}
\frac{f_j(X_{is})}{q(X_{is})}\phi(X_{is}).
\]

This is ordinary stratified importance sampling. The realized contributions
must not be divided by their realized sum; doing so would silently change the
calculation to self-normalized importance sampling. The exact ratio obeys
$f_j/q\le H$. The code checks that cap and records raw masses and effective
sample sizes.

The bank stores every $f_j(X_{is})$ and $q(X_{is})$. It is beta invariant and
is built once. For each beta, only the alternative density $g(X_{is})$ must be
evaluated on this bank. There is no second audit bank, no separately simulated
mixture-calibration sample, and no support compression in the direct GKM path.
All $H$ fitted weights enter every cutoff and power calculation. The
normalized log weights are the authoritative numerical representation, so a
very small positive GKM weight cannot disappear through floating-point
underflow; both `fitted_log_weights` and ordinary `fitted_weights` are saved.

## 5. Exact per-alternative algorithm

For each non-null beta, the driver implements GKM Supplement D.3.2 as follows.

1. Use the common bank containing $N_0$ independent draws from each $f_j$ and
   all stored null densities.
2. Evaluate the alternative density $g$ on every bank observation.
3. Initialize $\mu^{(0)}=(-2,\ldots,-2)$.
4. For $s=0,\ldots,O-1$, form

   \[
   \phi^{(s)}(x)=1\!\left\{g(x)>
       \sum_{i=1}^H e^{\mu_i^{(s)}}f_i(x)\right\}
   \]

   and update every coordinate using GKM's fixed step,

   \[
   \mu_j^{(s+1)}=\mu_j^{(s)}+
       2\bigl(\widehat{RP}_j(\phi^{(s)})-\alpha\bigr).
   \]

   Reportable runs use exactly $O=600$. There is no sign-switch damping,
   early stopping, or best-iterate substitution.
5. Normalize the final iterate,

   \[
   \lambda_j=\frac{e^{\mu_j^{(O)}}}
   {\sum_i e^{\mu_i^{(O)}}}.
   \]

6. Find $\kappa^*$ such that

   \[
   \sum_{j=1}^H\lambda_j
   \left(\widehat{RP}_j\!\left[
   1\{g>\kappa^*h_\lambda\}\right]-\alpha\right)=0.
   \]

   This is GKM Step 6. The implementation uses the same raw ordinary-IS
   contributions as equation (D.1), with explicit randomization at a finite-
   sample tie.
7. Draw $N_1$ fresh observations directly from the alternative, independently
   of the null bank, and estimate

   \[
   \bar\pi=N_1^{-1}\sum_{i=1}^{N_1}
   1\{g(X_i)>\kappa^*h_\lambda(X_i)\}.
   \]

8. On the same pooled null bank, find $\widetilde\kappa\ge\kappa^*$ such that

   \[
   \max_{j=1,\ldots,H}
   \left(\widehat{RP}_j\!\left[
   1\{g>\widetilde\kappa h_\lambda\}\right]-\alpha\right)=0.
   \]

9. On the same fresh alternative sample, estimate the plotted value

   \[
   \widetilde\pi=N_1^{-1}\sum_{i=1}^{N_1}
   1\{g(X_i)>\widetilde\kappa h_\lambda(X_i)\},
   \qquad \epsilon=\bar\pi-\widetilde\pi.
   \]

The saved standard error for $\widetilde\pi$ is the ordinary binomial Monte
Carlo standard error conditional on the fitted rule. It does not include
uncertainty from fitting the mixture or estimating either cutoff.

## 6. Adaptive matrix-hypergeometric truncation

GKM Supplement Section D.1 use a fixed total-degree truncation $M=200$ for
their $p=2$ calculations after experimenting with values as high as 500 and
finding no further change. That empirical choice cannot simply be assumed
adequate or efficient for the much harder $p=4$ densities.

The extension therefore retains per-density adaptive truncation. For every
pair $(\Omega,x)$, the evaluator:

1. chooses a trace-based initial total degree no smaller than `--m-start`;
2. evaluates the positive coefficient series by total degree;
3. checks that the coefficient peak is behind a contracting trailing window
   and that a geometric omitted-tail estimate is below `--mhg-rtol`;
4. retries only that density pair in `--m-step` increments; and
5. fails rather than accepting a value at `--m-max` or after a suspicious
   numerical coefficient collapse.

This changes only how accurately the common Wishart density is evaluated; it
does not change the GKM statistical algorithm. Selected-order histograms,
maximum order, remainder diagnostics, and retry counts are saved. The criterion
is a numerical tail diagnostic, not a symbolic error bound, so density
accuracy should be included in robustness checks.

## 7. Simulation budgets and runtime

GKM report

\[
N_0=10{,}000\text{ per null point},\qquad
N_1=100{,}000\text{ per alternative},\qquad O=600.
\]

These are the `reference` profile values. The less expensive `production`
profile is useful for deciding whether the comparison is promising, but it is
not the numerical scale reported by GKM.

| profile | $N_0$ (`n_fit`) | $N_1$ (`n_power`) | $O$ (`n_iter`) |
|---|---:|---:|---:|
| `production` | 2,000 per null row | 50,000 per non-null beta | 600 |
| `reference` | 10,000 per null row | 100,000 per non-null beta | 600 |

Only three simulation controls remain relevant:

- `n_fit` is $N_0$. It controls noise in the importance-sampled null rejection
  probabilities, the fitted weights, and both cutoffs.
- `n_power` is $N_1$. It controls direct alternative-power Monte Carlo noise.
- `n_iter` is $O$. It adds no density evaluations after the pooled table has
  been built; reportable direct-GKM runs use 600.

There are no `n_calibration`, `n_validation`, `curve_confidence`, or
`max_active_support` choices in this workflow. Those belonged to a different
confidence-protected implementation and are deliberately absent here.

Let $B$ be the number of non-null beta points. The exact logical density-pair
count, before adaptive retries, is

\[
N_{\rm pairs}=H^2N_0+B\{HN_0+(H+1)N_1\}.
\]

The first term is the one-time null table. For every alternative, $HN_0$
evaluations add its density to the bank, and $(H+1)N_1$ evaluations score the
fresh alternative sample under the alternative and every null density. The
default nine-point symmetric beta grid contains beta zero and therefore has
$B=8$ non-null points. With $H=68$, this is 37,936,000 logical pairs for the
production profile and 106,880,000 for the reference profile. Adaptive retries
can add raw hypergeometric evaluations beyond those totals.

`--preflight-only` prints the exact requested count without running the
simulation. `--benchmark-preflight` times representative adaptive density
calls on the target machine and extrapolates the request. The extrapolation is
only an estimate: selected hypergeometric orders vary by observation and
multiprocessing scaling is sublinear.

Fewer beta points reduce runtime almost proportionally after the common bank is
built. They change curve resolution, not the pointwise algorithm. Prefer an
odd count so beta zero is included. Repeating the full calculation at
prespecified independent seeds and with an expanded grid is a useful
sensitivity analysis, but the spread across runs is not automatically a
confidence band and should not be selectively reported.

## 8. Commands

Build the Koev library once:

```bash
sh koev/mhg15/build.sh
```

Print the exact production work count for the default nine beta points:

```bash
python3 alfd_eigval.py \
  --version 352515 \
  --profile production \
  --workers 48 \
  --beta-count 9 \
  --preflight-only
```

Measure representative density speed on the machine that will run the job:

```bash
python3 alfd_eigval.py \
  --version 352515 \
  --profile production \
  --workers 48 \
  --beta-count 9 \
  --benchmark-preflight \
  --benchmark-samples 96
```

Start or resume the production calculation:

```bash
python3 alfd_eigval.py \
  --version 352515 \
  --profile production \
  --workers 48 \
  --beta-count 9 \
  --acknowledge-expensive
```

For the GKM-reported Monte Carlo scale, replace `production` by `reference`,
run its preflight and benchmark first, and then acknowledge the measured cost.
The final result and compatible partial checkpoint live under
`352515/gkm_direct/`. A compatible interrupted run resumes automatically. Do not
use `--force` unless the existing artifact or checkpoint is intentionally being
replaced.

The finite-sample comparison curves must also exist for the live overlay. Build
their provenance-checked cache with:

```bash
python3 new_power_comparison.py \
  --version 352515 \
  --num-simulations 100000 \
  --workers 48 \
  --seed 20240101 \
  --chunk-size 5000 \
  --acknowledge-expensive
```

Then run the read-only watcher in another terminal or `tmux` pane:

```bash
python3 -m pip install wandb
wandb login
python3 watch_power_progress.py \
  --version 352515 \
  --wandb-project subvector-ar-hw \
  --wandb-mode online
```

The watcher writes an atomic local PNG and long-format CSV and optionally sends
the same snapshots to W&B. The presentation figure contains only $\chi^2$,
$c_1$, $c_3$, the green GKM $\widetilde\pi$ curve, and the gray $\alpha$
line. Numerical values remain in the CSV and NPZ; the plot is not the sole
record.

Test only the logging integration, with no scientific calculation, using:

```bash
python3 watch_power_progress.py \
  --version 352515 \
  --demo \
  --wandb-project subvector-ar-hw \
  --wandb-mode online
```

Every demo artifact is marked synthetic and is not a power-bound result.

## 9. What to report and check

Before using a curve in the paper, verify and retain:

- the exact 68 grid rows and the fact that this is our $m_W=3$ construction;
- $N_0$, $N_1$, 600 iterations, the beta grid, and the random seed;
- ordinary-IS raw masses, ratio caps, and effective sample sizes;
- final mixture weights and estimated null rejection probabilities;
- $\bar\pi$, $\widetilde\pi$, and $\epsilon$ at every beta;
- adaptive hypergeometric selected orders and absence of convergence failures;
- source and compiled-library hashes stored with the artifact; and
- sensitivity to a denser/wider null grid and, for a final result, a
  prespecified independent seed.

A concise methods description is:

> We extend the ALFD computation in GKM Supplement D.3.2 from $m_W=1$ to
> $m_W=3$. We replace their 42-point scalar null grid with a documented
> 68-point strength-by-shape discretization of the ordered three-dimensional
> nuisance cone, retain their pooled ordinary importance sampler, fixed
> 600-step EMW update, and Step-6/Step-8 calibrations, and report their Step-9
> grid-adjusted Monte Carlo power bound $\widetilde\pi$.

Do not describe the curve as a 95% or 99% confidence upper bound. It is the
direct GKM-style grid-adjusted Monte Carlo power-bound calculation.

## References

- Guggenberger, Kleibergen, and Mavroeidis (2019), main paper: PDF pages 10--12
  (journal pages 496--498) for the power problem and Figure 3; PDF page 11
  states that the numerical power-bound calculation is restricted to
  $m_W=1$ because higher-dimensional computation is overwhelming.
- GKM Supplement: PDF pages 3--7 for the scalar null/alternative grids and
  ordinary-IS equation (D.1); pages 12--16 for Steps 1--10, $N_0=10{,}000$,
  $N_1=100{,}000$, and the identification of $\widetilde\pi$ as the Figure-3
  bound; page 1 for the fixed-$M=200$ density calculation.
- Elliott, Müller, and Watson (2015): PDF page 9 (journal page 778) for Lemma 1
  and PDF pages 37--38 (journal pages 806--807) for the original eight-step
  ALFD algorithm.
