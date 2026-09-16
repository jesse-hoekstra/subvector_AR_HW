# subvector_AR_HW

A small research helper accompanying the paper: "Best Feasible Conditional Critical Values for a More Powerful Subvector Anderson-Rubin Test", to simulate the joint distribution of the smallest two eigenvalues of a noncentral real Wishart matrix $W = X^\top X$ and to plot the empirical conditional CDF of the smallest eigenvalue given the second smallest eigenvalue for different $\\kappa$ configurations, where the approximation given in GKM represents the condititional cdf when the $p-2$ largest eigenvalues are $\infty$.

The original conditional-CDF executable is `simulation_plot_executable.py`.
The repository also contains direct $m_W=2$ and $m_W=3$ extensions of the GKM
power-bound calculation in `alfd_eigval.py`, finite-sample comparison curves
from `new_power_comparison.py`, and the live overlay in
`watch_power_progress.py`.

## What it does

- Simulates joint eigenvalues $(\hat{\kappa}\_{p}, \hat{\kappa}\_{p-1})$ for a given dimension `p` and number of instruments `k`, with noncentrality specified by `mu`.
- Estimates the median of $\\hat{\kappa}_{p-1}$ from a marginal simulation to define the conditioning value.
- Computes and plots the empirical conditional CDF of $\hat{\kappa}\_{p} \mid \hat{\kappa}\_{p-1}$ for several $\kappa$ configurations.
- Overlays an analytical conditional CDF approximation via the functions `g_k1`, `conditional_density`, and `get_conditional_cdf_GKM`, which represents the condititional cdf when the $p-2$ largest $\\kappa$ values are $\\infty$.

## Requirements

- Python 3.9+ (3.10/3.11 also fine)
- Packages:
  - numpy
  - scipy
  - matplotlib
  - wandb (optional; only for the live power-bound dashboard)

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib
```

## Run

From the repository root:

```bash
python simulation_plot_executable.py
```

The script is interactive; it will prompt for `p` and `n`:

```
Enter a p>2: 3
Enter the number of instruments k: 10
```
An interactive Matplotlib window will open, showing:
![Example image for p=3 and n=10](figures/example_figure.png)

## GKM/EMW power bound

`alfd_eigval.py` is a direct computational extension of GKM Supplement
Section D.3.2 from $m_W=1$ ($p=2$) to $m_W=3$ ($p=4$). It uses one common
68-point discretization of the three-dimensional null nuisance cone, one
pooled stratified null bank, GKM's ordinary (not self-normalized) importance
sampler, the fixed 600-step EMW update, and GKM's Step-6 and Step-8 cutoffs.
The plotted green value is the Step-9 quantity $\widetilde\pi$, which GKM use
for Figure 3. The Step-7 value $\bar\pi$ and
$\epsilon=\bar\pi-\widetilde\pi$ are saved as diagnostics.

These are Monte Carlo point estimates following GKM; there is no confidence
band or simultaneous-confidence claim. The number 68 is also not taken from
GKM: their scalar calculation uses 42 points. Our transparent higher-
dimensional design is the origin, nine nuisance-shape rays at seven
log-spaced strengths, and four exact stress anchors. The path-dependent rays
come from a fixed 81-point design path, so changing `--beta-count` does not
change the null grid. Hypergeometric
truncation adapts separately for every density pair because GKM's fixed
$M=200$ check was performed for $p=2$, not for this $p=4$ extension.

```bash
sh koev/mhg15/build.sh
python3 alfd_eigval.py --version 352515 --profile production \
  --beta-count 9 --preflight-only --workers 48
python3 alfd_eigval.py --version 352515 --profile production \
  --beta-count 9 --benchmark-preflight --benchmark-samples 96 --workers 48
python3 alfd_eigval.py --version 352515 --profile production \
  --beta-count 9 --workers 48 \
  --acknowledge-expensive
python3 new_power_comparison.py --version 352515 --preflight-only
python3 new_power_comparison.py --version 352515 --acknowledge-expensive
```

Direct-GKM bound artifacts are written under `<version>/gkm_direct/`, and
provenance-checked finite-sample caches under `<version>/dgp/`. Legacy
`M_trunc_*` files are intentionally not loaded. Matrix-hypergeometric order is
selected separately for every density evaluation, so a curve no longer needs
to be rerun at a sequence of fixed `M_trunc` values. The default is nine
symmetric beta points from -2 through 2, including the exact beta-zero point.
See `docs/ALFD_power_bound_method.md` for the exact algorithm, meanings of
$\widetilde\pi$, $\bar\pi$, and $\epsilon$, paper-scale budgets, runtime
accounting, and the limitation of a finite grid when $m_W=3$.

### Build only the null bank for m_W=2, kappa=[100,15]

The `10015` preset uses `m_W=2` (`p=3`), `k=7`, `n=250`, and
`alpha=0.05`. Its design retains the first two nuisance regressors of the
existing Appendix A.3 implementation: the leading 4-by-4 error covariance,
the first two columns of `A`, and `gamma=[-1,1]`; `pi_x` keeps all seven
instrument coordinates. This is an explicit reduction of the existing design.

The two-dimensional null grid contains the origin and five shape directions
at seven log-spaced strengths from 0.1 through 100, including the exact
`(100,15)` point and the rank-one and equal-eigenvalue boundaries. Directions
selected from the alternative path use the fixed 81-point design path, so
changing `--beta-count` does not change this bank. As with the three-dimensional
calculation, this is a finite-grid approximation. The five directions retain
the configuration ratio, both boundaries, and both extreme alternative-path
ratios; duplicate ratios are replaced by fixed interior directions. Use
`--grid-shapes 9` to select the denser 64-point grid for a sensitivity check.
The existing `m_W=3` presets keep nine directions; `352515` retains 68 null points.

On the compute machine, activate the Python environment and build the native
library once if needed. The commands use the reference profile to retain the
10,000 draws per null recorded in the previous `352515` run. Set `--workers`
to the number of CPUs allocated to the job; the commands below use 48.
Limit BLAS threads so each process does not
also start its own group of numerical-library threads:

```bash
source .venv/bin/activate
sh koev/mhg15/build.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

# Inspect exact null-bank cost without simulating or writing files.
python alfd_eigval.py --version 10015 --profile reference \
  --null-bank-only --workers 48 --preflight-only

# Measure this configuration on the target machine, without saving a bank.
python alfd_eigval.py --version 10015 --profile reference \
  --null-bank-only --workers 48 --benchmark-preflight --benchmark-samples 96

# Build and save only the reusable null bank.
python -u alfd_eigval.py --version 10015 --profile reference \
  --null-bank-only --workers 48 --acknowledge-expensive
```

The reference profile uses `N0=10,000` draws per null: **36 null points,
360,000 pooled draws, and 12,960,000 density pairs**. This is 68.4% fewer
density evaluations than the denser 64-point grid at the same draw budget.
The production profile uses `N0=2,000`, giving 72,000 pooled draws and
2,592,000 pairs. The costly density evaluations run in a process pool
with dynamically assigned sample chunks, capped at 100 samples per task.
Draws are generated from one deterministic seed before density evaluation;
changing the number of workers preserves the bank's draws and cache identity.
`--workers` distributes work across CPUs on one machine.

The completed bank is saved to `10015/gkm_direct/pooled_gkm_<hash>.npz`,
with a log at `10015/gkm_direct/null_bank_run.log`. Bank-only mode performs
no EMW weight fitting or alternative-power simulation and writes no partial
or final power result. Rerunning the command loads a compatible completed
bank. The bank is saved after all chunks finish; an interrupted initial build
must restart. Use a persistent session such as `tmux` for the long run.

To calculate the bound curve later, omit `--null-bank-only`:

```bash
python -u alfd_eigval.py --version 10015 --profile reference \
  --workers 48 --beta-count 9 --acknowledge-expensive
```

This reuses the bank and saves the bound values in
`10015/gkm_direct/gkm_eigval_10015.npz`. Keep the code, native library,
environment, grid, seed, `--n-fit`, and adaptive-M settings unchanged between
building and using the bank. Worker count, beta count, `--n-power`, and
`--n-iter` do not enter the bank identity. Existing `352515` caches remain
on disk, but the strict source-hash check means they need the original code
revision for reuse. The finite-sample comparison also supports `10015`;
the existing live watcher and refinement scripts target `m_W=3` workflows.

### Use several SSH-accessible CPU nodes with shared storage

`--workers` starts processes on the current node. To use multiple nodes without
a scheduler, `null_bank_cluster.py` divides the bank into numbered pieces
(shards). Each piece covers different sampled observations and evaluates all
36 null densities for those observations. Merging restores the original
observation order and produces the same bank format and cache identity as a
single-node run.

The example below uses four nodes and 48 local workers per node, for 192 worker
processes in total. This is a starting point for otherwise idle nodes with
48 physical cores each. The [Xeon Gold 6252N](https://www.intel.com/content/www/us/en/products/sku/193951/intel-xeon-gold-6252n-processor-35-75m-cache-2-30-ghz/specifications.html)
has 24 physical cores and 48 hardware threads per processor. A node reporting
96 logical CPUs commonly has two sockets, 24 cores per socket, and two threads
per core; confirm these fields in `lscpu`. Logical threads are not additional
physical cores, so 96 workers is not automatically faster than 48. Use fewer
workers when cores are shared with other work. The best count requires sustained
timing on the target node; the short built-in benchmark includes pool startup.

Replace `/shared/path/subvector_AR_HW` with the directory visible on all nodes,
and use the same Python environment and code on each node.

**1. Prepare once, on any node.** This generates the common draws and settings;
it does not evaluate densities.

```bash
cd /shared/path/subvector_AR_HW
source .venv/bin/activate
python null_bank_cluster.py prepare --version 10015 --profile reference \
  --shards 4 --directory 10015/gkm_direct/distributed
```

At the reference budget, each of four pieces contains 90,000 observations
and 3,240,000 density evaluations. Observations are interleaved across pieces
so every node receives draws from every null stratum.

**2. SSH into each node and start a separate tmux session.** On each node:

```bash
tmux new -s null10015
cd /shared/path/subvector_AR_HW
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
```

Then run the worker command below, using a different `--shard` on each node:

| Node | Shard |
|---|---:|
| A | 0 |
| B | 1 |
| C | 2 |
| D | 3 |

```bash
# Node A; replace 0 with 1, 2, or 3 on the other nodes.
null_shard=0
set -o pipefail
python -u null_bank_cluster.py worker \
  --directory 10015/gkm_direct/distributed --shard "$null_shard" --workers 48 \
  2>&1 | tee -a "10015/gkm_direct/distributed/shard_${null_shard}.log"
```

The distributed worker itself prints to the terminal; `tee -a` also saves both
normal output and errors to a separate log for each piece, preserving previous
attempts. The shared folder makes these logs visible from every node. Progress
lines show completed/total chunks and density pairs, pairs per second, elapsed
time, and estimated time remaining. They are emitted when a chunk finishes and
at least 15 seconds have passed since the previous update, plus at completion.
Long-running chunks can cause a longer gap. Watch all started pieces with:

```bash
tail -n 2 -f 10015/gkm_direct/distributed/shard_*.log
```

`status` below verifies which entire pieces are complete, missing, or invalid;
the logs provide progress within each running piece. This logging command does
not change the code or prepared bank identity.

Detach with **Ctrl-b, then d**. Reconnect by SSHing to the same node and
running `tmux attach -t null10015`. Each node may use a different worker count.
Completed pieces are checked and reused on rerun; an interrupted piece is
recomputed. A lock prevents concurrent writers to the same piece. Run one
different piece per node, rather than the full `alfd_eigval.py` bank command
on every node, which would repeat the calculation.

**3. Check completion and merge once, on any node.**

```bash
python null_bank_cluster.py status --directory 10015/gkm_direct/distributed
python null_bank_cluster.py merge --directory 10015/gkm_direct/distributed \
  --cache-dir 10015/gkm_direct
```

Merge rejects missing, incompatible, or damaged pieces. It saves the usual
`10015/gkm_direct/pooled_gkm_<hash>.npz`, which the standard power command above
loads automatically. Wait for merge to finish before starting that command.
Keep the prepared directory until the merged bank has
been verified. Preparing the same directory again is a no-op when the settings
match; choose a new directory for different settings or a different shard count.

### Calculate the 17-point power bound on four nodes using the finished bank

Copy **only `power_bound_cluster.py`** into the shared project directory. Keep
`alfd_eigval.py`, `null_bank_cluster.py`, the native MHG library, and the Python
environment used to build the bank unchanged. The new driver loads and checks
an existing merged `pooled_gkm_*.npz`; it has no null-bank-building path.

From the shared project directory, prepare the power run once:

```bash
source .venv/bin/activate
python power_bound_cluster.py prepare --version 10015 --profile reference \
  --bank-dir 10015/gkm_direct --beta-count 17 --shards 4 \
  --directory 10015/gkm_direct/power17
```

If the bank directory contains more than one compatible bank, select the intended
file with `--bank /path/to/pooled_gkm_<hash>.npz` instead of `--bank-dir`.
Missing, corrupted, or incompatible banks cause an error before work starts.
Preparation fixes the beta values `[-2, -1.75, ..., 0, ..., 1.75, 2]`, the
bank identity, and deterministic per-beta random seeds. It records the exact
beta-zero result (`alpha=0.05`) without a simulation, then assigns the 16
nonzero betas as follows:

| Node | `--shard` | Beta values |
|---|---:|---|
| A | 0 | -2, -1, 0.25, 1.25 |
| B | 1 | -1.75, -0.75, 0.5, 1.5 |
| C | 2 | -1.5, -0.5, 0.75, 1.75 |
| D | 3 | -1.25, -0.25, 1, 2 |

On each node, start `tmux new -s power10015` (or use a free existing session),
enter the shared project directory, activate the environment, and run:

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
power_shard=0  # use 1, 2, or 3 on the other nodes
set -o pipefail
python -u power_bound_cluster.py worker \
  --directory 10015/gkm_direct/power17 --shard "$power_shard" --workers 48 \
  2>&1 | tee -a "10015/gkm_direct/power17/shard_${power_shard}.log"
```

Use 48 workers only if the node has 48 physical cores available to your run;
otherwise lower `--workers` (for example, to 16). A report of 96 logical CPUs
alone does not establish the available physical-core count.

Each node calculates four beta points sequentially, using its local workers
for density evaluations. A separate authenticated `beta_<index>.npz` is saved
after each point. Rerunning the same worker command verifies and skips its
completed points; only an interrupted beta has to restart. Locks prevent
duplicate computation of the same point. The bank is read without modification.

With the 36-null reference bank (`N0=10,000`), the reference power settings use
`N1=100,000` alternative draws and 600 EMW updates per nonzero beta. There are
4,060,000 new density pairs per beta, or **64,960,000 across all 16 points**.
The bank's original 12,960,000 density pairs are not recomputed. The EMW update
stage works within each node's main process; the expensive density phases use
the local process pool. Keep the same prepared power directory and script
version throughout a run.

Monitor from any node:

```bash
python power_bound_cluster.py status --directory 10015/gkm_direct/power17
tail -n 2 -f 10015/gkm_direct/power17/shard_*.log
```

After all 17 results are present, merge once:

```bash
python power_bound_cluster.py merge --directory 10015/gkm_direct/power17
```

The merged result is sorted by beta and written below `10015/gkm_direct/power17/`:
`gkm_eigval_10015.npz` holds the full scientific results and diagnostics,
`gkm_bounds_10015.csv` contains the bound estimates and Monte Carlo standard
errors, and `power_bound_10015.png` plots the bound curve.
The separate existing `m_W=3` watcher and refinement scripts are not required.

### Simulate the other power curves locally for 10015

`new_power_comparison.py` now supports the same two-nuisance design as the
`10015` bound: `kappa=[100,15]`, `k=7`, `n=250`, `alpha=0.05`. It simulates
the feasible tests labelled chi-squared, `c_1`, and `c_2` (the last conditions
on the second-smallest of three eigenvalues). The beta-zero rejection
probability is simulated too. No null bank or native MHG library is needed.

From the local repository root:

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

python -u new_power_comparison.py --version 10015 \
  --beta-count 17 --num-simulations 100000 --workers 4 \
  --seed 20240101 --chunk-size 5000 \
  --acknowledge-expensive --no-show
```

Choose the local worker count according to the cores available on your computer.
This is 1,700,000 finite-sample draws across `[-2,-1.75,...,0,...,1.75,2]`.
Add `--preflight-only` to inspect the run without simulation. For smoother
feasible-test lines use `--beta-count 81` instead (8,100,000 draws); the bound
can remain at 17 points. Choose the grid before the run: the cache records it,
and changing it requires explicit replacement with `--force`.

Outputs below `10015/dgp/`:

- `dgp_curves_10015.npz`: the three curves with settings and provenance.
- `dgp_curves_10015.csv`: power and binomial Monte Carlo standard errors.
- `power_curve_kappas_100_15.png`: the three feasible-test curves.
- `dgp_curves_run.log`: simulation progress and completion messages.

The NPZ uses the existing key `power_cp1` for the `c_2` curve; the CSV calls it
`power_c2`. Results are reproducible across worker counts when the beta grid,
seed, simulation count, and chunk size stay fixed. A completed compatible
cache is reused; an interrupted simulation must restart the DGP sweep.

Once the four-node bound has been merged, copy just its completed
`gkm_eigval_10015.npz` into local `10015/gkm_direct/power17/`, then run:

```bash
python plot_power_comparison.py --version 10015 \
  --bound 10015/gkm_direct/power17/gkm_eigval_10015.npz
```

This writes `10015/power_comparison_10015.png` and a CSV beside it with all
plotted values and Monte Carlo standard errors. It validates the experiment
settings and the bound's saved per-beta hashes; viewing a result does not
require the remote machine's environment or its null bank. Omit `--bound`
to plot only the saved feasible-test curves. The curves are Monte Carlo
estimates; the CSV standard errors describe simulation noise, not the
finite-null-grid approximation in the bound.

### Add midpoints to a completed bound curve

For an existing nine-point curve, `refine_power_curve.py` adds the eight
midpoints and saves a merged 17-point result. Copy this additional script to
the server alongside the existing scripts, and use the original Python
environment. It inherits the original simulation settings, loads and verifies
the existing `pooled_gkm_*.npz` bank, and copies all previously calculated
per-beta results unchanged. Missing or incompatible banks cause an error;
the refinement driver has no bank-building path. Keep `alfd_eigval.py` and
the MHG library unchanged because their hashes identify the cached bank.

```bash
python3 refine_power_curve.py --version 352515 --beta-count 17 \
  --workers 48 --preflight-only
python3 refine_power_curve.py --version 352515 --beta-count 17 \
  --workers 48 --acknowledge-expensive
```

The new beta values are `[-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]`.
Outputs are separate from the original run, under `352515/gkm_direct/refined/`:
`gkm_eigval_352515.npz` contains the complete merged result, and
`gkm_bounds_352515.csv` contains beta, bound, and Monte Carlo standard error.
A partial NPZ and CSV are updated after each completed midpoint. Rerunning
the same command resumes only missing midpoints; a completed refinement is
a no-op. Refinement provenance records the original file hash and distinct,
deterministic seeds for the added beta values.

After completion, render the denser curve with the existing DGP overlay:

```bash
python3 watch_power_progress.py --version 352515 --once \
  --partial-path 352515/gkm_direct/refined/gkm_eigval_352515.partial.npz \
  --final-path 352515/gkm_direct/refined/gkm_eigval_352515.npz \
  --output 352515/gkm_direct/refined/live_power_progress_352515.png
```

For live monitoring, omit `--once`; the existing W&B options also work. The
plot and completed count include both original and added points. The existing
watcher's `latest_beta` scalar stays at 2 because it selects the largest
completed beta, which is already present in the original result.

Bank reuse avoids the initial null-density computation. Each new beta still
requires fitting and density evaluations on fresh alternative draws, so the
refinement remains a substantial calculation.

### Follow a long bound run live

The W&B integration is deliberately a separate read-only watcher. This keeps
W&B network activity and background processes out of the numerical program's
multiprocessing pools. First create or validate the finite-sample DGP cache:

```bash
python3 new_power_comparison.py \
  --version 352515 \
  --num-simulations 100000 \
  --workers 48 \
  --seed 20240101 \
  --chunk-size 5000 \
  --acknowledge-expensive
```

Start the numerical driver in the first `tmux` pane so its new partial
checkpoint exists before any older completed artifact can be inspected:

```bash
python3 alfd_eigval.py \
  --version 352515 \
  --profile production \
  --workers 48 \
  --beta-count 9 \
  --acknowledge-expensive
```

Then start the watcher in a second pane or terminal:

```bash
python3 -m pip install wandb
wandb login
python3 watch_power_progress.py \
  --version 352515 \
  --wandb-project subvector-ar-hw \
  --wandb-mode online
```

It validates the provenance-checked DGP cache, polls the atomically written
GKM/ALFD checkpoint, saves a local progress PNG and long-format CSV under
`352515/gkm_direct/`, and uploads the same overlay after every completed beta.
The numeric files are `gkm_eigval_352515.npz` (complete scientific result) and
`live_power_progress_352515.csv` (the plotted values in a readable table).
The figure contains only the three cached curves, labelled $\chi^2$, $c_1$,
and $c_3$, the GKM Step-9 $\widetilde\pi$ curve, and a gray horizontal
$\alpha$ reference line. The Step-7 $\bar\pi$, $\epsilon$, and numerical
diagnostics remain in the NPZ artifact but are deliberately omitted from
the presentation plot.

To test only the W&B plumbing in a few seconds, without a DGP cache or any
matrix-hypergeometric work, run:

```bash
python3 watch_power_progress.py \
  --version 352515 \
  --demo \
  --wandb-project subvector-ar-hw \
  --wandb-mode online
```

The demo is visibly marked synthetic, gets a separate demo run ID,
and writes only below `352515/gkm_direct/demo/`. It is not a power calculation.


## Inputs and configurable knobs

- **p**: integer > 2 (matrix dimension)
- **k**: integer > p (number of instruments)
- **Simulation sizes** (inside `main()`):
  - `num_simulations_marginal = 100000` (used to estimate the conditioning value $\\hat{\kappa}_{p-1}$)
  - `num_simulations_conditional = 1000000` (used for the empirical conditional CDF)
- **Noncentrality patterns**: the script compares three settings:
  - `get_mu_list(2, 0, p)`
  - `get_mu_list(5, 0, p)`
  - `get_mu_list(10, 0, p)`

`get_mu_list(start, end, p, middle_value=2)` returns `p` points that are evenly spaced after a square-root transform and include $\\sqrt{\text{middle value}}$.

## How it works (high level)

- `simulate_joint_eigenvalues(p, n, mu, num_simulations)` constructs `X ~ N(M, I)` where `M` is diagonal with entries from `mu`, forms `W = X.T @ X`, and records the two smallest eigenvalues per simulation.
- The conditioning value $\\hat{\kappa}_{p-1}$ is the median of the simulated second-smallest eigenvalue from a marginal run. 
- With these ingredients the empirical CDF of the smallest eigenvalue given the second-smallest eigenvalue can be calculated and plotted.
- The “GKM” section builds an approximate conditional density via `g_k1` and `conditional_density`, and integrates it with `get_conditional_cdf_GKM` to obtain a CDF curve where the $p-2$ largest $\\kappa$ values are $\\infty$ for comparison.
