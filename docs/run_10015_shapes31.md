# 10015 sensitivity run with 31 shapes

Use the existing scripts unchanged. Set `--grid-shapes 31` on null-bank
**prepare**; workers, merge, and power calculations read the saved grid.

## Why 31

For seven strength values and `N0=10,000` draws per null, the number of nulls
is `H = 1 + 7 * shapes`, and the bank evaluates `N0 * H * H` density pairs.

| Shapes | Nulls | Bank density pairs | Work relative to nine shapes |
|---:|---:|---:|---:|
| 9 | 64 | 40,960,000 | 1.00x |
| 29 | 204 | 416,160,000 | 10.16x |
| **31** | **218** | **475,240,000** | **11.60x** |
| 32 | 225 | 506,250,000 | 12.36x |
| 90 | 631 | 3,981,610,000 | 97.21x |

31 keeps every nine-shape null and spreads further ratios across `[0,1]`
using the existing deterministic golden-ratio sequence. The maximum gap
between adjacent ratios is 0.05572809, compared with 0.25 for nine shapes
and 0.09016994 for 29 or 30. The 32nd shape does not reduce the largest gap.

Keep the seven strengths, their maximum of 100, the seed, and Monte Carlo
budgets unchanged to study shape refinement. This covers shape more densely
within the existing strength range; the full null parameter space is
unbounded. The seven strengths remain a coarse discretization from 0.1 to 100.
A stable result across shape counts is evidence about that refinement, not a
certificate of the tightest bound over the full null space.

The 31-shape bank has 2,180,000 pooled observations. Its main density matrix
occupies 3.80 GB (3.54 GiB); merge and power fitting allocate several additional
large arrays. The four nodes share density work, but each power worker loads
the complete merged bank. Allow substantially more RAM than the matrix alone.
The density-pair ratios above are workload counts, not measured runtime ratios.

## 1. Prepare once on any node

From the shared repository directory, using the existing Python environment:

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

# Optional: inspect cost without creating a bank or running simulations.
python alfd_eigval.py --version 10015 --profile reference \
  --null-bank-only --grid-shapes 31 --workers 48 --preflight-only

python null_bank_cluster.py prepare \
  --version 10015 --profile reference --seed 42 \
  --grid-shapes 31 --grid-strengths 7 --grid-max-strength 100 \
  --n-fit 10000 --shards 4 \
  --directory 10015/gkm_direct/shapes31/distributed
```

The fresh directory preserves earlier runs. No source-code updates or native
library rebuilds are needed.

## 2. Run one null-bank shard per node

Start `tmux new -s null10015_31` on each of the four nodes, enter the shared
repository directory, then run the block below. Assign shards 0, 1, 2, and 3
to the respective nodes. Use fewer local workers if fewer cores are available.

```bash
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
set -o pipefail
null_shard=0  # CHANGE to 1, 2, or 3 on the other nodes

python -u null_bank_cluster.py worker \
  --directory 10015/gkm_direct/shapes31/distributed \
  --shard "$null_shard" --workers 48 \
  2>&1 | tee -a "10015/gkm_direct/shapes31/distributed/shard_${null_shard}.log"
```

Each shard covers 545,000 observations and 118,810,000 density pairs. Detach
with Ctrl+B, then D. Inspect completion from any node:

```bash
python null_bank_cluster.py status \
  --directory 10015/gkm_direct/shapes31/distributed
```

## 3. Merge the completed null bank once

```bash
python null_bank_cluster.py merge \
  --directory 10015/gkm_direct/shapes31/distributed \
  --cache-dir 10015/gkm_direct/shapes31
```

## 4. Prepare the 17-point power run once

```bash
python power_bound_cluster.py prepare \
  --version 10015 --profile reference --seed 42 \
  --beta-count 17 --shards 4 --n-power 100000 --n-iter 600 \
  --bank-dir 10015/gkm_direct/shapes31 \
  --directory 10015/gkm_direct/shapes31/power17
```

This inherits all 218 nulls from the new bank. Each nonzero beta requires
24,080,000 new density pairs, 3.37 times the nine-shape power-stage count.

## 5. Run one power shard per node

In a tmux session on each node, use the same environment and thread limits as
above. Assign a different power shard to each node:

```bash
set -o pipefail
power_shard=0  # CHANGE to 1, 2, or 3 on the other nodes

python -u power_bound_cluster.py worker \
  --directory 10015/gkm_direct/shapes31/power17 \
  --shard "$power_shard" --workers 48 \
  2>&1 | tee -a "10015/gkm_direct/shapes31/power17/shard_${power_shard}.log"
```

Rerunning skips verified completed beta points. Inspect completion:

```bash
python power_bound_cluster.py status \
  --directory 10015/gkm_direct/shapes31/power17
```

## 6. Merge the power results once

```bash
python power_bound_cluster.py merge \
  --directory 10015/gkm_direct/shapes31/power17
```

## 7. Plot locally with the existing DGP curves

Copy `10015/gkm_direct/shapes31/power17/gkm_eigval_10015.npz` from the shared
filesystem to the corresponding local path. Reuse `10015/dgp/`; shape count
does not change the finite-sample DGP curves.

```bash
python plot_power_comparison.py --version 10015 \
  --bound 10015/gkm_direct/shapes31/power17/gkm_eigval_10015.npz \
  --output 10015/power_comparison_10015_shapes31.png
```

When comparing runs, inspect the plotted `bounds` as well as `mixture_power`
and `epsilon_grid`. The saved `diagnostics_json` also contains the fitting
residual and importance-sampling mass/ESS diagnostics. The plotted quantity
is the grid-adjusted rule's estimated power; more shapes do not by themselves
guarantee a smaller estimate or certify global optimality. The saved
`bounds_se` covers alternative Monte Carlo evaluation, not grid or fitting
uncertainty.
