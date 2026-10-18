# Research code accompanying [Best Feasible Conditional Critical Values for a More Powerful Subvector Anderson-Rubin Test](https://arxiv.org/pdf/2601.17843). 

## The repository contains power comparisons, power bound calculation and the Vietnam empirical analysis.

| File | Purpose |
|---|---|
| `empirical_Vietnam.py` | Vietnam empirical analysis and confidence intervals. |
| `simulation_plot_executable.py` | Interactive illustration of conditional eigenvalue distributions. |
| `gkm.py` | Conditional eigenvalue distributions and critical values. |
| `new_power_comparison.py` | Simulate power of the different tests. |
| `alfd_eigval.py` | Calculate power bounds for two or three nuisance parameters. |
| `plot_power_comparison.py` | Combine saved power curves and bounds into a figure. |

`alfd_eigval.py` is a direct extension of the power-bound calculations and MATLAB
implementation of Guggenberger, Kleibergen and Mavroeidis (2019, GKM) to two or
three nuisance parameters. See [Supplement Section D.3.2][gkm-supp] and the
accompanying [MATLAB code][gkm-code].

The matrix-hypergeometric calculations use Plamen Koev's MHG code, included in
`koev/` ([Koev and Edelman, 2006][koev-paper]; [original licence][koev-licence]).

With Python 3.10 or later, set up the environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
sh koev/mhg15/build.sh
```

The final command is needed only for the power bound and requires a C compiler
on macOS or Linux. Use `--help` for simulation, bound and plotting options.
The Vietnam analysis requires the original replication files in `data/Vietnam/`;
these are not included.

[gkm-supp]: https://www.econometricsociety.org/publications/quantitative-economics/2019/05/01/A-more-powerful-subvector-Anderson-Rubin-test-in-linear-instrumental-variables-regression/supp/666-3193-1-SP.pdf
[gkm-code]: https://www.econometricsociety.org/publications/quantitative-economics/2019/05/01/A-more-powerful-subvector-Anderson-Rubin-test-in-linear-instrumental-variables-regression/supp/666-3195-1-SP.zip
[koev-paper]: https://math.mit.edu/~edelman/publications/efficient_evaluation.pdf
[koev-licence]: koev/mhg15/COPYRIGHT.TXT
