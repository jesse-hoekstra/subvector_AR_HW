"""GKM conditional-CDF formulas for the smallest Wishart eigenvalue.

Closed-form approximation to the conditional distribution of kappa_p | kappa_{p-1}
when the p-2 largest kappa values are infinite. Shared by
``simulation_plot_executable.py`` (empirical-vs-GKM plot) and
``new_power_comparison.py`` (feasible conditional critical values).

Depends only on numpy/scipy -- no matplotlib and no mhg C library -- so it is
safe to import from the standalone illustrative script.
"""

from math import pi

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma, hyp1f1
from scipy.stats import chi2


def g_k1(hat_k1, k):
    """
    Compute g(hat_k1) based on the given formula.

    Parameters:
    hat_k1 (float): The input value of hat_k1.
    k (int): The parameter k in the formula.

    Returns:
    float: The value of g(hat_k1).
    """
    # Gamma function term
    gamma_term = gamma((k + 2) / 2)

    # Exponential coefficient
    coefficient = 2 ** ((k + 1) / 2)

    # Hypergeometric function
    hypergeom_term = hyp1f1((k - 1) / 2, (k + 2) / 2, -hat_k1 / 2)

    # Final g(hat_k1) formula
    g = (
        gamma_term
        * coefficient
        / (hat_k1 ** (k / 2) * np.sqrt(pi) * hypergeom_term)
    )

    return g


def conditional_density(x2, g_k1, hat_k1, k):
    """
    Compute the conditional density f*_{x2 | hat_k1}(x2 | hat_k1).

    Parameters:
    x2 (float): The value of x2 where the density is evaluated.
    g_k1 (function): A function of k1, representing g(k1) in the formula.
    hat_k1 (float): The value of hat_k1.

    Returns:
    float: The value of the conditional density at x2.
    """
    # Ensure x2 is within the range [0, hat_k1]
    if not (0 <= x2 <= hat_k1):
        return 0.0

    # Compute the density of x2 from the chi-squared distribution with k2 degrees of freedom
    f_x2 = chi2.pdf(x2, df=k-1)


    factor = (hat_k1 - x2) ** (1 / 2)

    # Multiply by g(hat_k1)
    density = f_x2 * factor * g_k1(hat_k1, k)

    return density


def get_conditional_cdf_GKM(conditional_density, g_k1, hat_k1, k):
    """
    Calculate the conditional cumulative distribution function (CDF) based on the GKM paper.

    Parameters
    ----------
    conditional_density : function
        The conditional density function f*(x2|hat_k1) from the GKM paper.
    g_k1 : function
        The g(hat_k1) function that appears in the conditional density formula.
    hat_k1 : float
        The conditioning value (observed value of the largest eigenvalue).
    k : int
        The degrees of freedom parameter.

    Returns
    -------
    tuple
        x2_values : ndarray
            Array of points where the CDF is evaluated (from 0 to hat_k1).
        cdf_values : list
            Corresponding CDF values at each point in x2_values.
    """
    # Range of x2 values (for the CDF computation)
    x2_values = np.linspace(0, hat_k1, 500)

    # Compute the CDF values
    cdf_values = []
    for x in x2_values:
        # Integrate the conditional density from 0 to x
        cdf, _ = quad(conditional_density, 0, x, args=(g_k1, hat_k1, k))
        cdf_values.append(cdf)

    return x2_values, cdf_values


def critical_value(hat_k1, alpha, k):
    """Conditional critical value: the x with GKM CDF(x | hat_k1) = 1 - alpha."""
    target = 1.0 - alpha
    def cdf_minus_target(x):
        cdf, _ = quad(conditional_density, 0, x, args=(g_k1, hat_k1, k))
        return cdf - target
    return brentq(cdf_minus_target, 0, hat_k1)
