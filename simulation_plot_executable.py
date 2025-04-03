import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma, hyp1f1
from math import pi
from scipy.stats import chi2
from scipy.integrate import quad

#GKM formula begin
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
#GKM formula end

def get_omega_list(start, end, p, middle_value=2):
    """
    Creates a list of p equispaced points under a square root, from sqrt(start) to sqrt(end),
    ensuring sqrt(middle_value) is included.

    Parameters
    ----------
    start : float
        Starting value before square root
    end : float
        Ending value before square root
    p : int
        Number of points (must be > 2)
    middle_value : float, optional
        Value that must be included (default=2)

    Returns
    -------
    ndarray
        Array of p equispaced points after taking square root
    """
    if p <= 2:
        raise ValueError("p must be greater than 2")
        
    if p % 2 == 0:  # even number of points
        first_half_points = p // 2
        second_half_points = first_half_points + 1
    else:  # odd number of points
        first_half_points = (p + 1) // 2
        second_half_points = first_half_points

    first_half = np.sqrt(np.linspace(start, middle_value, first_half_points))
    second_half = np.sqrt(np.linspace(middle_value, end, second_half_points)[1:])
    return np.concatenate([first_half, second_half])

def simulate_joint_eigenvalues(p, n, omega, num_simulations=1000):
    """
    Simulates the joint distribution of the smallest and second smallest eigenvalues
    of a noncentral real Wishart matrix.
    
    Parameters:
    p (int): Dimensionality of the Wishart matrix.
    n (int): Degrees of freedom (number of samples).
    omega (ndarray): Eigenvalues of M^T M (noncentrality parameters).
    num_simulations (int): Number of simulations.
    
    Returns:
    joint_eigenvalues (ndarray): Array of joint eigenvalues (smallest, second smallest).
    """
    M = np.vstack([np.zeros((n-p, p)), np.diag(omega)])  # Diagonal matrix with sqrt(omega) as entries
    joint_eigenvalues = []

    for _ in range(num_simulations):
        X = np.random.randn(n, p)
        X += M
        W = X.T @ X  # Compute the Wishart matrix
        eigvals = np.sort(np.linalg.eigvalsh(W))  # Sort eigenvalues in ascending order
        joint_eigenvalues.append([eigvals[0], eigvals[1]])  # Store smallest and second smallest

    return np.array(joint_eigenvalues)

def main():
    try:
        # Get user input
        p = int(input("Enter a p>2:"))
        n = int(input("Enter the number of instruments n: "))
        num_simulations_marginal = 100000
        num_simulations_conditional = 1000000
        if not (p > 2):
            raise ValueError("p must be larger than 2")
        if (n < p):
            raise ValueError("n must be a positive integer bigger than p.")

        #simulation to get conditioning value
        kappa_2_hats = simulate_joint_eigenvalues(p,n, get_omega_list(2,0,p), num_simulations_marginal)[:,1]
        kappa_2_hat = round(np.median(kappa_2_hats))

        #conditional simulation
        bin_width = 0.1
        omega_values = [get_omega_list(2,0,p), get_omega_list(5,0,p), get_omega_list(10,0,p)] 

        for omega in omega_values:
            # Simulate the joint eigenvalues
            joint_eigenvalues = simulate_joint_eigenvalues(p, n, omega, num_simulations_conditional)
            smallest, second_smallest = joint_eigenvalues[:, 0], joint_eigenvalues[:, 1]

            # Filter smallest eigenvalues conditioned on second smallest eigenvalue
            condition_mask = (second_smallest > kappa_2_hat - bin_width) & \
                            (second_smallest < kappa_2_hat + bin_width)
            conditioned_smallest = smallest[condition_mask]

            # Compute the empirical CDF of the smallest eigenvalue
            conditioned_smallest_sorted = np.sort(conditioned_smallest)
            empirical_cdf = np.arange(1, len(conditioned_smallest_sorted) + 1) / len(conditioned_smallest_sorted)

            plt.plot(conditioned_smallest_sorted, empirical_cdf, 
             label=f'(sim) CDF $\\hat{{\\kappa}}_{p}|\\hat{{\\kappa}}_{p-1}$ = {kappa_2_hat}, ' \
            f'$\\kappa = ({", ".join(f"{x**2:.3g}" for x in omega)})$')

        #gkm conditional cdf
        x2_values, conditional_cdf_GKM = get_conditional_cdf_GKM(conditional_density, g_k1, kappa_2_hat, n) 

        plt.plot(x2_values, conditional_cdf_GKM, 
         label=fr'(approx) CDF $\hat{{\kappa}}_{p}|\hat{{\kappa}}_{p-1}$ = {kappa_2_hat}, ' \
        fr'$\kappa_{p-2} = (\infty)$')

        # Finalize the plot
        plt.title(fr'Conditional CDF $\hat{{\kappa}}_{p}|\hat{{\kappa}}_{p-1}$ for $k={n}$ and Different Values $\kappa$')
        plt.xlabel(fr'$\hat{{\kappa}}_{p}$')
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.grid()
        plt.show()

    except ValueError as e:
        print(f"Input error: {e}")

if __name__ == "__main__":
    main()