import matplotlib.pyplot as plt
import numpy as np

from gkm import conditional_density, g_k1, get_conditional_cdf_GKM

def get_mu_list(start, end, p, middle_value=2):
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

def simulate_joint_eigenvalues(p, k, mu, num_simulations=1000):
    """
    Simulates the joint distribution of the smallest and second smallest eigenvalues
    of a noncentral real Wishart matrix.
    
    Parameters:
    p (int): Dimensionality of the Wishart matrix.
    k (int): Number of  instruments.
    mu (ndarray): Singular values of M.
    num_simulations (int): Number of simulations.
    
    Returns:
    joint_eigenvalues (ndarray): Array of joint eigenvalues (smallest, second smallest).
    """
    M = np.vstack([np.zeros((k-p, p)), np.diag(mu)])  # Diagonal matrix with sqrt(kappa) as entries
    joint_eigenvalues = []

    for _ in range(num_simulations):
        X = np.random.randn(k, p)
        X += M
        W = X.T @ X  # Compute the Wishart matrix
        eigvals = np.sort(np.linalg.eigvalsh(W))  # Sort eigenvalues in ascending order
        joint_eigenvalues.append([eigvals[0], eigvals[1]])  # Store smallest and second smallest

    return np.array(joint_eigenvalues)

def main():
    try:
        # Get user input
        p = int(input("Enter a p>2:"))
        k = int(input("Enter the number of instruments k: "))
        num_simulations_marginal = 100000
        num_simulations_conditional = 1000000
        if not (p > 2):
            raise ValueError("p must be larger than 2")
        if (k < p):
            raise ValueError("k must be a positive integer bigger than p.")

        #simulation to get conditioning value
        kappa_2_hats = simulate_joint_eigenvalues(p,k, get_mu_list(2,0,p), num_simulations_marginal)[:,1]
        kappa_2_hat = round(np.median(kappa_2_hats))

        #conditional simulation
        bin_width = 0.1
        mu_values = [get_mu_list(2,0,p), get_mu_list(5,0,p), get_mu_list(10,0,p)] 

        for mu in mu_values:
            # Simulate the joint eigenvalues
            joint_eigenvalues = simulate_joint_eigenvalues(p, k, mu, num_simulations_conditional)
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
            f'$\\kappa = ({", ".join(f"{x**2:.3g}" for x in mu)})$')

        #gkm conditional cdf
        x2_values, conditional_cdf_GKM = get_conditional_cdf_GKM(conditional_density, g_k1, kappa_2_hat, k) 

        plt.plot(x2_values, conditional_cdf_GKM, 
         label=fr'(approx) CDF $\hat{{\kappa}}_{p}|\hat{{\kappa}}_{p-1}$ = {kappa_2_hat}, ' \
        fr'$\kappa_{p-2} = (\infty)$')

        # Finalize the plot
        plt.title(fr'Conditional CDF $\hat{{\kappa}}_{p}|\hat{{\kappa}}_{p-1}$ for $k={k}$ and Different Values $\kappa$')
        plt.xlabel(fr'$\hat{{\kappa}}_{p}$')
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.grid()
        plt.show()

    except ValueError as e:
        print(f"Input error: {e}")

if __name__ == "__main__":
    main()