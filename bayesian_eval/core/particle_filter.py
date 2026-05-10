"""
Particle filter algorithms.
"""

import numpy as np
from typing import Dict, Tuple, Callable


def systematic_resampling(weights: np.ndarray) -> np.ndarray:
    """
    Systematic resampling algorithm.
    
    Lower variance than multinomial resampling; implements the systematic resampling
    
    Args:
        weights: Normalized weight array.
    
    Returns:
        Resampled index array.
    """
    M = len(weights)
    indices = np.zeros(M, dtype=int)
    cumsum = np.cumsum(weights)

    u0 = np.random.uniform(0, 1.0 / M)
    u = u0 + np.arange(M) / M

    j = 0
    for i in range(M):
        while j < M - 1 and u[i] > cumsum[j]:
            j += 1
        indices[i] = j

    return indices


def compute_ess(weights: np.ndarray) -> float:
    """
    Compute the effective sample size (ESS).
    
    ESS = 1 / sum(w_i^2)
    
    Args:
        weights: Normalized weight array.
    
    Returns:
        ESS value.
    """
    return 1.0 / np.sum(weights ** 2)


def mcmc_move_fast(
    particles: np.ndarray,
    weights: np.ndarray,
    eta_map: np.ndarray,
    human_judgments: Dict,
    llm_judgments: Dict,
    llm_id: str,
    sigma_beta: float,
    epsilon: float,
    likelihood_fn,
    human_likelihood_fn
) -> np.ndarray:
    """    
    Moves particles using the Metropolis-Hastings algorithm to prevent particle degeneracy.
    
    Args:
        particles: Current particle array.
        weights: Current weight array.
        eta_map: MAP estimate of eta_t.
        human_judgments: Human judgments.
        llm_judgments: LLM judgments.
        llm_id: ID of the current LLM.
        sigma_beta: Prior standard deviation of beta.
        epsilon: MCMC step size.
        likelihood_fn: LLM likelihood function.
        human_likelihood_fn: Human likelihood function.
    
    Returns:
        Moved particle array.
    """
    new_particles = particles.copy()
    M = len(particles)
    
    log_lik_human_human = human_likelihood_fn(eta_map, human_judgments)
    
    for m in range(M):
        beta_star = np.random.normal(particles[m], epsilon)

        log_lik_curr = log_lik_human_human + likelihood_fn(
            eta_map, particles[m], {llm_id: llm_judgments[llm_id]}
        )
        log_lik_star = log_lik_human_human + likelihood_fn(
            eta_map, beta_star, {llm_id: llm_judgments[llm_id]}
        )

        log_prior_ratio = -0.5 * (beta_star ** 2 - particles[m] ** 2) / (sigma_beta ** 2)

        log_alpha = log_lik_star - log_lik_curr + log_prior_ratio
        alpha = np.exp(min(0, log_alpha))

        if np.random.random() < alpha:
            new_particles[m] = beta_star

    return new_particles


def mcmc_move_slow(
    particles: np.ndarray,
    weights: np.ndarray,
    eta_map: np.ndarray,
    human_judgments: Dict,
    llm_judgments: Dict,
    llm_id: str,
    sigma_beta: float,
    epsilon: float,
    likelihood_fn: Callable,
    human_likelihood_fn: Callable,
    mu_eta: np.ndarray,
    inv_Sigma_eta: np.ndarray,
    all_llm_ids: list,
    beta_particles_dict: Dict[str, np.ndarray],
    llm_idx_in_dict: int,
    max_refine_steps: int = 3,
    refine_lr: float = 0.05,
) -> np.ndarray:
    """    
    Key change: for each proposed beta*, optionally re-infer the corresponding eta*
    instead of fixing eta_map.
    
    Args:
        particles: Current particle array.
        weights: Current weight array.
        eta_map: MAP estimate of eta_t.
        human_judgments: Human judgments.
        llm_judgments: LLM judgments.
        llm_id: ID of the current LLM.
        sigma_beta: Prior standard deviation of beta.
        epsilon: MCMC step size.
        likelihood_fn: LLM likelihood function.
        human_likelihood_fn: Human likelihood function.
        mu_eta: Prior mean of eta.
        inv_Sigma_eta: Inverse of the prior covariance of eta.
        all_llm_ids: List of IDs of all LLMs.
        beta_particles_dict: Particle dictionary for all LLMs {llm_id: (M,)}.
        llm_idx_in_dict: Index of the current LLM in all_llm_ids.
        max_refine_steps: Number of fast refinement steps.
            - 0: fully fixed eta_map (original behavior, fastest).
            - 1-3: quick gradient adjustment from eta_map (balanced).
            - -1: full MAP re-inference (slowest but most accurate).
        refine_lr: Learning rate for fast refinement.
    
    Returns:
        Moved particle array.
    """
    new_particles = particles.copy()
    M = len(particles)
    
    def build_beta_dict(current_m: int, current_beta: float) -> Dict[str, float]:
        """Build the full beta_dict: current LLM uses current_beta, others use particle m's value."""
        beta_dict = {}
        for i, lid in enumerate(all_llm_ids):
            if i == llm_idx_in_dict:
                beta_dict[lid] = current_beta
            else:
                beta_dict[lid] = beta_particles_dict[lid][current_m]
        return beta_dict
    
    def quick_refine_eta(eta_init: np.ndarray, beta_dict: Dict[str, float], 
                        steps: int, lr: float) -> np.ndarray:
        """Quick gradient adjustment of eta from eta_init to adapt to a new beta configuration."""
        eta = eta_init.copy()
        
        for _ in range(steps):
            grad = np.zeros(2)
            h = 1e-4
            
            for dim in range(2):
                eta_plus = eta.copy()
                eta_minus = eta.copy()
                eta_plus[dim] += h
                eta_minus[dim] -= h
                
                def obj(et):
                    diff = et - mu_eta
                    log_prior = -0.5 * (diff @ inv_Sigma_eta @ diff)
                    log_lik_human = human_likelihood_fn(et, human_judgments)
                    
                    log_lik_llm = 0.0
                    for lid, (y_ab, y_ba) in llm_judgments.items():
                        if lid in beta_dict:
                            log_lik_llm += likelihood_fn(
                                et, beta_dict[lid], {lid: (y_ab, y_ba)}
                            )
                    return -(log_prior + log_lik_human + log_lik_llm)
                
                grad[dim] = (obj(eta_plus) - obj(eta_minus)) / (2 * h)
            
            eta = eta - lr * grad
        
        return eta
    
    def full_map_eta(beta_dict: Dict[str, float]) -> np.ndarray:
        """Invoke full MAP optimization."""
        def neg_log_posterior(eta):
            diff = eta - mu_eta
            log_prior = -0.5 * (diff @ inv_Sigma_eta @ diff)
            log_lik_human = human_likelihood_fn(eta, human_judgments)
            
            log_lik_llm = 0.0
            for lid, (y_ab, y_ba) in llm_judgments.items():
                if lid in beta_dict:
                    log_lik_llm += likelihood_fn(
                        eta, beta_dict[lid], {lid: (y_ab, y_ba)}
                    )
            return -(log_prior + log_lik_human + log_lik_llm)
        
        result = minimize(neg_log_posterior, mu_eta, method="L-BFGS-B")
        return result.x
    
    if max_refine_steps == 0:
        get_eta_for_beta = lambda m, beta: eta_map
        eta_strategy = "fixed"
    elif max_refine_steps == -1:
        get_eta_for_beta = lambda m, beta: full_map_eta(build_beta_dict(m, beta))
        eta_strategy = "full_map"
    else:
        get_eta_for_beta = lambda m, beta: quick_refine_eta(
            eta_map, build_beta_dict(m, beta), max_refine_steps, refine_lr
        )
        eta_strategy = f"quick_refine_{max_refine_steps}"
    
    for m in range(M):
        beta_m = particles[m]
        
        beta_star = np.random.normal(beta_m, epsilon)
        
        eta_m = get_eta_for_beta(m, beta_m)
        eta_star = get_eta_for_beta(m, beta_star)
        
        log_lik_curr = human_likelihood_fn(eta_m, human_judgments) + \
                       likelihood_fn(eta_m, beta_m, {llm_id: llm_judgments[llm_id]})
        
        log_lik_star = human_likelihood_fn(eta_star, human_judgments) + \
                       likelihood_fn(eta_star, beta_star, {llm_id: llm_judgments[llm_id]})
        
        log_prior_ratio = -0.5 * (beta_star**2 - beta_m**2) / (sigma_beta**2)
        
        log_alpha = log_lik_star - log_lik_curr + log_prior_ratio
        alpha = np.exp(min(0, log_alpha))
        
        if np.random.random() < alpha:
            new_particles[m] = beta_star
    
    return new_particles



def weighted_statistics(particles: np.ndarray, weights: np.ndarray) -> Dict:
    """
    Compute weighted statistics of particles.
    
    Returns:
        {
            "mean": weighted mean,
            "variance": weighted variance,
            "ci_95": 95% credible interval,
            "ess": effective sample size
        }
    """
    mean = np.average(particles, weights=weights)
    
    variance = np.average((particles - mean) ** 2, weights=weights)
    
    sorted_idx = np.argsort(particles)
    sorted_p = particles[sorted_idx]
    sorted_w = weights[sorted_idx]
    cumsum_w = np.cumsum(sorted_w)

    idx_lower = np.searchsorted(cumsum_w, 0.025)
    idx_upper = np.searchsorted(cumsum_w, 0.975)
    ci_lower = sorted_p[idx_lower]
    ci_upper = sorted_p[idx_upper]
    
    ess = 1.0 / np.sum(weights ** 2)

    return {
        "mean": float(mean),
        "variance": float(variance),
        "std": float(np.sqrt(variance)),
        "ci_95": (float(ci_lower), float(ci_upper)),
        "ess": float(ess),
        "converged": bool(variance < 0.01)
    }
