"""
Likelihood function computation.
"""

import numpy as np
from scipy.special import softmax
from typing import Dict, Tuple
from ..data.data_types import LabelMapping


def softmax_transform(eta: np.ndarray) -> np.ndarray:
    """    
    Maps latent parameters eta_t = (theta_t, tau_t) to the simplex verdict distribution pi_t in Delta^2:
    Z_t = e^{theta_t} + e^{tau_t} + e^{-theta_t}
    pi_t = (e^{theta_t}/Z_t, e^{tau_t}/Z_t, e^{-theta_t}/Z_t)
    
    Args:
        eta: [theta, tau] latent parameters.
    
    Returns:
        pi: [pi_A, pi_T, pi_B] probabilities corresponding to A better than B, Tie, B better than A.
    """
    theta, tau = eta[0], eta[1]
    logits = np.array([theta, tau, -theta])
    
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    Z = np.sum(exp_logits)
    
    return exp_logits / Z


def compute_human_log_likelihood(
    eta: np.ndarray,
    human_judgments: Dict[str, str],
    C_estimates: Dict[str, np.ndarray]
) -> float:
    """    
    Model: mu_{tr} = C_r^T pi_t in Delta^2
           y_{tr}^H | eta_t, C_r ~ Categorical(mu_{tr})
    
    Log-likelihood: sum_{r in R_t} log P(y_{tr}^H | eta_t, C_r) = sum_r log(mu_{tr, y_{tr}^H})
    
    Args:
        eta: Current sample latent parameters [theta, tau].
        human_judgments: {rater_id: label}.
        C_estimates: {rater_id: C_r matrix (3x3)}.
    
    Returns:
        Log-likelihood value.
    """
    pi = softmax_transform(eta)
    log_lik = 0.0
    import pdb
    
    for rater_id, label in human_judgments.items():
        if rater_id not in C_estimates:
            continue

        C_r = C_estimates[rater_id]
        mu = C_r.T @ pi
        
        mu = np.clip(mu, 1e-10, 1.0)
        k = LabelMapping.to_index(label)
        log_lik += np.log(mu[k])

    return log_lik


def compute_llm_log_likelihood_swap(
    eta: np.ndarray,
    beta: float,
    llm_judgments: Dict[str, Tuple[str, str]],
    use_swap_logits_ba: bool = True
) -> float:
    """    
    Swap-design mechanism:
    AB order: Prompt shows answer A first (position 1), then answer B (position 2).
        - Model outputs "A" -> chooses position 1 (sample A)
        - Model outputs "B" -> chooses position 2 (sample B)
    
    BA order: Prompt shows answer B first (position 1), then answer A (position 2).
        - Model outputs "B" -> chooses position 1 (sample B)
        - Model outputs "A" -> chooses position 2 (sample A)
    
    Args:
        eta: Latent parameters [theta, tau].
        beta: LLM position bias coefficient.
        llm_judgments: {llm_id: (label_AB, label_BA)}.
        use_swap_logits_ba: Whether to use swapped-order logits_ba.
            - True (default): logits_ba = [-theta + beta, tau, theta - beta]
            - False: logits_ba = [theta - beta, tau, -theta + beta]
    
    Returns:
        Log-likelihood value.
    """
    theta, tau = eta[0], eta[1]
    log_lik = 0.0

    for llm_id, (label_ab, label_ba) in llm_judgments.items():
        logits_ab = np.array([theta + beta, tau, - theta - beta])
        probs_ab = softmax(logits_ab)
        k_ab = LabelMapping.to_index(label_ab)
        lik_ab = probs_ab[k_ab]
        if lik_ab < 1e-10:
            return -np.inf
        if use_swap_logits_ba:
            logits_ba = np.array([- theta + beta, tau, theta - beta])
            probs_ba = softmax(logits_ba)
            k_ba = LabelMapping.to_index(label_ba)
            lik_ba = probs_ba[k_ba]
            if lik_ba < 1e-10:
                return -np.inf
            log_lik += np.log(lik_ab) + np.log(lik_ba)
        else:
            log_lik += np.log(lik_ab)

    return log_lik
