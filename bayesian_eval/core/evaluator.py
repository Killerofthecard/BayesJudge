
from scipy.special import logsumexp
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from functools import partial

from ..data.data_types import SampleData, LabelMapping
from .likelihood import (
    softmax_transform,
    compute_human_log_likelihood,
    compute_llm_log_likelihood_swap
)
from .particle_filter import (
    systematic_resampling,
    compute_ess,
    mcmc_move_fast,
    weighted_statistics
)
from .diagnostics import DiagnosticsPrinter, generate_learning_report


class StrictBayesianEvaluator:

    def __init__(
        self,
        n_particles: int = 1000,
        lambda_diag: float = 2.0,
        lambda_off: float = 1.0,
        sigma_beta: float = 1.0,
        mu_eta: Optional[np.ndarray] = None,
        Sigma_eta: Optional[np.ndarray] = None,
        ess_threshold_factor: float = 0.5,
        mcmc_step_size: float = 0.5,
        verbose: bool = False,
        max_print_raters: int = 2,
        print_interval: int = 1,
        update_human_params: bool = True,
        update_llm_params: bool = True,
        use_swap_logits_ba: bool = True,
        zero_human_likelihood: bool = False,
        zero_llm_likelihood: bool = False,
        human_priors: Optional[Dict[str, np.ndarray]] = None
    ):
        """        
        Args:
            n_particles: Number of particles M.
            lambda_diag: Diagonal parameter of the Dirichlet prior.
            lambda_off: Off-diagonal parameter of the Dirichlet prior.
            sigma_beta: Prior standard deviation of the LLM position bias beta.
            mu_eta: Prior mean of eta_t, mu_0.
            Sigma_eta: Prior covariance of eta_t, Sigma_0.
            ess_threshold_factor: ESS resampling threshold factor.
            mcmc_step_size: Initial MCMC move step size.
            verbose: Whether to enable detailed diagnostic output.
            max_print_raters: Maximum number of raters to display in a single print.
            print_interval: Print diagnostics every N samples.
            update_human_params: Whether to update the human confusion matrix C_r.
            update_llm_params: Whether to update the LLM bias parameter beta.
            use_swap_logits_ba: Whether to use swapped-order logits_ba for LLM likelihood.
            zero_human_likelihood: Whether to set human log-likelihood to 0 (for ablation).
            zero_llm_likelihood: Whether to set LLM log-likelihood to 0 (for ablation).
            human_priors: Custom prior confusion matrices for human raters {rater_id: 3x3 numpy array}.
        """
        self.update_human_params = update_human_params
        self.update_llm_params = update_llm_params
        self.use_swap_logits_ba = use_swap_logits_ba
        self.zero_human_likelihood = zero_human_likelihood
        self.zero_llm_likelihood = zero_llm_likelihood
        self.human_priors = human_priors if human_priors is not None else {}
        self.M = n_particles
        self.lambda_diag = lambda_diag
        self.lambda_off = lambda_off
        self.sigma_beta = sigma_beta
        self.ess_threshold = n_particles * ess_threshold_factor
        self.epsilon_0 = mcmc_step_size

        self.mu_eta = mu_eta if mu_eta is not None else np.array([0.0, 0.0])
        self.Sigma_eta = Sigma_eta if Sigma_eta is not None else np.eye(2)
        self.inv_Sigma_eta = np.linalg.inv(self.Sigma_eta)

        self.beta_particles: Dict[str, np.ndarray] = {}
        self.weights: Dict[str, np.ndarray] = {}

        self.C_params: Dict[str, np.ndarray] = {}

        self.t = 0

        self.beta_history: Dict[str, List] = defaultdict(list)
        self.ess_history: Dict[str, List] = defaultdict(list)
        self.pi_history: List[np.ndarray] = []
        self.variance_history: Dict[str, List] = defaultdict(list)

        self._current_eta: Optional[np.ndarray] = None

        self.diagnostics = DiagnosticsPrinter(max_print_raters, print_interval)
        self.verbose = verbose

    def _init_llm_particles(self, llm_id: str):
        """        
        Initialization (t=0): beta_{j,0}^{(m)} ~ i.i.d. N(0, sigma_beta^2), w_{j,0}^{(m)} = 1/M.
        """
        if llm_id not in self.beta_particles:
            self.beta_particles[llm_id] = np.random.normal(
                0, self.sigma_beta, self.M
            )
            self.weights[llm_id] = np.ones(self.M) / self.M

    def _init_rater_params(self, rater_id: str):
        """
        Initialize Dirichlet parameters for a rater (prior distribution).
        """
        if rater_id not in self.C_params:
            if rater_id in self.human_priors:
                self.C_params[rater_id] = self.human_priors[rater_id].copy()
            else:
                alpha_mat = np.full((3, 3), self.lambda_off)
                np.fill_diagonal(alpha_mat, self.lambda_diag)
                self.C_params[rater_id] = alpha_mat

    def _get_prior_C_estimate(self, rater_id: str) -> np.ndarray:
        """Return the prior C_r estimate for a rater."""
        if rater_id in self.human_priors:
            alpha_mat = self.human_priors[rater_id].copy()
        else:
            alpha_mat = np.full((3, 3), self.lambda_off)
            np.fill_diagonal(alpha_mat, self.lambda_diag)
        row_sums = alpha_mat.sum(axis=1, keepdims=True)
        return alpha_mat / row_sums

    def _get_C_estimates(self, human_judgments: Optional[Dict[str, str]] = None) -> Dict[str, np.ndarray]:
        """
        Return the posterior mean estimates of C_r.
        
        Dirichlet posterior mean: E[C_r[k,l]] = alpha_{r,k,l} / sum_{l'} alpha_{r,k,l'}.
        """
        C_estimates = {}

        for rater_id, alpha_mat in self.C_params.items():
            row_sums = alpha_mat.sum(axis=1, keepdims=True)
            C_estimates[rater_id] = alpha_mat / row_sums

        if human_judgments is not None:
            for rater_id in human_judgments.keys():
                if rater_id not in C_estimates:
                    C_estimates[rater_id] = self._get_prior_C_estimate(rater_id)

        return C_estimates

    def _local_map_estimate(
        self,
        human_judgments: Dict[str, str],
        llm_judgments: Dict[str, Tuple[str, str]],
        beta_fixed: Union[float, Dict[str, float]],
        C_estimates: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """        
        Args:
            beta_fixed: A single float (backward compatibility) or Dict[str, float] (per-LLM independent beta).
        """
        def neg_log_posterior(eta):
            diff = eta - self.mu_eta
            log_prior = -0.5 * (diff @ self.inv_Sigma_eta @ diff)


            if self.zero_human_likelihood:
                log_lik_human = 0.0
            else:
                log_lik_human = compute_human_log_likelihood(eta, human_judgments, C_estimates)
            
            if self.zero_llm_likelihood:
                log_lik_llm = 0.0
            else:
                if isinstance(beta_fixed, dict):
                    log_lik_llm = 0.0
                    for lid, (y_ab, y_ba) in llm_judgments.items():
                        if lid in beta_fixed:
                            log_lik_llm += compute_llm_log_likelihood_swap(
                                eta, beta_fixed[lid], {lid: (y_ab, y_ba)},
                                use_swap_logits_ba=self.use_swap_logits_ba
                            )
                else:
                    log_lik_llm = compute_llm_log_likelihood_swap(
                        eta, beta_fixed, llm_judgments,
                        use_swap_logits_ba=self.use_swap_logits_ba
                    )
            
            return -(log_prior + log_lik_human + log_lik_llm)

        result = minimize(neg_log_posterior, self.mu_eta, method='L-BFGS-B')
        return result.x

    def _update_C_params(self, human_judgments: Dict[str, str], pi_current: np.ndarray):
        """
        Online cumulative update of the human confusion matrix C_r.
        """
        
        for rater_id, label in human_judgments.items():
            self._init_rater_params(rater_id)
            observed_k = LabelMapping.to_index(label)

            k_star = np.argmax(pi_current)
            self.C_params[rater_id][k_star, observed_k] += 1


