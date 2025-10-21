"""
Experimental Rasch-based vulnerability model for sparse TVD-MI data.
Attempts to fit Item Response Theory (IRT) models to predict unobserved interactions.
WARNING: Model convergence and prediction accuracy have NOT been validated on real data.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.optimize import minimize
from scipy.special import expit, logit
import json
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fit_rasch_additive(S: np.ndarray, mask: np.ndarray,
                       lambda_a: float = 1e-4, lambda_b: float = 1e-4,
                       max_iter: int = 100, tol: float = 1e-4) -> Dict[str, Any]:
    """
    Fit Rasch model using alternating least squares (ALS).
    Based on sparse-tvd-irt implementation for robust IRT fitting.

    Args:
        S: Sparse TVD-MI matrix
        mask: Boolean mask of observed positions
        lambda_a: L2 regularization for attack severity
        lambda_b: L2 regularization for agent resistance
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Dictionary with fitted parameters
    """
    K, J = S.shape  # K attacks, J agents

    # Initialize parameters with small random values
    np.random.seed(42)  # For reproducibility
    a = np.random.randn(K) * 0.1  # Attack severity
    b = np.random.randn(J) * 0.1  # Agent resistance

    converged = False

    for iteration in range(max_iter):
        a_old, b_old = a.copy(), b.copy()

        # Update a (attack severity) - holding b fixed
        for i in range(K):
            mask_i = mask[i]
            n_obs = mask_i.sum()
            if n_obs > 0:
                # Observed values for attack i
                S_i = S[i, mask_i]
                b_i = b[mask_i]

                # Least squares update with regularization
                numerator = np.sum(S_i - b_i)
                denominator = n_obs + lambda_a
                a[i] = numerator / denominator

        # Update b (agent resistance) - holding a fixed
        for j in range(J):
            mask_j = mask[:, j]
            n_obs = mask_j.sum()
            if n_obs > 0:
                # Observed values for agent j
                S_j = S[mask_j, j]
                a_j = a[mask_j]

                # Least squares update with regularization
                numerator = np.sum(S_j - a_j)
                denominator = n_obs + lambda_b
                b[j] = numerator / denominator

        # Gauge fixing - center parameters to avoid drift
        mean_a = np.mean(a)
        a = a - mean_a
        b = b + mean_a  # Preserve predictions

        # Check convergence
        change_a = np.linalg.norm(a - a_old)
        change_b = np.linalg.norm(b - b_old)

        if change_a + change_b < tol:
            converged = True
            logger.info(f"Rasch fitting converged at iteration {iteration + 1}")
            break

    if not converged:
        logger.warning(f"Rasch fitting did not converge after {max_iter} iterations")

    # Compute predictions for all positions
    pred = np.outer(a, np.ones(J)) + np.outer(np.ones(K), b)

    # Compute fit statistics
    observed_values = S[mask]
    predicted_values = pred[mask]
    mse = np.mean((observed_values - predicted_values) ** 2)

    return {
        'a': a,  # Attack severity parameters
        'b': b,  # Agent resistance parameters
        'pred': pred,  # Full prediction matrix
        'iterations': iteration + 1,
        'converged': converged,
        'mse': mse
    }


@dataclass
class RaschParameters:
    """Container for fitted Rasch model parameters."""
    attack_severity: np.ndarray  # Attack difficulty parameters (a)
    agent_resistance: np.ndarray  # Agent ability parameters (b)
    global_discrimination: float  # Global discrimination parameter
    log_likelihood: float  # Final log-likelihood
    convergence: bool  # Whether fitting converged


class VulnerabilityModel:
    """
    Rasch-based vulnerability model for sparse evaluation data.
    Models P(vulnerability) = logistic(attack_severity - agent_resistance).
    """

    def __init__(self, tvd_matrix: np.ndarray, sparse_mask: np.ndarray):
        """
        Initialize the vulnerability model.

        Args:
            tvd_matrix: Sparse TVD-MI matrix (may contain zeros for unobserved)
            sparse_mask: Boolean mask indicating observed positions
        """
        self.S = tvd_matrix
        self.mask = sparse_mask
        self.num_attacks, self.num_agents = tvd_matrix.shape
        self.observed_indices = np.argwhere(sparse_mask)

        # Model parameters (to be fitted)
        self.fit_result = None
        self.attack_severity = None
        self.agent_resistance = None
        self.predictions = None

        logger.info(f"Initialized vulnerability model with {np.sum(mask)} observations "
                   f"out of {mask.size} possible ({np.mean(mask):.1%} coverage)")

    def fit(self, max_iter: int = 100, tol: float = 1e-4,
           regularization: float = 0.01) -> 'VulnerabilityModel':
        """
        Fit Rasch model using alternating least squares (ALS).

        Args:
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            regularization: L2 regularization strength

        Returns:
            Self (fitted model)
        """
        logger.info("Fitting Rasch vulnerability model using ALS...")

        # Use the robust ALS fitting function
        fit_result = fit_rasch_additive(
            S=self.S,
            mask=self.mask,
            lambda_a=regularization,
            lambda_b=regularization,
            max_iter=max_iter,
            tol=tol
        )

        # Extract fitted parameters
        self.attack_severity = fit_result['a']
        self.agent_resistance = fit_result['b']
        self.predictions = fit_result['pred']

        # Compute log-likelihood for fit quality assessment
        observed_values = self.S[self.mask]
        predicted_values = self.predictions[self.mask]

        # For continuous TVD-MI values in [0,1], use beta-like likelihood
        # Clip predictions to avoid log(0)
        pred_clipped = np.clip(predicted_values, 1e-10, 1 - 1e-10)
        log_likelihood = np.sum(
            observed_values * np.log(pred_clipped) +
            (1 - observed_values) * np.log(1 - pred_clipped)
        )

        # Store fit results
        self.fit_result = RaschParameters(
            attack_severity=self.attack_severity,
            agent_resistance=self.agent_resistance,
            global_discrimination=1.0,  # Fixed in basic Rasch model
            log_likelihood=log_likelihood,
            convergence=fit_result['converged']
        )

        logger.info(f"Fitting complete. Converged: {fit_result['converged']}, "
                   f"Iterations: {fit_result['iterations']}, "
                   f"MSE: {fit_result['mse']:.4f}")

        return self

    def _compute_predictions(self) -> np.ndarray:
        """
        Compute full prediction matrix using fitted parameters.

        Returns:
            Predicted vulnerability matrix
        """
        predictions = np.zeros((self.num_attacks, self.num_agents))

        for i in range(self.num_attacks):
            for j in range(self.num_agents):
                # Rasch model prediction
                predictions[i, j] = expit(self.attack_severity[i] - self.agent_resistance[j])

        return predictions

    def predict_vulnerability(self, attack_idx: int, agent_idx: int) -> float:
        """
        Predict vulnerability for a specific attack-agent pair.

        Args:
            attack_idx: Attack index
            agent_idx: Agent index

        Returns:
            Predicted vulnerability score
        """
        if self.predictions is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return float(self.predictions[attack_idx, agent_idx])

    def rank_agents(self) -> List[int]:
        """
        Rank agents by resistance (most resistant first).

        Returns:
            List of agent indices sorted by resistance
        """
        if self.agent_resistance is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Higher resistance = less vulnerable
        return np.argsort(-self.agent_resistance).tolist()

    def rank_attacks(self) -> List[int]:
        """
        Rank attacks by severity (most severe first).

        Returns:
            List of attack indices sorted by severity
        """
        if self.attack_severity is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Higher severity = more effective
        return np.argsort(-self.attack_severity).tolist()

    def get_vulnerability_scores(self) -> Dict[str, np.ndarray]:
        """
        Get comprehensive vulnerability scores.

        Returns:
            Dictionary with agent and attack vulnerability scores
        """
        if self.predictions is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return {
            'agent_vulnerability': np.mean(self.predictions, axis=0),  # Mean across attacks
            'attack_effectiveness': np.mean(self.predictions, axis=1),  # Mean across agents
            'max_vulnerabilities': np.max(self.predictions, axis=0),    # Worst-case per agent
            'max_effectiveness': np.max(self.predictions, axis=1)       # Best-case per attack
        }

    def bootstrap_rankings(self, n_bootstrap: int = 500, seed: int = 42) -> Dict[str, Any]:
        """
        Bootstrap confidence intervals for rankings.

        Args:
            n_bootstrap: Number of bootstrap samples
            seed: Random seed for reproducibility

        Returns:
            Dictionary with confidence intervals and stability metrics
        """
        if self.fit_result is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        logger.info(f"Running bootstrap with {n_bootstrap} iterations...")

        rng = np.random.default_rng(seed)
        agent_ranks_bootstrap = []
        attack_ranks_bootstrap = []
        agent_scores_bootstrap = []
        attack_scores_bootstrap = []

        # Get observed positions
        observed_positions = self.observed_indices
        n_observed = len(observed_positions)

        for iteration in range(n_bootstrap):
            # Resample observed positions with replacement
            boot_indices = rng.choice(n_observed, size=n_observed, replace=True)
            boot_positions = observed_positions[boot_indices]

            # Create bootstrap mask and matrix
            boot_mask = np.zeros_like(self.mask)
            boot_matrix = np.zeros_like(self.S)

            for i, j in boot_positions:
                boot_mask[i, j] = True
                boot_matrix[i, j] = self.S[i, j]

            # Fit model on bootstrap sample
            boot_model = VulnerabilityModel(boot_matrix, boot_mask)
            try:
                boot_model.fit(max_iter=50)  # Fewer iterations for speed

                # Store rankings
                agent_ranks_bootstrap.append(boot_model.rank_agents())
                attack_ranks_bootstrap.append(boot_model.rank_attacks())
                agent_scores_bootstrap.append(boot_model.agent_resistance)
                attack_scores_bootstrap.append(boot_model.attack_severity)
            except Exception as e:
                logger.warning(f"Bootstrap iteration {iteration} failed: {e}")
                continue

        # Compute confidence intervals
        agent_ranks_array = np.array(agent_ranks_bootstrap)
        attack_ranks_array = np.array(attack_ranks_bootstrap)
        agent_scores_array = np.array(agent_scores_bootstrap)
        attack_scores_array = np.array(attack_scores_bootstrap)

        # Rank stability (how consistent are the rankings)
        agent_rank_stability = []
        for j in range(self.num_agents):
            positions = agent_ranks_array[:, j] if agent_ranks_array.size > 0 else []
            if len(positions) > 0:
                agent_rank_stability.append(np.std(positions))
            else:
                agent_rank_stability.append(float('inf'))

        attack_rank_stability = []
        for i in range(self.num_attacks):
            positions = attack_ranks_array[:, i] if attack_ranks_array.size > 0 else []
            if len(positions) > 0:
                attack_rank_stability.append(np.std(positions))
            else:
                attack_rank_stability.append(float('inf'))

        # Parameter confidence intervals
        agent_resistance_ci = []
        attack_severity_ci = []

        if agent_scores_array.size > 0:
            for j in range(self.num_agents):
                scores = agent_scores_array[:, j]
                agent_resistance_ci.append([
                    float(np.percentile(scores, 2.5)),
                    float(np.percentile(scores, 97.5))
                ])

        if attack_scores_array.size > 0:
            for i in range(self.num_attacks):
                scores = attack_scores_array[:, i]
                attack_severity_ci.append([
                    float(np.percentile(scores, 2.5)),
                    float(np.percentile(scores, 97.5))
                ])

        return {
            'agent_resistance_ci': agent_resistance_ci,
            'attack_severity_ci': attack_severity_ci,
            'agent_rank_stability': agent_rank_stability,
            'attack_rank_stability': attack_rank_stability,
            'mean_agent_stability': float(np.mean(agent_rank_stability)),
            'mean_attack_stability': float(np.mean(attack_rank_stability)),
            'n_successful_bootstraps': len(agent_ranks_bootstrap)
        }

    def evaluate_fit(self) -> Dict[str, float]:
        """
        Evaluate model fit quality.

        Returns:
            Dictionary with fit evaluation metrics
        """
        if self.predictions is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Compute metrics on observed data only
        observed_true = self.S[self.mask]
        observed_pred = self.predictions[self.mask]

        # Mean squared error
        mse = np.mean((observed_true - observed_pred) ** 2)

        # Mean absolute error
        mae = np.mean(np.abs(observed_true - observed_pred))

        # Pseudo R-squared (McFadden's)
        null_likelihood = -len(observed_true) * np.log(2)  # Assuming uniform 0.5 prediction
        if self.fit_result:
            pseudo_r2 = 1 - (self.fit_result.log_likelihood / null_likelihood)
        else:
            pseudo_r2 = 0.0

        # AIC and BIC
        n_params = self.num_attacks + self.num_agents
        n_obs = np.sum(self.mask)
        aic = 2 * n_params - 2 * self.fit_result.log_likelihood if self.fit_result else float('inf')
        bic = n_params * np.log(n_obs) - 2 * self.fit_result.log_likelihood if self.fit_result else float('inf')

        return {
            'mse': float(mse),
            'mae': float(mae),
            'pseudo_r2': float(pseudo_r2),
            'aic': float(aic),
            'bic': float(bic),
            'convergence': self.fit_result.convergence if self.fit_result else False
        }

    def predict_missing(self) -> np.ndarray:
        """
        Predict values for unobserved positions.

        Returns:
            Matrix with predictions for missing values only
        """
        if self.predictions is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        missing_predictions = np.zeros_like(self.S)
        missing_mask = ~self.mask

        missing_predictions[missing_mask] = self.predictions[missing_mask]

        return missing_predictions

    def save_model(self, filepath: str) -> None:
        """
        Save fitted model to JSON file.

        Args:
            filepath: Path to save model
        """
        if self.fit_result is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        model_data = {
            'num_attacks': int(self.num_attacks),
            'num_agents': int(self.num_agents),
            'attack_severity': self.attack_severity.tolist(),
            'agent_resistance': self.agent_resistance.tolist(),
            'log_likelihood': float(self.fit_result.log_likelihood),
            'convergence': bool(self.fit_result.convergence),
            'fit_metrics': self.evaluate_fit(),
            'observed_coverage': float(np.mean(self.mask))
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)

        logger.info(f"Saved model to {filepath}")

    @classmethod
    def load_model(cls, filepath: str, tvd_matrix: np.ndarray, mask: np.ndarray) -> 'VulnerabilityModel':
        """
        Load model from JSON file.

        Args:
            filepath: Path to load model from
            tvd_matrix: TVD-MI matrix
            mask: Sampling mask

        Returns:
            Loaded vulnerability model
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        model = cls(tvd_matrix, mask)
        model.attack_severity = np.array(model_data['attack_severity'])
        model.agent_resistance = np.array(model_data['agent_resistance'])
        model.predictions = model._compute_predictions()

        model.fit_result = RaschParameters(
            attack_severity=model.attack_severity,
            agent_resistance=model.agent_resistance,
            global_discrimination=1.0,
            log_likelihood=model_data['log_likelihood'],
            convergence=model_data['convergence']
        )

        logger.info(f"Loaded model from {filepath}")
        return model