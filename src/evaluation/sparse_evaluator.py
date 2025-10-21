"""
Sparse evaluation strategies for red-teaming experiments.
Note: Efficiency claims have not been empirically validated.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparseEvaluator:
    """
    Creates sampling masks for sparse evaluation.
    Supports random, nlogn, and informed sampling strategies.
    """

    def __init__(self, n_attacks: int, n_agents: int):
        """
        Initialize the sparse evaluator.

        Args:
            n_attacks: Number of attack scenarios
            n_agents: Number of agents in chain
        """
        self.n_attacks = n_attacks
        self.n_agents = n_agents
        self.total_tests = n_attacks * n_agents

        logger.info(f"Initialized sparse evaluator for {n_attacks}x{n_agents} matrix")

    def create_random_mask(self, coverage: float = 0.33, seed: int = 42) -> np.ndarray:
        """
        Create a random sampling mask.

        Args:
            coverage: Fraction of tests to sample (0 to 1)
            seed: Random seed for reproducibility

        Returns:
            Boolean mask array
        """
        np.random.seed(seed)
        mask = np.random.random((self.n_attacks, self.n_agents)) < coverage

        logger.info(f"Created random mask with {mask.sum()}/{mask.size} tests "
                   f"({mask.mean():.1%} coverage)")

        return mask

    def create_nlogn_mask(self, seed: int = 42) -> np.ndarray:
        """
        Create an n*log(n) sampling mask.
        Samples O(n*log(n)) positions for efficient coverage.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Boolean mask array
        """
        np.random.seed(seed)

        # Calculate nlogn sample size
        n = max(self.n_attacks, self.n_agents)
        nlogn_samples = int(n * np.log(n) * 2)  # Factor of 2 for better coverage
        nlogn_samples = min(nlogn_samples, self.total_tests)

        # Create mask
        mask = np.zeros((self.n_attacks, self.n_agents), dtype=bool)

        # Randomly select positions
        indices = np.random.choice(
            self.total_tests,
            size=nlogn_samples,
            replace=False
        )

        for idx in indices:
            i = idx // self.n_agents
            j = idx % self.n_agents
            mask[i, j] = True

        logger.info(f"Created nlogn mask with {mask.sum()}/{mask.size} tests "
                   f"({mask.mean():.1%} coverage)")

        return mask

    def create_informed_mask(self, coverage: float = 0.33,
                            high_value_ratio: float = 0.7,
                            seed: int = 42) -> np.ndarray:
        """
        Create an informed sampling mask based on attack patterns.
        Prioritizes high-value tests based on propagation analysis.

        Args:
            coverage: Fraction of tests to sample
            high_value_ratio: Ratio of high-value to random samples
            seed: Random seed

        Returns:
            Boolean mask array
        """
        np.random.seed(seed)

        n_samples = int(self.total_tests * coverage)
        n_high_value = int(n_samples * high_value_ratio)
        n_random = n_samples - n_high_value

        mask = np.zeros((self.n_attacks, self.n_agents), dtype=bool)

        # High-value sampling patterns
        # 1. Sample first and last agents (entry/exit points)
        for i in range(self.n_attacks):
            if np.random.random() < 0.8:  # 80% chance
                mask[i, 0] = True
                mask[i, -1] = True

        # 2. Sample middle agents for critical paths
        mid_agent = self.n_agents // 2
        for i in range(self.n_attacks):
            if np.random.random() < 0.6:  # 60% chance
                mask[i, mid_agent] = True

        # 3. Sample high-severity attacks more densely
        high_severity_attacks = list(range(0, self.n_attacks, 2))  # Even indices
        for i in high_severity_attacks:
            extra_samples = np.random.choice(
                self.n_agents,
                size=min(2, self.n_agents),
                replace=False
            )
            for j in extra_samples:
                mask[i, j] = True

        # 4. Add random samples to reach target coverage
        current_samples = mask.sum()
        remaining = max(0, n_samples - current_samples)

        if remaining > 0:
            # Get unsampled positions
            unsampled = np.argwhere(~mask)
            if len(unsampled) > 0:
                random_indices = np.random.choice(
                    len(unsampled),
                    size=min(remaining, len(unsampled)),
                    replace=False
                )
                for idx in random_indices:
                    i, j = unsampled[idx]
                    mask[i, j] = True

        logger.info(f"Created informed mask with {mask.sum()}/{mask.size} tests "
                   f"({mask.mean():.1%} coverage)")

        return mask

    def analyze_coverage(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the coverage properties of a sampling mask.

        Args:
            mask: Boolean sampling mask

        Returns:
            Dictionary with coverage statistics
        """
        stats = {
            'total_tests': int(mask.sum()),
            'coverage': float(mask.mean()),
            'attacks_covered': int(np.any(mask, axis=1).sum()),
            'agents_covered': int(np.any(mask, axis=0).sum()),
            'attack_coverage': mask.mean(axis=1).tolist(),
            'agent_coverage': mask.mean(axis=0).tolist(),
            'full_attack_coverage': int((mask.sum(axis=1) == self.n_agents).sum()),
            'full_agent_coverage': int((mask.sum(axis=0) == self.n_attacks).sum()),
        }

        # Compute sparsity pattern metrics
        # Check for connected components
        stats['has_full_row'] = bool(np.any(mask.sum(axis=1) == self.n_agents))
        stats['has_full_col'] = bool(np.any(mask.sum(axis=0) == self.n_attacks))

        # Measure uniformity
        row_variance = np.var(mask.sum(axis=1))
        col_variance = np.var(mask.sum(axis=0))
        stats['row_uniformity'] = float(1 / (1 + row_variance))
        stats['col_uniformity'] = float(1 / (1 + col_variance))

        return stats

    def compare_strategies(self, coverage: float = 0.33,
                          seed: int = 42) -> Dict[str, Dict[str, Any]]:
        """
        Compare different sampling strategies.

        Args:
            coverage: Target coverage level
            seed: Random seed

        Returns:
            Comparison results for each strategy
        """
        strategies = {
            'random': self.create_random_mask(coverage, seed),
            'nlogn': self.create_nlogn_mask(seed),
            'informed': self.create_informed_mask(coverage, 0.7, seed)
        }

        results = {}
        for name, mask in strategies.items():
            results[name] = self.analyze_coverage(mask)
            results[name]['strategy'] = name

        return results

    def visualize_mask(self, mask: np.ndarray) -> str:
        """
        Create a text visualization of the sampling mask.

        Args:
            mask: Boolean sampling mask

        Returns:
            String visualization
        """
        lines = []
        lines.append("Sampling Mask Visualization:")
        lines.append("  Agents →")
        lines.append("A +" + "-" * (self.n_agents * 2 - 1) + "+")

        for i in range(self.n_attacks):
            row = f"{i:1d} |"
            for j in range(self.n_agents):
                row += "■ " if mask[i, j] else "□ "
            row += f"| {mask[i].sum()}/{self.n_agents}"
            lines.append(row)

        lines.append("  +" + "-" * (self.n_agents * 2 - 1) + "+")

        # Column totals
        col_totals = "   "
        for j in range(self.n_agents):
            col_totals += f"{mask[:, j].sum():1d} "
        lines.append(col_totals)

        lines.append(f"\nTotal: {mask.sum()}/{mask.size} ({mask.mean():.1%})")

        return "\n".join(lines)