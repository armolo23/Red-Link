"""
Sparse sampling strategies for efficient red-team evaluation.
Implements informed, nlogn, and random sampling approaches.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InformedSampler:
    """Create informed sampling masks based on pattern analysis."""

    def __init__(self, pattern_analysis: Optional['PatternDiscovery'] = None):
        """
        Initialize the informed sampler.

        Args:
            pattern_analysis: PatternDiscovery instance with analysis results
        """
        self.patterns = pattern_analysis
        if pattern_analysis:
            self.high_value_tests = pattern_analysis.identify_high_value_tests()
        else:
            self.high_value_tests = np.array([])

        logger.info("Initialized informed sampler")

    def create_sampling_mask(self,
                           K: int,
                           J: int,
                           coverage: float = 0.33,
                           strategy: str = 'informed',
                           min_degree: int = 3,
                           random_seed: Optional[int] = 42) -> np.ndarray:
        """
        Create a sampling mask for sparse evaluation.

        Args:
            K: Number of attacks
            J: Number of agents
            coverage: Fraction of tests to sample (0-1)
            strategy: Sampling strategy ('informed', 'nlogn', 'random')
            min_degree: Minimum samples per row and column
            random_seed: Random seed for reproducibility

        Returns:
            Boolean mask of shape (K, J) indicating which tests to run
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(f"Creating {strategy} sampling mask with {coverage:.1%} coverage for {K}×{J} matrix")

        if strategy == 'informed':
            mask = self._informed_sampling(K, J, coverage)
        elif strategy == 'nlogn':
            mask = self._nlogn_sampling(K, J)
        elif strategy == 'random':
            mask = self._random_sampling(K, J, coverage)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        # Enforce minimum degree constraints
        mask = self._enforce_min_degree(mask, min_degree)

        # Log statistics
        actual_coverage = np.mean(mask)
        logger.info(f"Created mask with actual coverage: {actual_coverage:.1%} ({np.sum(mask)} tests)")

        return mask

    def _informed_sampling(self, K: int, J: int, coverage: float) -> np.ndarray:
        """
        Create informed sampling mask using pattern analysis results.

        Args:
            K: Number of attacks
            J: Number of agents
            coverage: Target coverage fraction

        Returns:
            Boolean sampling mask
        """
        mask = np.zeros((K, J), dtype=bool)
        total_tests = int(coverage * K * J)

        if self.patterns and len(self.high_value_tests) > 0:
            # Use 70% of budget for high-value tests
            num_high_value = int(0.7 * total_tests)
            num_random = total_tests - num_high_value

            # Add high-value tests
            high_value_count = 0
            for flat_idx in self.high_value_tests:
                if high_value_count >= num_high_value:
                    break
                i, j = flat_idx // J, flat_idx % J
                if i < K and j < J:
                    mask[i, j] = True
                    high_value_count += 1

            # Add random coverage for exploration
            mask = self._add_random_coverage(mask, num_random)

            # Add critical positions if pattern analysis is available
            if hasattr(self.patterns, 'position_importance_ranking'):
                position_ranking = self.patterns.position_importance_ranking()
                # Ensure top 3 positions are well covered
                for pos, _ in position_ranking[:min(3, J)]:
                    for attack_idx in range(min(2, K)):  # At least 2 attacks per critical position
                        mask[attack_idx, pos] = True

        else:
            # Fallback to stratified random sampling
            logger.warning("No pattern analysis available, using stratified random sampling")
            mask = self._stratified_random_sampling(K, J, coverage)

        return mask

    def _nlogn_sampling(self, K: int, J: int) -> np.ndarray:
        """
        Create O(n log n) sampling mask based on matrix completion theory.

        Args:
            K: Number of attacks
            J: Number of agents

        Returns:
            Boolean sampling mask
        """
        mask = np.zeros((K, J), dtype=bool)
        n_total = K + J
        n_samples = int(2.0 * n_total * np.log(n_total))

        # Ensure we don't exceed matrix size
        n_samples = min(n_samples, K * J)

        # Random sampling with higher probability on diagonal-like patterns
        indices = []

        # Add diagonal and near-diagonal elements (important for structure)
        for offset in range(-1, 2):
            for i in range(K):
                j = i + offset
                if 0 <= j < J:
                    indices.append((i, j))

        # Add random samples for the rest
        remaining_samples = n_samples - len(indices)
        if remaining_samples > 0:
            all_indices = [(i, j) for i in range(K) for j in range(J)
                          if (i, j) not in indices]
            if all_indices:
                random_indices = np.random.choice(
                    len(all_indices),
                    min(remaining_samples, len(all_indices)),
                    replace=False
                )
                indices.extend([all_indices[idx] for idx in random_indices])

        # Set mask
        for i, j in indices[:n_samples]:
            mask[i, j] = True

        logger.info(f"nlogn sampling: {n_samples} samples for {K}×{J} matrix")
        return mask

    def _random_sampling(self, K: int, J: int, coverage: float) -> np.ndarray:
        """
        Create uniform random sampling mask.

        Args:
            K: Number of attacks
            J: Number of agents
            coverage: Target coverage fraction

        Returns:
            Boolean sampling mask
        """
        mask = np.random.rand(K, J) < coverage
        return mask

    def _stratified_random_sampling(self, K: int, J: int, coverage: float) -> np.ndarray:
        """
        Create stratified random sampling to ensure coverage across regions.

        Args:
            K: Number of attacks
            J: Number of agents
            coverage: Target coverage fraction

        Returns:
            Boolean sampling mask
        """
        mask = np.zeros((K, J), dtype=bool)
        total_tests = int(coverage * K * J)

        # Divide matrix into strata
        strata_k = min(3, K)  # Number of attack strata
        strata_j = min(3, J)  # Number of agent strata

        tests_per_stratum = total_tests // (strata_k * strata_j)

        for i_strat in range(strata_k):
            for j_strat in range(strata_j):
                # Define stratum boundaries
                k_start = i_strat * K // strata_k
                k_end = (i_strat + 1) * K // strata_k
                j_start = j_strat * J // strata_j
                j_end = (j_strat + 1) * J // strata_j

                # Sample within stratum
                stratum_positions = [
                    (i, j)
                    for i in range(k_start, k_end)
                    for j in range(j_start, j_end)
                ]

                if stratum_positions:
                    n_samples = min(tests_per_stratum, len(stratum_positions))
                    sampled_indices = np.random.choice(
                        len(stratum_positions),
                        n_samples,
                        replace=False
                    )

                    for idx in sampled_indices:
                        i, j = stratum_positions[idx]
                        mask[i, j] = True

        return mask

    def _add_random_coverage(self, mask: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Add random samples to existing mask.

        Args:
            mask: Current sampling mask
            n_samples: Number of random samples to add

        Returns:
            Updated mask
        """
        # Find unsampled positions
        unsampled = np.where(~mask.ravel())[0]

        if len(unsampled) > 0 and n_samples > 0:
            # Randomly select positions
            n_to_add = min(n_samples, len(unsampled))
            new_indices = np.random.choice(unsampled, n_to_add, replace=False)

            # Update mask
            mask.ravel()[new_indices] = True

        return mask

    def _enforce_min_degree(self, mask: np.ndarray, min_d: int) -> np.ndarray:
        """
        Enforce minimum degree constraints for matrix completion.

        Args:
            mask: Current sampling mask
            min_d: Minimum number of samples per row and column

        Returns:
            Updated mask with min degree constraints satisfied
        """
        K, J = mask.shape
        max_iterations = 100
        iteration = 0

        while iteration < max_iterations:
            row_degrees = mask.sum(axis=1)
            col_degrees = mask.sum(axis=0)

            # Check if constraints are satisfied
            if np.all(row_degrees >= min_d) and np.all(col_degrees >= min_d):
                break

            # Add samples to rows with insufficient degree
            for i in range(K):
                if row_degrees[i] < min_d:
                    # Find columns with lowest degree
                    available_cols = np.where(~mask[i, :])[0]
                    if len(available_cols) > 0:
                        n_needed = min_d - row_degrees[i]
                        # Prefer columns with low degree
                        col_scores = col_degrees[available_cols]
                        sorted_cols = available_cols[np.argsort(col_scores)]
                        for j in sorted_cols[:n_needed]:
                            mask[i, j] = True

            # Add samples to columns with insufficient degree
            for j in range(J):
                if col_degrees[j] < min_d:
                    # Find rows with lowest degree
                    available_rows = np.where(~mask[:, j])[0]
                    if len(available_rows) > 0:
                        n_needed = min_d - col_degrees[j]
                        # Prefer rows with low degree
                        row_scores = row_degrees[available_rows]
                        sorted_rows = available_rows[np.argsort(row_scores)]
                        for i in sorted_rows[:n_needed]:
                            mask[i, j] = True

            iteration += 1

        if iteration == max_iterations:
            logger.warning(f"Could not fully satisfy min_degree={min_d} constraint")

        return mask

    def create_multiple_masks(self,
                            K: int,
                            J: int,
                            coverage_levels: List[float],
                            strategies: List[str],
                            random_seed: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Create multiple sampling masks for comparison.

        Args:
            K: Number of attacks
            J: Number of agents
            coverage_levels: List of coverage fractions to test
            strategies: List of sampling strategies to compare
            random_seed: Random seed for reproducibility

        Returns:
            Nested dictionary of masks: {strategy: {coverage: mask}}
        """
        masks = {}

        for strategy in strategies:
            masks[strategy] = {}
            for coverage in coverage_levels:
                # Use different seeds for different configurations
                seed = random_seed + hash(f"{strategy}_{coverage}") % 1000
                mask = self.create_sampling_mask(
                    K, J, coverage, strategy, random_seed=seed
                )
                masks[strategy][f"{int(coverage*100)}%"] = mask

                # Log mask statistics
                logger.info(f"{strategy} @ {coverage:.1%}: {np.sum(mask)} tests, "
                          f"row coverage: {np.mean(mask.sum(axis=1)):.1f}, "
                          f"col coverage: {np.mean(mask.sum(axis=0)):.1f}")

        return masks

    def save_masks(self, masks: Dict[str, Dict[str, np.ndarray]], directory: str) -> None:
        """
        Save sampling masks to files.

        Args:
            masks: Dictionary of sampling masks
            directory: Directory to save masks
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        for strategy, coverage_masks in masks.items():
            for coverage, mask in coverage_masks.items():
                filename = f"mask_{strategy}_{coverage}.json"
                filepath = dir_path / filename

                mask_data = {
                    'strategy': strategy,
                    'coverage': coverage,
                    'shape': list(mask.shape),
                    'mask': mask.tolist(),
                    'statistics': {
                        'total_samples': int(np.sum(mask)),
                        'actual_coverage': float(np.mean(mask)),
                        'min_row_samples': int(np.min(mask.sum(axis=1))),
                        'max_row_samples': int(np.max(mask.sum(axis=1))),
                        'min_col_samples': int(np.min(mask.sum(axis=0))),
                        'max_col_samples': int(np.max(mask.sum(axis=0)))
                    }
                }

                with open(filepath, 'w') as f:
                    json.dump(mask_data, f, indent=2)

        logger.info(f"Saved {len(masks)} × {len(next(iter(masks.values())))} masks to {directory}")

    @staticmethod
    def load_mask(filepath: str) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Load a sampling mask from file.

        Args:
            filepath: Path to mask file

        Returns:
            Tuple of (mask array, metadata)
        """
        with open(filepath, 'r') as f:
            mask_data = json.load(f)

        mask = np.array(mask_data['mask'], dtype=bool)
        metadata = {
            'strategy': mask_data['strategy'],
            'coverage': mask_data['coverage'],
            'statistics': mask_data['statistics']
        }

        return mask, metadata