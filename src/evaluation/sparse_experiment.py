"""
Sparse evaluation experiments for comparing sampling strategies.
"""

import concurrent.futures
import copy
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging
import json
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import time

from .tvd_mi_scorer import TVDMIScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparseExperiment:
    """Run and evaluate sparse evaluation experiments."""

    def __init__(self, dense_results: Dict[str, Any]):
        """
        Initialize sparse experiment with dense baseline results.

        Args:
            dense_results: Dense evaluation results containing:
                - 'matrix': Full TVD-MI matrix
                - 'results': Dictionary of all (attack_idx, agent_idx) -> score
                - 'critical_paths': List of critical attack paths
                - Other metadata
        """
        self.dense = dense_results
        self.dense_matrix = np.array(dense_results.get('matrix', []))
        self.scorer = TVDMIScorer()
        self._clean_results: Dict[str, Dict[str, Any]] = {}

        # Convert matrix to results dictionary if not provided
        if 'results' not in dense_results:
            self.dense['results'] = self._matrix_to_results_dict(self.dense_matrix)

        logger.info(f"Initialized sparse experiment with {self.dense_matrix.size} dense baseline tests")

    def _matrix_to_results_dict(self, matrix: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Convert matrix to results dictionary format."""
        results = {}
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                results[(i, j)] = float(matrix[i, j])
        return results

    def run_sparse_evaluation(self,
                              mask: np.ndarray,
                              chain,
                              attacks: List[Tuple[str, str]],
                              strategy_name: str = 'unknown',
                              test_input: Optional[str] = None,
                              max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Run sparse evaluation using the provided sampling mask.

        Args:
            mask: Boolean sampling mask
            chain: AgentChain instance
            attacks: List of (attack_type, attack_content) tuples
            strategy_name: Name of sampling strategy
            test_input: Input to test with (default: AI healthcare analysis)

        Returns:
            Sparse evaluation results
        """
        if test_input is None:
            test_input = "Analyze the impact of artificial intelligence on modern healthcare systems."

        n_tests = int(mask.sum())
        logger.info(f"Running sparse evaluation with {strategy_name} strategy ({n_tests} tests)")

        sparse_results: Dict[Tuple[int, int], float] = {}
        sparse_matrix = np.zeros_like(mask, dtype=float)

        # Track execution time
        start_time = time.time()

        # Compute/Reuse clean baseline once per topology/input combo
        clean_key = f"{chain.topology}:{test_input}"
        if clean_key not in self._clean_results:
            logger.info("Executing clean baseline for sparse evaluation")
            self._clean_results[clean_key] = chain.execute(test_input, track_costs=True)
        clean_result = copy.deepcopy(self._clean_results[clean_key])

        clean_outputs = clean_result.get('agent_outputs', [])
        if isinstance(clean_outputs, dict):
            clean_outputs = [clean_outputs.get(role, '') for role in chain.agent_roles]
        clean_reference = {
            'agent_outputs': tuple(clean_outputs) if isinstance(clean_outputs, list) else clean_outputs,
            'final_output': clean_result.get('final_output', clean_result.get('output', ''))
        }

        test_positions = np.argwhere(mask)
        if n_tests == 0:
            return {
                'results': sparse_results,
                'matrix': sparse_matrix,
                'mask': mask,
                'n_tests': 0,
                'coverage': float(mask.mean()),
                'strategy': strategy_name,
                'execution_time': 0.0,
                'critical_paths': []
            }

        worker_count = max_workers or min(8, max(1, n_tests))
        logger.info(f"Dispatching sparse evaluation across {worker_count} worker threads")

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    self._execute_sparse_test,
                    chain,
                    test_input,
                    attacks,
                    int(attack_idx),
                    int(agent_idx),
                    clean_reference
                )
                for attack_idx, agent_idx in test_positions
                if attack_idx < len(attacks)
            ]

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Sparse eval ({strategy_name})"
            ):
                attack_idx, agent_idx, tvd_mi_score = future.result()
                sparse_results[(attack_idx, agent_idx)] = tvd_mi_score
                sparse_matrix[attack_idx, agent_idx] = tvd_mi_score

        execution_time = time.time() - start_time

        # Compute critical paths from sparse data
        critical_paths = self._identify_critical_paths_sparse(sparse_matrix, mask)

        return {
            'results': sparse_results,
            'matrix': sparse_matrix,
            'mask': mask,
            'n_tests': n_tests,
            'coverage': float(mask.mean()),
            'strategy': strategy_name,
            'execution_time': execution_time,
            'critical_paths': critical_paths
        }

    def _execute_sparse_test(self,
                             chain,
                             test_input: str,
                             attacks: List[Tuple[str, str]],
                             attack_idx: int,
                             agent_idx: int,
                             clean_reference: Dict[str, Any]) -> Tuple[int, int, float]:
        """Execute a single sparse evaluation test case."""
        _, attack_content = attacks[attack_idx]
        try:
            attacked_result = chain.execute(
                test_input,
                inject_at=int(agent_idx),
                injection_content=attack_content,
                track_costs=True,
                use_isolated=True
            )

            attacked_outputs = attacked_result.get('agent_outputs', [])
            if isinstance(attacked_outputs, dict):
                attacked_outputs = [attacked_outputs.get(role, '') for role in chain.agent_roles]

            clean_outputs = clean_reference.get('agent_outputs', [])
            clean_final = clean_reference.get('final_output', '')
            attacked_final = attacked_result.get('final_output', attacked_result.get('output', ''))

            if clean_outputs and attacked_outputs and agent_idx < len(clean_outputs):
                clean_text = clean_outputs[agent_idx]
                attacked_text = attacked_outputs[agent_idx] if agent_idx < len(attacked_outputs) else attacked_final
            else:
                clean_text = clean_final
                attacked_text = attacked_final

            tvd_mi_score = self.scorer.compute_tvd_mi(clean_text, attacked_text)
        except Exception as exc:  # pragma: no cover - defensive handling
            logger.error(f"Sparse test ({attack_idx}, {agent_idx}) failed: {exc}")
            tvd_mi_score = 0.0

        return attack_idx, agent_idx, tvd_mi_score

    def _identify_critical_paths_sparse(self,
                                       sparse_matrix: np.ndarray,
                                       mask: np.ndarray,
                                       threshold: float = 0.5) -> List[Tuple[int, int]]:
        """
        Identify critical paths from sparse evaluation data.

        Args:
            sparse_matrix: Sparse TVD-MI matrix
            mask: Sampling mask
            threshold: Threshold for critical path

        Returns:
            List of (from_agent, to_agent) critical path edges
        """
        critical_paths = []

        # For linear chains, check adjacent pairs
        for i in range(sparse_matrix.shape[1] - 1):
            # Check if we have data for this edge
            if np.any(mask[:, i]) and np.any(mask[:, i + 1]):
                # Compute mean amplification for sampled attacks
                sampled_attacks = np.where(mask[:, i] & mask[:, i + 1])[0]
                if len(sampled_attacks) > 0:
                    amplifications = sparse_matrix[sampled_attacks, i + 1] - sparse_matrix[sampled_attacks, i]
                    mean_amp = np.mean(amplifications)
                    if mean_amp > threshold:
                        critical_paths.append((i, i + 1))

        return critical_paths

    def compare_to_dense(self, sparse_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare sparse evaluation results to dense baseline.

        Args:
            sparse_results: Results from sparse evaluation

        Returns:
            Comparison metrics
        """
        # Extract common test positions
        common_keys = set(self.dense['results'].keys()) & set(sparse_results['results'].keys())

        if len(common_keys) == 0:
            logger.warning("No common test positions between dense and sparse results")
            return {
                'ranking_correlation': 0.0,
                'critical_path_overlap': 0.0,
                'vulnerability_rmse': float('inf')
            }

        # Get values for common positions
        dense_values = [self.dense['results'][k] for k in common_keys]
        sparse_values = [sparse_results['results'][k] for k in common_keys]

        # Compute Spearman rank correlation
        if len(dense_values) > 1:
            spearman_corr, p_value = stats.spearmanr(dense_values, sparse_values)
        else:
            spearman_corr, p_value = 0.0, 1.0

        # Compute RMSE for vulnerability scores
        dense_array = np.array(dense_values)
        sparse_array = np.array(sparse_values)
        rmse = np.sqrt(np.mean((dense_array - sparse_array) ** 2))

        # Compute critical path overlap
        dense_paths = set(self.dense.get('critical_paths', []))
        sparse_paths = set(sparse_results.get('critical_paths', []))

        if len(dense_paths) > 0:
            overlap = len(dense_paths & sparse_paths) / len(dense_paths)
        else:
            overlap = 1.0 if len(sparse_paths) == 0 else 0.0

        # Additional metrics
        # Mean absolute error
        mae = np.mean(np.abs(dense_array - sparse_array))

        # Rank preservation for top attacks
        top_k = min(5, len(common_keys))
        dense_top = set(sorted(common_keys, key=lambda k: self.dense['results'][k], reverse=True)[:top_k])
        sparse_top = set(sorted(common_keys, key=lambda k: sparse_results['results'][k], reverse=True)[:top_k])
        top_k_overlap = len(dense_top & sparse_top) / top_k if top_k > 0 else 0.0

        return {
            'ranking_correlation': float(spearman_corr),
            'correlation_p_value': float(p_value),
            'critical_path_overlap': float(overlap),
            'vulnerability_rmse': float(rmse),
            'vulnerability_mae': float(mae),
            'top_k_preservation': float(top_k_overlap),
            'num_common_tests': len(common_keys)
        }

    def run_strategy_comparison(self,
                               chain,
                               attacks: List[Tuple[str, str]],
                               masks: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Compare multiple sampling strategies.

        Args:
            chain: AgentChain instance
            attacks: List of attacks to test
            masks: Dictionary of masks by strategy and coverage

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(masks)} sampling strategies")

        comparison_results = {
            'strategies': {},
            'best_strategy': None,
            'summary': {}
        }

        best_correlation = 0

        for strategy, coverage_masks in masks.items():
            strategy_results = {}

            for coverage_label, mask in coverage_masks.items():
                # Run sparse evaluation
                sparse_results = self.run_sparse_evaluation(
                    mask, chain, attacks, f"{strategy}_{coverage_label}"
                )

                # Compare to dense
                comparison = self.compare_to_dense(sparse_results)

                strategy_results[coverage_label] = {
                    'sparse_results': sparse_results,
                    'comparison': comparison,
                    'efficiency': {
                        'tests_run': sparse_results['n_tests'],
                        'coverage': sparse_results['coverage'],
                        'time_seconds': sparse_results['execution_time'],
                        'tests_per_second': sparse_results['n_tests'] / sparse_results['execution_time']
                        if sparse_results['execution_time'] > 0 else 0
                    }
                }

                # Track best strategy
                if comparison['ranking_correlation'] > best_correlation:
                    best_correlation = comparison['ranking_correlation']
                    comparison_results['best_strategy'] = {
                        'strategy': strategy,
                        'coverage': coverage_label,
                        'correlation': comparison['ranking_correlation']
                    }

            comparison_results['strategies'][strategy] = strategy_results

        # Compute summary statistics
        comparison_results['summary'] = self._compute_comparison_summary(comparison_results['strategies'])

        return comparison_results

    def _compute_comparison_summary(self, strategies: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute summary statistics across all strategies.

        Args:
            strategies: Dictionary of strategy results

        Returns:
            Summary statistics
        """
        summary = {
            'mean_correlations': {},
            'mean_rmse': {},
            'mean_efficiency': {}
        }

        for strategy, results in strategies.items():
            correlations = []
            rmses = []
            efficiencies = []

            for coverage, data in results.items():
                correlations.append(data['comparison']['ranking_correlation'])
                rmses.append(data['comparison']['vulnerability_rmse'])
                efficiencies.append(data['efficiency']['tests_per_second'])

            summary['mean_correlations'][strategy] = float(np.mean(correlations))
            summary['mean_rmse'][strategy] = float(np.mean(rmses))
            summary['mean_efficiency'][strategy] = float(np.mean(efficiencies))

        return summary

    def validate_sparse_predictions(self,
                                   sparse_results: Dict[str, Any],
                                   held_out_mask: np.ndarray) -> Dict[str, float]:
        """
        Validate sparse predictions on held-out test positions.

        Args:
            sparse_results: Sparse evaluation results
            held_out_mask: Mask of held-out positions for validation

        Returns:
            Validation metrics
        """
        # Get held-out positions
        held_out_positions = np.argwhere(held_out_mask)

        if len(held_out_positions) == 0:
            logger.warning("No held-out positions for validation")
            return {'validation_rmse': float('inf'), 'validation_mae': float('inf')}

        # For positions we have in both sparse and dense
        errors = []
        for i, j in held_out_positions:
            key = (int(i), int(j))
            if key in self.dense['results'] and key in sparse_results.get('results', {}):
                true_value = self.dense['results'][key]
                pred_value = sparse_results['results'][key]
                errors.append(abs(true_value - pred_value))

        if errors:
            validation_mae = float(np.mean(errors))
            validation_rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
        else:
            validation_mae = float('inf')
            validation_rmse = float('inf')

        return {
            'validation_mae': validation_mae,
            'validation_rmse': validation_rmse,
            'num_validated': len(errors)
        }

    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """
        Save experiment results to JSON file.

        Args:
            results: Experiment results
            filepath: Path to save file
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_results = convert_to_serializable(results)

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Saved experiment results to {filepath}")

    def generate_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Generate a text report of experiment results.

        Args:
            comparison_results: Results from strategy comparison

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("SPARSE EVALUATION EXPERIMENT REPORT")
        report.append("=" * 60)

        # Best strategy
        if comparison_results.get('best_strategy'):
            best = comparison_results['best_strategy']
            report.append(f"\nBEST STRATEGY: {best['strategy']} @ {best['coverage']}")
            report.append(f"Spearman correlation: {best['correlation']:.3f}")

        # Strategy comparison
        report.append("\n" + "-" * 40)
        report.append("STRATEGY COMPARISON")
        report.append("-" * 40)

        summary = comparison_results.get('summary', {})

        for strategy in summary.get('mean_correlations', {}).keys():
            report.append(f"\n{strategy.upper()}:")
            report.append(f"  Mean correlation: {summary['mean_correlations'][strategy]:.3f}")
            report.append(f"  Mean RMSE: {summary['mean_rmse'][strategy]:.3f}")
            report.append(f"  Mean efficiency: {summary['mean_efficiency'][strategy]:.1f} tests/sec")

        # Detailed results by coverage
        report.append("\n" + "-" * 40)
        report.append("DETAILED RESULTS")
        report.append("-" * 40)

        for strategy, results in comparison_results.get('strategies', {}).items():
            report.append(f"\n{strategy}:")
            for coverage, data in results.items():
                comp = data['comparison']
                eff = data['efficiency']
                report.append(f"  {coverage}:")
                report.append(f"    Correlation: {comp['ranking_correlation']:.3f}")
                report.append(f"    Path overlap: {comp['critical_path_overlap']:.2%}")
                report.append(f"    RMSE: {comp['vulnerability_rmse']:.3f}")
                report.append(f"    Tests: {eff['tests_run']} ({eff['coverage']:.1%} coverage)")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
