"""
Vulnerability heatmap visualizations for multi-agent systems.
Creates various heatmaps and rankings for vulnerability analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="whitegrid", context="notebook")


class VulnerabilityVisualizer:
    """Create vulnerability-focused visualizations."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the vulnerability visualizer.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize

    def plot_vulnerability_rankings(self,
                                   agent_names: List[str],
                                   agent_resistance: np.ndarray,
                                   attack_names: List[str],
                                   attack_severity: np.ndarray,
                                   confidence_intervals: Optional[Dict] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot agent and attack rankings with confidence intervals.

        Args:
            agent_names: List of agent names
            agent_resistance: Agent resistance scores
            attack_names: List of attack names
            attack_severity: Attack severity scores
            confidence_intervals: Optional CI data from bootstrap
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Agent resistance ranking
        agent_order = np.argsort(agent_resistance)[::-1]  # Most resistant first
        y_pos = np.arange(len(agent_names))

        ax1.barh(y_pos, agent_resistance[agent_order], alpha=0.7, color='steelblue')

        # Add confidence intervals if available
        if confidence_intervals and 'agent_resistance_ci' in confidence_intervals:
            ci_data = confidence_intervals['agent_resistance_ci']
            if ci_data:
                for i, idx in enumerate(agent_order):
                    if idx < len(ci_data):
                        ci = ci_data[idx]
                        ax1.errorbar(agent_resistance[idx], i, xerr=[[agent_resistance[idx] - ci[0]],
                                                                     [ci[1] - agent_resistance[idx]]],
                                   fmt='ko', capsize=3)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([agent_names[i] for i in agent_order])
        ax1.set_xlabel('Resistance Score')
        ax1.set_title('Agent Resistance Ranking\n(Higher = More Resistant)', fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')

        # Attack severity ranking
        attack_order = np.argsort(attack_severity)[::-1]  # Most severe first
        y_pos = np.arange(len(attack_names))

        ax2.barh(y_pos, attack_severity[attack_order], alpha=0.7, color='crimson')

        # Add confidence intervals if available
        if confidence_intervals and 'attack_severity_ci' in confidence_intervals:
            ci_data = confidence_intervals['attack_severity_ci']
            if ci_data:
                for i, idx in enumerate(attack_order):
                    if idx < len(ci_data):
                        ci = ci_data[idx]
                        ax2.errorbar(attack_severity[idx], i, xerr=[[attack_severity[idx] - ci[0]],
                                                                    [ci[1] - attack_severity[idx]]],
                                   fmt='ko', capsize=3)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([attack_names[i] for i in attack_order])
        ax2.set_xlabel('Severity Score')
        ax2.set_title('Attack Severity Ranking\n(Higher = More Severe)', fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')

        plt.suptitle('Vulnerability Rankings', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved vulnerability rankings to {save_path}")

        return fig

    def plot_sampling_mask(self,
                          mask: np.ndarray,
                          strategy_name: str,
                          attack_names: Optional[List[str]] = None,
                          agent_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the sampling mask pattern.

        Args:
            mask: Boolean sampling mask
            strategy_name: Name of sampling strategy
            attack_names: Optional list of attack names
            agent_names: Optional list of agent names
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Sampling mask heatmap
        im = ax1.imshow(mask.astype(int), cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)

        if attack_names:
            ax1.set_yticks(range(len(attack_names)))
            ax1.set_yticklabels(attack_names)
        else:
            ax1.set_ylabel('Attack Index')

        if agent_names:
            ax1.set_xticks(range(len(agent_names)))
            ax1.set_xticklabels(agent_names, rotation=45, ha='right')
        else:
            ax1.set_xlabel('Agent Index')

        ax1.set_title(f'{strategy_name} Sampling Mask\n(Coverage: {np.mean(mask):.1%})',
                     fontweight='bold')

        # Add grid
        ax1.set_xticks(np.arange(mask.shape[1] + 1) - 0.5, minor=True)
        ax1.set_yticks(np.arange(mask.shape[0] + 1) - 0.5, minor=True)
        ax1.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

        # Coverage statistics
        row_coverage = mask.sum(axis=1)
        col_coverage = mask.sum(axis=0)

        ax2_top = ax2.twiny()

        # Row coverage (attacks)
        ax2.barh(range(len(row_coverage)), row_coverage, alpha=0.7, color='steelblue')
        ax2.set_ylabel('Attack Index')
        ax2.set_xlabel('Samples per Attack', color='steelblue')
        ax2.tick_params(axis='x', labelcolor='steelblue')
        ax2.set_ylim(-0.5, len(row_coverage) - 0.5)
        ax2.invert_yaxis()

        # Column coverage (agents)
        ax2_top.bar(range(len(col_coverage)), col_coverage, alpha=0.7, color='crimson')
        ax2_top.set_xlabel('Samples per Agent', color='crimson')
        ax2_top.tick_params(axis='x', labelcolor='crimson')
        ax2_top.set_xlim(-0.5, len(col_coverage) - 0.5)

        ax2.set_title('Sampling Coverage Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sampling mask to {save_path}")

        return fig

    def plot_strategy_comparison(self,
                                comparison_results: Dict[str, Any],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare multiple sampling strategies.

        Args:
            comparison_results: Results from strategy comparison
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Extract data
        strategies = list(comparison_results.get('strategies', {}).keys())
        if not strategies:
            fig.text(0.5, 0.5, 'No Strategy Data Available',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=14)
            return fig

        # Prepare data for each coverage level
        coverage_levels = ['10%', '20%', '33%']
        metrics_by_coverage = {level: {} for level in coverage_levels}

        for strategy, results in comparison_results['strategies'].items():
            for coverage, data in results.items():
                if coverage in metrics_by_coverage:
                    metrics_by_coverage[coverage][strategy] = data['comparison']

        # Panel 1: Correlation by coverage
        ax1 = axes[0, 0]
        x = np.arange(len(strategies))
        width = 0.25

        for i, coverage in enumerate(coverage_levels):
            if metrics_by_coverage[coverage]:
                correlations = [metrics_by_coverage[coverage].get(s, {}).get('ranking_correlation', 0)
                              for s in strategies]
                ax1.bar(x + i * width, correlations, width, label=coverage, alpha=0.8)

        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Spearman Correlation')
        ax1.set_title('Ranking Correlation by Coverage', fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(strategies)
        ax1.legend()
        ax1.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='Target')
        ax1.grid(True, alpha=0.3, axis='y')

        # Panel 2: RMSE comparison
        ax2 = axes[0, 1]
        for i, coverage in enumerate(coverage_levels):
            if metrics_by_coverage[coverage]:
                rmses = [metrics_by_coverage[coverage].get(s, {}).get('vulnerability_rmse', 0)
                        for s in strategies]
                ax2.bar(x + i * width, rmses, width, label=coverage, alpha=0.8)

        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Vulnerability RMSE by Coverage', fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(strategies)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Panel 3: Efficiency (tests required)
        ax3 = axes[1, 0]
        for strategy in strategies:
            coverages = []
            num_tests = []
            for coverage, data in comparison_results['strategies'][strategy].items():
                coverages.append(float(coverage.rstrip('%')))
                num_tests.append(data['efficiency']['tests_run'])
            ax3.plot(coverages, num_tests, 'o-', label=strategy, linewidth=2, markersize=8)

        ax3.set_xlabel('Coverage (%)')
        ax3.set_ylabel('Number of Tests')
        ax3.set_title('Evaluation Cost by Strategy', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Cost-Fidelity Tradeoff
        ax4 = axes[1, 1]
        for strategy in strategies:
            costs = []
            fidelities = []
            for coverage, data in comparison_results['strategies'][strategy].items():
                costs.append(data['efficiency']['tests_run'])
                fidelities.append(data['comparison']['ranking_correlation'])
            ax4.scatter(costs, fidelities, s=100, label=strategy, alpha=0.7)
            ax4.plot(costs, fidelities, '--', alpha=0.5)

        ax4.set_xlabel('Number of Tests (Cost)')
        ax4.set_ylabel('Spearman Correlation (Fidelity)')
        ax4.set_title('Cost-Fidelity Tradeoff', fontweight='bold')
        ax4.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='Target Fidelity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle('Sampling Strategy Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved strategy comparison to {save_path}")

        return fig

    def plot_model_predictions(self,
                             observed_matrix: np.ndarray,
                             predicted_matrix: np.ndarray,
                             mask: np.ndarray,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare observed vs predicted vulnerability values.

        Args:
            observed_matrix: Observed TVD-MI values
            predicted_matrix: Model predictions
            mask: Sampling mask
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Observed values (sparse)
        ax1 = axes[0, 0]
        masked_observed = np.ma.masked_where(~mask, observed_matrix)
        im1 = ax1.imshow(masked_observed, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax1.set_title('Observed Values (Sparse)', fontweight='bold')
        ax1.set_xlabel('Agent Index')
        ax1.set_ylabel('Attack Index')
        plt.colorbar(im1, ax=ax1)

        # Panel 2: Predicted values (full)
        ax2 = axes[0, 1]
        im2 = ax2.imshow(predicted_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax2.set_title('Model Predictions (Full)', fontweight='bold')
        ax2.set_xlabel('Agent Index')
        ax2.set_ylabel('Attack Index')
        plt.colorbar(im2, ax=ax2)

        # Panel 3: Scatter plot of observed vs predicted
        ax3 = axes[1, 0]
        observed_vals = observed_matrix[mask]
        predicted_vals = predicted_matrix[mask]
        ax3.scatter(observed_vals, predicted_vals, alpha=0.6, s=20)
        ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Prediction')
        ax3.set_xlabel('Observed TVD-MI')
        ax3.set_ylabel('Predicted TVD-MI')
        ax3.set_title('Prediction Accuracy', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)

        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(observed_vals, predicted_vals)
        ax3.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax3.transAxes,
                fontsize=12, verticalalignment='top')

        # Panel 4: Residuals
        ax4 = axes[1, 1]
        residuals = predicted_vals - observed_vals
        ax4.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Prediction Error')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Residual Distribution', fontweight='bold')
        ax4.axvline(0, color='red', linestyle='--', alpha=0.5, label='Zero Error')
        ax4.axvline(np.mean(residuals), color='green', linestyle='--', alpha=0.5,
                   label=f'Mean: {np.mean(residuals):.3f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Vulnerability Model Validation', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model predictions to {save_path}")

        return fig