"""
Attack propagation visualization for multi-agent chains.
Creates graph-based visualizations of attack flow and impact.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="whitegrid", context="notebook")


class AttackGraphVisualizer:
    """Visualize attack propagation through agent chains."""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the attack graph visualizer.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.color_palette = sns.color_palette("viridis", as_cmap=True)

    def plot_attack_diffusion(self,
                             tvd_matrix: np.ndarray,
                             attack_names: List[str],
                             agent_names: List[str],
                             title: str = "Attack Diffusion Heatmap",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap showing attack diffusion across agents.

        Args:
            tvd_matrix: TVD-MI matrix
            attack_names: List of attack names
            agent_names: List of agent names
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create heatmap
        sns.heatmap(
            tvd_matrix,
            xticklabels=agent_names,
            yticklabels=attack_names,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'TVD-MI Score'},
            ax=ax
        )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Agent Position', fontsize=12)
        ax.set_ylabel('Attack Type', fontsize=12)

        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attack diffusion heatmap to {save_path}")

        return fig

    def plot_chain_graph(self,
                        chain_topology: str,
                        agent_names: List[str],
                        vulnerability_scores: np.ndarray,
                        critical_edges: Optional[List[Tuple[int, int, float]]] = None,
                        title: Optional[str] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot agent chain as network graph with vulnerability coloring.

        Args:
            chain_topology: Type of chain (linear, star, hierarchical)
            agent_names: List of agent names
            vulnerability_scores: Vulnerability score per agent
            critical_edges: Optional list of critical edges
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create graph
        G = nx.DiGraph()

        # Add nodes
        for i, name in enumerate(agent_names):
            G.add_node(i, label=name, vulnerability=vulnerability_scores[i])

        # Add edges based on topology
        if chain_topology == 'linear':
            for i in range(len(agent_names) - 1):
                G.add_edge(i, i + 1)
        elif chain_topology == 'star':
            # Coordinator (0) connects to all others
            for i in range(1, len(agent_names)):
                G.add_edge(0, i)
        elif chain_topology == 'hierarchical':
            # Juniors (0, 1) connect to senior (2)
            if len(agent_names) >= 3:
                G.add_edge(0, 2)
                G.add_edge(1, 2)

        # Set layout
        if chain_topology == 'linear':
            pos = {i: (i, 0) for i in range(len(agent_names))}
        elif chain_topology == 'star':
            pos = nx.spring_layout(G, k=2, iterations=50)
        else:
            pos = nx.hierarchical_layout(G)

        # Draw edges
        edge_colors = []
        edge_widths = []
        for u, v in G.edges():
            # Check if this is a critical edge
            is_critical = False
            if critical_edges:
                for edge in critical_edges:
                    if (edge[0] == u and edge[1] == v):
                        is_critical = True
                        break

            edge_colors.append('red' if is_critical else 'gray')
            edge_widths.append(3 if is_critical else 1)

        nx.draw_networkx_edges(
            G, pos, edge_color=edge_colors, width=edge_widths,
            arrows=True, arrowsize=20, alpha=0.7, ax=ax
        )

        # Draw nodes
        node_colors = vulnerability_scores
        nodes = nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=2000,
            cmap='RdYlBu_r', vmin=0, vmax=1, ax=ax
        )

        # Add labels
        labels = {i: name for i, name in enumerate(agent_names)}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

        # Add colorbar
        plt.colorbar(nodes, ax=ax, label='Vulnerability Score')

        # Set title
        if title is None:
            title = f"{chain_topology.capitalize()} Chain Vulnerability"
        ax.set_title(title, fontsize=16, fontweight='bold')

        ax.axis('off')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved chain graph to {save_path}")

        return fig

    def plot_attack_propagation_flow(self,
                                    propagation_metrics: Dict[str, Any],
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create multi-panel visualization of attack propagation.

        Args:
            propagation_metrics: Dictionary with propagation analysis results
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Propagation Depth Distribution
        ax1 = axes[0, 0]
        depths = propagation_metrics.get('attack_analysis', {}).get('propagation_depths', [])
        if depths:
            ax1.hist(depths, bins=min(10, len(set(depths))), edgecolor='black', alpha=0.7)
            ax1.set_xlabel('Propagation Depth')
            ax1.set_ylabel('Number of Attacks')
            ax1.set_title('Attack Propagation Depth Distribution')
            ax1.axvline(np.mean(depths), color='red', linestyle='--',
                       label=f'Mean: {np.mean(depths):.1f}')
            ax1.legend()

        # Panel 2: Amplification Factors
        ax2 = axes[0, 1]
        amplifications = propagation_metrics.get('attack_analysis', {}).get('amplification_factors', [])
        if amplifications:
            attack_indices = range(len(amplifications))
            colors = ['red' if a > 0 else 'blue' for a in amplifications]
            ax2.bar(attack_indices, amplifications, color=colors, alpha=0.7)
            ax2.set_xlabel('Attack Index')
            ax2.set_ylabel('Amplification Factor')
            ax2.set_title('Attack Amplification Across Chain')
            ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)

        # Panel 3: Position Vulnerability
        ax3 = axes[1, 0]
        position_vulns = propagation_metrics.get('position_analysis', {}).get('position_vulnerabilities', [])
        if position_vulns:
            positions = range(len(position_vulns))
            ax3.plot(positions, position_vulns, 'o-', markersize=10, linewidth=2)
            ax3.fill_between(positions, position_vulns, alpha=0.3)
            ax3.set_xlabel('Agent Position')
            ax3.set_ylabel('Vulnerability Score')
            ax3.set_title('Vulnerability by Position')
            ax3.grid(True, alpha=0.3)

        # Panel 4: Attack Severity Ranking
        ax4 = axes[1, 1]
        vuln_scores = propagation_metrics.get('attack_analysis', {}).get('vulnerability_scores', [])
        if vuln_scores:
            sorted_indices = np.argsort(vuln_scores)[::-1]
            sorted_scores = [vuln_scores[i] for i in sorted_indices]
            y_pos = range(len(sorted_scores))
            ax4.barh(y_pos, sorted_scores, alpha=0.7)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([f"Attack {i}" for i in sorted_indices])
            ax4.set_xlabel('Vulnerability Score')
            ax4.set_title('Attack Severity Ranking')
            ax4.invert_yaxis()

        plt.suptitle('Attack Propagation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved propagation flow to {save_path}")

        return fig

    def plot_critical_paths(self,
                           agent_names: List[str],
                           critical_edges: List[Tuple[int, int, float]],
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize critical attack paths.

        Args:
            agent_names: List of agent names
            critical_edges: List of (from, to, amplification) tuples
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if not critical_edges:
            ax.text(0.5, 0.5, 'No Critical Paths Identified',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            ax.axis('off')
        else:
            # Create directed graph of critical paths
            G = nx.DiGraph()

            for from_idx, to_idx, amp in critical_edges:
                G.add_edge(from_idx, to_idx, weight=amp)

            # Position nodes
            pos = nx.spring_layout(G, k=2, iterations=50)

            # Draw edges with width proportional to amplification
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1

            nx.draw_networkx_edges(
                G, pos, edge_color=weights, width=[5 * w / max_weight for w in weights],
                edge_cmap=plt.cm.Reds, arrows=True, arrowsize=20, ax=ax
            )

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                 node_size=1500, ax=ax)

            # Add labels
            labels = {n: agent_names[n] if n < len(agent_names) else f"Agent {n}"
                     for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

            # Add edge labels with amplification values
            edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

            ax.set_title('Critical Attack Propagation Paths', fontsize=16, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved critical paths to {save_path}")

        return fig

    def create_attack_report_figure(self,
                                  dense_results: Dict[str, Any],
                                  sparse_results: Dict[str, Any],
                                  model_results: Optional[Dict[str, Any]] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive attack analysis report figure.

        Args:
            dense_results: Dense evaluation results
            sparse_results: Sparse evaluation results
            model_results: Optional vulnerability model results
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Top row: Dense heatmap
        ax1 = fig.add_subplot(gs[0, :])
        if 'matrix' in dense_results:
            matrix = dense_results['matrix']
            im = ax1.imshow(matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            ax1.set_xlabel('Agent Index')
            ax1.set_ylabel('Attack Index')
            ax1.set_title('Dense TVD-MI Evaluation', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax1, label='TVD-MI Score')

        # Middle left: Coverage comparison
        ax2 = fig.add_subplot(gs[1, 0])
        if sparse_results:
            strategies = list(sparse_results.keys())
            coverages = [r.get('coverage', 0) * 100 for r in sparse_results.values()]
            ax2.bar(strategies, coverages, alpha=0.7)
            ax2.set_ylabel('Coverage (%)')
            ax2.set_title('Sampling Coverage by Strategy')
            ax2.set_ylim(0, 100)

        # Middle center: Correlation comparison
        ax3 = fig.add_subplot(gs[1, 1])
        if sparse_results:
            correlations = [r.get('ranking_correlation', 0) for r in sparse_results.values()]
            ax3.bar(strategies, correlations, alpha=0.7, color='green')
            ax3.set_ylabel('Spearman Correlation')
            ax3.set_title('Ranking Preservation')
            ax3.set_ylim(0, 1)
            ax3.axhline(0.8, color='red', linestyle='--', label='Target: 0.8')
            ax3.legend()

        # Middle right: Efficiency
        ax4 = fig.add_subplot(gs[1, 2])
        if sparse_results:
            efficiencies = [r.get('n_tests', 0) for r in sparse_results.values()]
            ax4.bar(strategies, efficiencies, alpha=0.7, color='orange')
            ax4.set_ylabel('Number of Tests')
            ax4.set_title('Evaluation Efficiency')

        # Bottom: Model predictions if available
        if model_results:
            ax5 = fig.add_subplot(gs[2, :])
            if 'predictions' in model_results:
                im = ax5.imshow(model_results['predictions'], cmap='RdYlBu_r',
                              aspect='auto', vmin=0, vmax=1)
                ax5.set_xlabel('Agent Index')
                ax5.set_ylabel('Attack Index')
                ax5.set_title('Vulnerability Model Predictions', fontsize=14, fontweight='bold')
                plt.colorbar(im, ax=ax5, label='Predicted Vulnerability')

        plt.suptitle('AgentRedChain Attack Analysis Report', fontsize=16, fontweight='bold')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attack report to {save_path}")

        return fig