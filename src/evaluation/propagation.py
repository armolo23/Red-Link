"""
Attack propagation analyzer for understanding how attacks spread through agent chains.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PropagationMetrics:
    """Container for propagation analysis metrics."""
    depth: List[int]
    amplification_factors: List[float]
    critical_edges: List[Tuple[int, int, float]]
    vulnerability_scores: np.ndarray
    position_vulnerabilities: np.ndarray
    attack_severity_ranking: List[int]
    agent_resistance_ranking: List[int]


class PropagationAnalyzer:
    """Analyze how attacks propagate through multi-agent chains."""

    def __init__(self, tvd_mi_matrix: np.ndarray, chain_topology: Optional[str] = None):
        """
        Initialize the propagation analyzer.

        Args:
            tvd_mi_matrix: TVD-MI matrix (attacks × agents)
            chain_topology: Optional chain topology type
        """
        self.matrix = tvd_mi_matrix
        self.chain_topology = chain_topology
        self.num_attacks, self.num_agents = tvd_mi_matrix.shape
        logger.info(f"Initialized propagation analyzer for {self.num_attacks} attacks × {self.num_agents} agents")

    def compute_depth(self, threshold: float = 0.3) -> List[int]:
        """
        Compute propagation depth for each attack.
        Depth is the number of agents significantly affected.

        Args:
            threshold: TVD-MI threshold for considering an agent affected

        Returns:
            List of propagation depths per attack
        """
        depths = []
        for attack_idx in range(self.num_attacks):
            scores = self.matrix[attack_idx]
            # Count agents with score above threshold
            depth = len([s for s in scores if s > threshold])
            depths.append(depth)

        logger.info(f"Computed propagation depths: mean={np.mean(depths):.2f}, max={max(depths)}")
        return depths

    def compute_amplification(self) -> List[float]:
        """
        Compute amplification factor for each attack.
        Measures how much the attack effect increases through the chain.

        Returns:
            List of amplification factors per attack
        """
        amplification_factors = []

        for attack_idx in range(self.num_attacks):
            scores = self.matrix[attack_idx]

            if self.num_agents > 1:
                # Calculate rate of change across agents
                differences = np.diff(scores)
                # Amplification is the mean positive change
                positive_changes = differences[differences > 0]

                if len(positive_changes) > 0:
                    amplification = np.mean(positive_changes)
                else:
                    amplification = 0.0
            else:
                amplification = 0.0

            amplification_factors.append(float(amplification))

        return amplification_factors

    def identify_critical_edges(self, significance_threshold: float = 0.1) -> List[Tuple[int, int, float]]:
        """
        Identify critical edges where attacks amplify significantly.

        Args:
            significance_threshold: Minimum amplification to consider edge critical

        Returns:
            List of (from_agent, to_agent, amplification) tuples
        """
        critical_edges = []

        # For each adjacent pair of agents
        for i in range(self.num_agents - 1):
            # Calculate mean amplification across all attacks
            amplifications = self.matrix[:, i + 1] - self.matrix[:, i]
            mean_amplification = np.mean(amplifications)

            if mean_amplification > significance_threshold:
                critical_edges.append((i, i + 1, float(mean_amplification)))

        # Sort by amplification strength
        critical_edges.sort(key=lambda x: x[2], reverse=True)

        logger.info(f"Found {len(critical_edges)} critical edges")
        return critical_edges

    def rank_vulnerabilities(self) -> np.ndarray:
        """
        Rank attacks by their overall effectiveness.

        Returns:
            Array of vulnerability scores per attack
        """
        # Mean TVD-MI score across all agents for each attack
        vulnerability_scores = np.mean(self.matrix, axis=1)
        return vulnerability_scores

    def compute_position_vulnerability(self) -> np.ndarray:
        """
        Compute vulnerability score for each agent position.

        Returns:
            Array of vulnerability scores per agent position
        """
        # Mean TVD-MI score across all attacks for each agent
        position_vulnerabilities = np.mean(self.matrix, axis=0)
        return position_vulnerabilities

    def find_attack_patterns(self) -> Dict[str, Any]:
        """
        Identify patterns in attack effectiveness.

        Returns:
            Dictionary of discovered patterns
        """
        patterns = {
            'most_effective_positions': [],
            'least_effective_positions': [],
            'consistent_attacks': [],
            'variable_attacks': [],
            'position_correlations': {}
        }

        # Find most/least vulnerable positions
        position_vuln = self.compute_position_vulnerability()
        sorted_positions = np.argsort(position_vuln)

        patterns['least_effective_positions'] = sorted_positions[:min(3, len(sorted_positions))].tolist()
        patterns['most_effective_positions'] = sorted_positions[-min(3, len(sorted_positions)):].tolist()

        # Find consistent vs variable attacks
        attack_variances = np.var(self.matrix, axis=1)
        sorted_variances = np.argsort(attack_variances)

        patterns['consistent_attacks'] = sorted_variances[:min(3, len(sorted_variances))].tolist()
        patterns['variable_attacks'] = sorted_variances[-min(3, len(sorted_variances)):].tolist()

        # Compute position correlations for linear chains
        if self.chain_topology == 'linear' and self.num_agents > 2:
            for i in range(self.num_agents - 1):
                correlation = np.corrcoef(self.matrix[:, i], self.matrix[:, i + 1])[0, 1]
                patterns['position_correlations'][f"{i}->{i+1}"] = float(correlation)

        return patterns

    def compute_attack_clusters(self, n_clusters: int = 3) -> Dict[str, List[int]]:
        """
        Cluster attacks based on their propagation patterns.

        Args:
            n_clusters: Number of clusters to identify

        Returns:
            Dictionary mapping cluster names to attack indices
        """
        from sklearn.cluster import KMeans

        # Use attack patterns as features
        features = self.matrix

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)

        # Organize results
        cluster_dict = {}
        for cluster_id in range(n_clusters):
            attack_indices = np.where(clusters == cluster_id)[0].tolist()
            cluster_dict[f"cluster_{cluster_id}"] = attack_indices

        return cluster_dict

    def analyze_chain_weaknesses(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of chain weaknesses.

        Returns:
            Dictionary with detailed weakness analysis
        """
        depths = self.compute_depth()
        amplifications = self.compute_amplification()
        critical_edges = self.identify_critical_edges()
        vulnerability_scores = self.rank_vulnerabilities()
        position_vulns = self.compute_position_vulnerability()

        analysis = {
            'summary': {
                'mean_propagation_depth': float(np.mean(depths)),
                'max_propagation_depth': int(max(depths)),
                'mean_amplification': float(np.mean(amplifications)),
                'num_critical_edges': len(critical_edges),
                'most_vulnerable_position': int(np.argmax(position_vulns)),
                'least_vulnerable_position': int(np.argmin(position_vulns))
            },
            'attack_analysis': {
                'propagation_depths': depths,
                'amplification_factors': amplifications,
                'vulnerability_scores': vulnerability_scores.tolist(),
                'most_effective_attack': int(np.argmax(vulnerability_scores)),
                'least_effective_attack': int(np.argmin(vulnerability_scores))
            },
            'position_analysis': {
                'position_vulnerabilities': position_vulns.tolist(),
                'critical_edges': critical_edges,
                'vulnerability_gradient': float(np.max(position_vulns) - np.min(position_vulns))
            },
            'patterns': self.find_attack_patterns()
        }

        # Add chain-specific analysis
        if self.chain_topology:
            analysis['chain_topology'] = self.chain_topology
            analysis['topology_specific'] = self._topology_specific_analysis()

        return analysis

    def _topology_specific_analysis(self) -> Dict[str, Any]:
        """
        Perform topology-specific vulnerability analysis.

        Returns:
            Topology-specific metrics
        """
        specific_analysis = {}

        if self.chain_topology == 'linear':
            # Analyze cascading effects in linear chains
            cascading_scores = []
            for i in range(self.num_agents - 1):
                cascade = np.mean(self.matrix[:, i + 1]) - np.mean(self.matrix[:, i])
                cascading_scores.append(float(cascade))

            specific_analysis['cascading_effect'] = {
                'scores': cascading_scores,
                'mean': float(np.mean(cascading_scores)),
                'strongest_cascade': int(np.argmax(cascading_scores)) if cascading_scores else -1
            }

        elif self.chain_topology == 'star':
            # Analyze coordinator vulnerability
            if self.num_agents > 0:
                coordinator_vuln = self.matrix[:, 0]
                specialist_vulns = self.matrix[:, 1:] if self.num_agents > 1 else np.array([])

                specific_analysis['coordinator_analysis'] = {
                    'coordinator_vulnerability': float(np.mean(coordinator_vuln)),
                    'specialist_mean_vulnerability': float(np.mean(specialist_vulns)) if specialist_vulns.size > 0 else 0,
                    'coordinator_is_weakest': bool(np.mean(coordinator_vuln) > np.mean(specialist_vulns))
                    if specialist_vulns.size > 0 else False
                }

        elif self.chain_topology == 'hierarchical':
            # Analyze hierarchy levels
            if self.num_agents >= 3:
                junior_vulns = self.matrix[:, :2]
                senior_vuln = self.matrix[:, 2]

                specific_analysis['hierarchy_analysis'] = {
                    'junior_vulnerability': float(np.mean(junior_vulns)),
                    'senior_vulnerability': float(np.mean(senior_vuln)),
                    'hierarchy_amplification': float(np.mean(senior_vuln) - np.mean(junior_vulns))
                }

        return specific_analysis

    def get_metrics(self) -> PropagationMetrics:
        """
        Get all propagation metrics as a structured object.

        Returns:
            PropagationMetrics dataclass with all computed metrics
        """
        depths = self.compute_depth()
        amplifications = self.compute_amplification()
        critical_edges = self.identify_critical_edges()
        vulnerability_scores = self.rank_vulnerabilities()
        position_vulns = self.compute_position_vulnerability()

        # Rank attacks and agents
        attack_ranking = np.argsort(-vulnerability_scores).tolist()  # Descending
        agent_ranking = np.argsort(position_vulns).tolist()  # Ascending (most resistant first)

        return PropagationMetrics(
            depth=depths,
            amplification_factors=amplifications,
            critical_edges=critical_edges,
            vulnerability_scores=vulnerability_scores,
            position_vulnerabilities=position_vulns,
            attack_severity_ranking=attack_ranking,
            agent_resistance_ranking=agent_ranking
        )

    def save_analysis(self, path: str) -> None:
        """
        Save propagation analysis to JSON file.

        Args:
            path: File path to save analysis
        """
        analysis = self.analyze_chain_weaknesses()

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        analysis = convert_to_serializable(analysis)

        with open(path, 'w') as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Saved propagation analysis to {path}")