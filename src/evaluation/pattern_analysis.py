"""
Pattern discovery and analysis for identifying high-value tests in sparse evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternDiscovery:
    """Discover patterns in dense evaluation results to inform sparse sampling."""

    def __init__(self, dense_results: Dict[str, Any]):
        """
        Initialize pattern discovery with dense evaluation results.

        Args:
            dense_results: Dictionary containing:
                - 'matrix': TVD-MI matrix
                - 'attacks': List of attack names
                - 'agents': List of agent names
                - Other metadata
        """
        self.results = dense_results
        self.matrix = np.array(dense_results.get('matrix', []))
        self.attacks = dense_results.get('attacks', [])
        self.agents = dense_results.get('agents', [])
        self.num_attacks, self.num_agents = self.matrix.shape

        logger.info(f"Initialized pattern discovery for {self.num_attacks} attacks × {self.num_agents} agents")

    def correlation_analysis(self) -> Dict[str, Any]:
        """
        Analyze correlations between attack types, positions, and propagation.

        Returns:
            Dictionary with correlation analysis results
        """
        correlations = {
            'attack_correlations': {},
            'position_correlations': {},
            'attack_position_interaction': {}
        }

        # Attack-wise correlations
        attack_scores = np.mean(self.matrix, axis=1)  # Mean score per attack
        attack_variances = np.var(self.matrix, axis=1)  # Variance per attack

        # Correlation between attack effectiveness and variance
        if len(attack_scores) > 1:
            corr, p_value = stats.pearsonr(attack_scores, attack_variances)
            correlations['attack_correlations']['effectiveness_variance'] = {
                'correlation': float(corr),
                'p_value': float(p_value),
                'interpretation': 'high' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.4 else 'low'
            }

        # Position-wise correlations
        position_scores = np.mean(self.matrix, axis=0)  # Mean score per position
        if self.num_agents > 1:
            # Correlation with position index (linear trend)
            positions = np.arange(self.num_agents)
            corr, p_value = stats.pearsonr(positions, position_scores)
            correlations['position_correlations']['linear_trend'] = {
                'correlation': float(corr),
                'p_value': float(p_value),
                'trend': 'increasing' if corr > 0 else 'decreasing'
            }

        # Attack-position interaction effects
        for i in range(min(3, self.num_attacks)):  # Top 3 attacks
            attack_pattern = self.matrix[i, :]
            if self.num_agents > 1:
                # Check if attack has position-dependent pattern
                corr, p_value = stats.pearsonr(positions, attack_pattern)
                correlations['attack_position_interaction'][f'attack_{i}'] = {
                    'correlation': float(corr),
                    'pattern': 'position-dependent' if abs(corr) > 0.5 else 'position-independent'
                }

        return correlations

    def identify_high_value_tests(self, top_percent: float = 0.3) -> np.ndarray:
        """
        Identify high-value test positions based on variance and discriminative power.

        Args:
            top_percent: Percentage of tests to consider high-value

        Returns:
            Array of indices for high-value test positions (flattened matrix indices)
        """
        # Compute variance for each position
        position_variances = np.var(self.matrix, axis=0)

        # Compute discriminative power (how well each position separates attacks)
        discriminative_scores = []
        for j in range(self.num_agents):
            agent_scores = self.matrix[:, j]
            # Use coefficient of variation as discriminative measure
            if np.mean(agent_scores) > 0:
                cv = np.std(agent_scores) / np.mean(agent_scores)
            else:
                cv = 0
            discriminative_scores.append(cv)

        discriminative_scores = np.array(discriminative_scores)

        # Combine variance and discriminative power
        combined_scores = position_variances * discriminative_scores

        # Get top positions
        num_high_value = int(top_percent * self.num_agents)
        high_value_positions = np.argsort(combined_scores)[::-1][:num_high_value]

        # Convert to flattened matrix indices (considering all attack-agent pairs)
        high_value_indices = []
        for pos in high_value_positions:
            # Add this position for all attacks
            for attack_idx in range(self.num_attacks):
                flat_idx = attack_idx * self.num_agents + pos
                high_value_indices.append(flat_idx)

        # Also add positions with extreme values
        extreme_indices = self._find_extreme_positions()
        high_value_indices.extend(extreme_indices)

        # Remove duplicates and sort
        high_value_indices = sorted(list(set(high_value_indices)))

        logger.info(f"Identified {len(high_value_indices)} high-value test positions")
        return np.array(high_value_indices[:int(top_percent * self.matrix.size)])

    def _find_extreme_positions(self, percentile: float = 95) -> List[int]:
        """
        Find positions with extreme TVD-MI values.

        Args:
            percentile: Percentile threshold for extreme values

        Returns:
            List of flattened matrix indices with extreme values
        """
        threshold_high = np.percentile(self.matrix, percentile)
        threshold_low = np.percentile(self.matrix, 100 - percentile)

        extreme_positions = []
        for i in range(self.num_attacks):
            for j in range(self.num_agents):
                value = self.matrix[i, j]
                if value >= threshold_high or value <= threshold_low:
                    flat_idx = i * self.num_agents + j
                    extreme_positions.append(flat_idx)

        return extreme_positions

    def attack_clustering_analysis(self) -> Dict[str, Any]:
        """
        Analyze attack clustering patterns to identify similar attack behaviors.

        Returns:
            Dictionary with clustering analysis results
        """
        from sklearn.cluster import KMeans

        # Standardize attack patterns
        scaler = StandardScaler()
        standardized_patterns = scaler.fit_transform(self.matrix)

        # Determine optimal number of clusters (up to 5)
        max_clusters = min(5, self.num_attacks)
        inertias = []
        silhouette_scores = []

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(standardized_patterns)
            inertias.append(kmeans.inertia_)

            from sklearn.metrics import silhouette_score
            if k < self.num_attacks:
                score = silhouette_score(standardized_patterns, labels)
                silhouette_scores.append(score)

        # Use elbow method to find optimal k
        optimal_k = 3 if len(silhouette_scores) > 0 else 2

        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(standardized_patterns)

        # Analyze clusters
        clusters = {}
        for cluster_id in range(optimal_k):
            cluster_attacks = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_patterns = self.matrix[cluster_attacks, :]

            clusters[f'cluster_{cluster_id}'] = {
                'attacks': cluster_attacks,
                'size': len(cluster_attacks),
                'mean_effectiveness': float(np.mean(cluster_patterns)),
                'std_effectiveness': float(np.std(cluster_patterns)),
                'representative_attack': int(cluster_attacks[0]) if cluster_attacks else -1
            }

        return {
            'optimal_clusters': optimal_k,
            'clusters': clusters,
            'cluster_labels': cluster_labels.tolist()
        }

    def position_importance_ranking(self) -> List[Tuple[int, float]]:
        """
        Rank agent positions by their importance for sparse sampling.

        Returns:
            List of (position_index, importance_score) tuples, sorted by importance
        """
        importance_scores = []

        for j in range(self.num_agents):
            # Factors contributing to importance
            variance = np.var(self.matrix[:, j])
            mean_impact = np.mean(self.matrix[:, j])
            max_impact = np.max(self.matrix[:, j])

            # Information content (entropy-like measure)
            values = self.matrix[:, j]
            if np.sum(values) > 0:
                normalized = values / np.sum(values)
                entropy = -np.sum(normalized * np.log(normalized + 1e-10))
            else:
                entropy = 0

            # Combined importance score
            importance = variance * 0.4 + mean_impact * 0.3 + max_impact * 0.2 + entropy * 0.1
            importance_scores.append((j, float(importance)))

        # Sort by importance (descending)
        importance_scores.sort(key=lambda x: x[1], reverse=True)

        return importance_scores

    def dimensionality_reduction(self, n_components: int = 2) -> Dict[str, Any]:
        """
        Perform PCA to understand main variation patterns.

        Args:
            n_components: Number of principal components

        Returns:
            Dictionary with PCA analysis results
        """
        # Standardize the data
        scaler = StandardScaler()
        standardized_matrix = scaler.fit_transform(self.matrix.T)  # Transpose to analyze agents

        # Perform PCA
        pca = PCA(n_components=min(n_components, min(self.matrix.shape)))
        transformed = pca.fit_transform(standardized_matrix)

        # Analyze components
        components_analysis = []
        for i in range(pca.n_components_):
            component = pca.components_[i]
            # Find attacks that contribute most to this component
            top_attacks = np.argsort(np.abs(component))[::-1][:3]

            components_analysis.append({
                'component': i,
                'explained_variance': float(pca.explained_variance_ratio_[i]),
                'top_contributing_attacks': top_attacks.tolist(),
                'interpretation': self._interpret_component(component)
            })

        return {
            'total_variance_explained': float(np.sum(pca.explained_variance_ratio_)),
            'components': components_analysis,
            'agent_coordinates': transformed.tolist()
        }

    def _interpret_component(self, component: np.ndarray) -> str:
        """
        Interpret a PCA component based on loadings.

        Args:
            component: PCA component loadings

        Returns:
            String interpretation of the component
        """
        # Simple interpretation based on loading patterns
        if np.all(component > 0):
            return "General vulnerability factor"
        elif np.sum(component > 0) > np.sum(component < 0):
            return "Majority attack susceptibility"
        else:
            return "Differential attack response"

    def generate_sampling_recommendations(self) -> Dict[str, Any]:
        """
        Generate comprehensive recommendations for sparse sampling.

        Returns:
            Dictionary with sampling recommendations
        """
        high_value_tests = self.identify_high_value_tests()
        position_ranking = self.position_importance_ranking()
        clustering = self.attack_clustering_analysis()
        correlations = self.correlation_analysis()

        recommendations = {
            'high_priority_tests': {
                'indices': high_value_tests.tolist()[:20],  # Top 20
                'coverage_percent': float(len(high_value_tests) / self.matrix.size * 100)
            },
            'critical_positions': [pos for pos, _ in position_ranking[:3]],
            'representative_attacks': [],
            'sampling_strategy': {}
        }

        # Get representative attacks from each cluster
        for cluster_info in clustering['clusters'].values():
            if cluster_info['representative_attack'] >= 0:
                recommendations['representative_attacks'].append(cluster_info['representative_attack'])

        # Determine sampling strategy based on analysis
        if correlations['position_correlations'].get('linear_trend', {}).get('correlation', 0) > 0.7:
            recommendations['sampling_strategy']['type'] = 'progressive'
            recommendations['sampling_strategy']['description'] = 'Focus on later positions due to strong linear trend'
        elif len(recommendations['representative_attacks']) < self.num_attacks / 2:
            recommendations['sampling_strategy']['type'] = 'cluster-based'
            recommendations['sampling_strategy']['description'] = 'Sample representative attacks from each cluster'
        else:
            recommendations['sampling_strategy']['type'] = 'variance-based'
            recommendations['sampling_strategy']['description'] = 'Focus on high-variance positions'

        return recommendations

    def save_analysis(self, path: str) -> None:
        """
        Save pattern analysis results to JSON file.

        Args:
            path: File path to save analysis
        """
        analysis = {
            'correlations': self.correlation_analysis(),
            'clustering': self.attack_clustering_analysis(),
            'position_importance': self.position_importance_ranking(),
            'pca_analysis': self.dimensionality_reduction(),
            'recommendations': self.generate_sampling_recommendations()
        }

        with open(path, 'w') as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Saved pattern analysis to {path}")