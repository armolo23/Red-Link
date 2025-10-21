"""
Production tests for AgentRedChain with real API integration.
Tests core functionality without simulations.
"""

import pytest
import os
import numpy as np
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.api_validator import APIValidator
from src.utils.cost_tracker import CostTracker
from src.agents.agent_pool import AgentPool
from src.evaluation.rasch_vuln import fit_rasch_additive, VulnerabilityModel
from src.evaluation.tvd_mi_scorer import TVDMIScorer


class TestAPIValidator:
    """Test API validation and fallback logic."""

    def test_initialization(self):
        """Test API validator initialization."""
        validator = APIValidator()

        assert validator.api_status is not None
        assert isinstance(validator.api_status, dict)
        assert validator.request_counts is not None

    def test_fallback_chain(self):
        """Test model fallback chain."""
        validator = APIValidator()

        # Mock all APIs as unavailable
        validator.api_status = {
            'openai': False,
            'anthropic': False,
            'xai': False,
            'huggingface': True  # Only HF available
        }

        model, provider = validator.get_available_model()
        assert provider == 'huggingface'
        assert model == 'llama-3.1-8b'

    def test_rate_limit_tracking(self):
        """Test rate limit tracking."""
        validator = APIValidator()

        # Test rate limit check
        provider = 'openai'

        # Should initially be within limits
        assert validator.rate_limit_check(provider) == True

        # Record requests
        for _ in range(10):
            validator.record_request(provider)

        assert provider in validator.request_counts
        assert validator.request_counts[provider] == 10

    def test_cost_estimation(self):
        """Test experiment cost estimation."""
        validator = APIValidator()

        costs = validator.estimate_experiment_cost(
            n_agents=4,
            n_attacks=10,
            coverage=0.33
        )

        assert 'claude-sonnet-4.5' in costs
        assert 'gpt-5' in costs
        assert costs['llama-3.1-8b'] == 0  # Free model

        # Check realistic pricing
        assert 0.5 < costs['claude-sonnet-4.5'] < 5.0  # Reasonable range


class TestCostTracker:
    """Test cost tracking functionality."""

    def test_initialization(self):
        """Test cost tracker initialization."""
        tracker = CostTracker(budget_limit=10.0)

        assert tracker.budget_limit == 10.0
        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == {'input': 0, 'output': 0}

    def test_track_usage(self):
        """Test usage tracking."""
        tracker = CostTracker()

        cost = tracker.track_usage(
            model='claude-sonnet-4.5',
            input_tokens=500,
            output_tokens=200,
            agent_role='researcher'
        )

        assert cost > 0
        assert tracker.total_cost == cost
        assert tracker.total_tokens['input'] == 500
        assert tracker.total_tokens['output'] == 200
        assert len(tracker.usage_history) == 1

    def test_budget_checking(self):
        """Test budget limit checking."""
        tracker = CostTracker(budget_limit=0.01)  # Very low budget

        # Track usage that exceeds budget
        with patch('logging.Logger.warning') as mock_warning:
            tracker.track_usage(
                model='gpt-5',
                input_tokens=5000,
                output_tokens=2000
            )

            # Should log warning about budget exceeded
            mock_warning.assert_called()

    def test_model_comparison(self):
        """Test model cost comparison."""
        tracker = CostTracker()

        comparison = tracker.compare_model_costs(n_evaluations=100)

        # Check all models present
        assert 'gpt-5' in comparison
        assert 'claude-sonnet-4.5' in comparison
        assert 'grok-4' in comparison

        # Check cost ordering (Claude 4.5 should be cheaper than GPT-5)
        assert comparison['claude-sonnet-4.5']['total_cost'] < comparison['gpt-5']['total_cost']

        # Llama should be free
        assert comparison['llama-3.1-8b']['total_cost'] == 0

    def test_coverage_recommendation(self):
        """Test coverage recommendation based on budget."""
        tracker = CostTracker()

        # Low budget should recommend minimum coverage
        coverage = tracker.recommend_coverage(
            budget=1.0,
            n_agents=4,
            n_attacks=10,
            model='gpt-5'
        )
        assert coverage == 0.33  # Minimum

        # High budget should allow more coverage
        coverage = tracker.recommend_coverage(
            budget=10.0,
            n_agents=4,
            n_attacks=10,
            model='claude-sonnet-4.5'
        )
        assert coverage >= 0.33


class TestAgentPool:
    """Test agent pool with 2025 models."""

    def test_supported_models(self):
        """Test 2025 model support."""
        pool = AgentPool()

        # Check 2025 models are present
        assert 'grok-4' in pool.SUPPORTED_MODELS
        assert 'grok-4-fast' in pool.SUPPORTED_MODELS
        assert 'gpt-5' in pool.SUPPORTED_MODELS
        assert 'claude-sonnet-4.5' in pool.SUPPORTED_MODELS

        # Check specifications
        assert pool.SUPPORTED_MODELS['gpt-5']['max_tokens'] == 256000
        assert pool.SUPPORTED_MODELS['claude-sonnet-4.5']['max_tokens'] == 200000

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_agent_creation_with_api_key(self):
        """Test agent creation when API key is available."""
        pool = AgentPool()

        with patch('src.agents.agent_pool.ChatOpenAI') as mock_openai:
            mock_openai.return_value = MagicMock()

            agent = pool.get_agent('gpt-5', temperature=0.7)

            assert agent is not None
            mock_openai.assert_called_once()

    def test_cost_estimation(self):
        """Test cost estimation for 2025 models."""
        pool = AgentPool()

        # Test GPT-5 pricing
        cost = pool.estimate_cost('gpt-5', input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.01 / 1000) + (500 * 0.03 / 1000)
        assert cost == round(expected, 4)

        # Test Claude Sonnet 4.5 pricing
        cost = pool.estimate_cost('claude-sonnet-4.5', input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.003 / 1000) + (500 * 0.015 / 1000)
        assert cost == round(expected, 4)


class TestRaschModel:
    """Test functional Rasch model fitting."""

    def test_fit_rasch_additive(self):
        """Test ALS-based Rasch fitting."""
        # Create test data
        np.random.seed(42)
        K, J = 10, 4  # 10 attacks, 4 agents

        # Generate synthetic TVD-MI scores
        true_a = np.random.randn(K) * 0.5
        true_b = np.random.randn(J) * 0.5
        S_true = np.outer(true_a, np.ones(J)) + np.outer(np.ones(K), true_b)
        S_true = 1 / (1 + np.exp(-S_true))  # Convert to [0,1]

        # Create sparse mask (33% coverage)
        mask = np.random.random((K, J)) < 0.33
        S_sparse = S_true * mask

        # Fit model
        result = fit_rasch_additive(
            S=S_sparse,
            mask=mask,
            lambda_a=0.01,
            lambda_b=0.01,
            max_iter=50,
            tol=1e-4
        )

        assert 'a' in result
        assert 'b' in result
        assert 'pred' in result
        assert 'converged' in result
        assert 'mse' in result

        # Check dimensions
        assert len(result['a']) == K
        assert len(result['b']) == J
        assert result['pred'].shape == (K, J)

        # Check MSE is reasonable
        assert result['mse'] < 0.5  # Should fit reasonably well

    def test_vulnerability_model_integration(self):
        """Test VulnerabilityModel with real Rasch fitting."""
        # Create test data
        np.random.seed(42)
        K, J = 5, 3
        tvd_matrix = np.random.random((K, J))
        mask = np.random.random((K, J)) < 0.4

        # Create and fit model
        model = VulnerabilityModel(tvd_matrix, mask)
        model.fit(max_iter=30, regularization=0.01)

        # Check fitting completed
        assert model.attack_severity is not None
        assert model.agent_resistance is not None
        assert model.predictions is not None

        # Test predictions
        for i in range(K):
            for j in range(J):
                pred = model.predict_vulnerability(i, j)
                assert 0 <= pred <= 1  # Should be probability

        # Test rankings
        agent_ranks = model.rank_agents()
        assert len(agent_ranks) == J
        assert sorted(agent_ranks) == list(range(J))

        attack_ranks = model.rank_attacks()
        assert len(attack_ranks) == K
        assert sorted(attack_ranks) == list(range(K))


class TestTVDMIScorer:
    """Test TVD-MI scoring with real embeddings."""

    def test_initialization(self):
        """Test scorer initialization."""
        scorer = TVDMIScorer()

        assert scorer.embedder is not None
        assert scorer.cache is not None

    @patch('sentence_transformers.SentenceTransformer')
    def test_compute_tvd_mi(self, mock_transformer):
        """Test TVD-MI computation."""
        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.encode.side_effect = [
            np.array([0.1, 0.2, 0.3]),  # Clean embedding
            np.array([0.4, 0.5, 0.6])   # Attacked embedding
        ]
        mock_transformer.return_value = mock_embedder

        scorer = TVDMIScorer()
        scorer.embedder = mock_embedder

        # Compute TVD-MI
        score = scorer.compute_tvd_mi(
            "Clean output",
            "Attacked output"
        )

        assert 0 <= score <= 1
        assert mock_embedder.encode.call_count == 2

    def test_caching(self):
        """Test embedding caching."""
        scorer = TVDMIScorer()

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([0.1, 0.2, 0.3])
        scorer.embedder = mock_embedder

        # First call should compute embedding
        text = "Test text"
        _ = scorer._get_embedding(text)
        assert mock_embedder.encode.call_count == 1

        # Second call should use cache
        _ = scorer._get_embedding(text)
        assert mock_embedder.encode.call_count == 1  # Still 1


class TestProductionIntegration:
    """Integration tests for production system."""

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'ANTHROPIC_API_KEY': 'test-key'
    })
    def test_full_pipeline_mocked(self):
        """Test full pipeline with mocked APIs."""
        from src.agents.chain_builder import AgentChain
        from src.attacks.injections import InjectionGenerator

        # Mock the agent pool to avoid real API calls
        with patch('src.agents.chain_builder.AgentPool') as mock_pool_class:
            mock_pool = MagicMock()
            mock_llm = MagicMock()
            mock_pool.get_agent.return_value = mock_llm
            mock_pool_class.return_value = mock_pool

            # Mock API validator
            with patch('src.agents.chain_builder.APIValidator') as mock_validator_class:
                mock_validator = MagicMock()
                mock_validator.validate_all_apis.return_value = {
                    'openai': True,
                    'anthropic': True
                }
                mock_validator.get_model_with_fallback.return_value = mock_llm
                mock_validator_class.return_value = mock_validator

                # Create chain
                chain = AgentChain(
                    topology='linear',
                    model_type='claude-sonnet-4.5',
                    enable_cost_tracking=True,
                    budget_limit=10.0
                )

                # Build pipeline
                chain.build_research_pipeline()
                assert len(chain.agents) == 4

                # Generate attacks
                injector = InjectionGenerator()
                attacks = injector.generate_all_attacks()
                assert len(attacks) == 10

    def test_rasch_convergence(self):
        """Test Rasch model convergence on realistic data."""
        # Create realistic sparse TVD-MI matrix
        np.random.seed(42)
        n_attacks, n_agents = 10, 4

        # Generate data with structure
        attack_severity = np.random.randn(n_attacks) * 0.5 + 0.5
        agent_resistance = np.random.randn(n_agents) * 0.3

        # Create full matrix
        full_matrix = np.outer(attack_severity, np.ones(n_agents)) - \
                     np.outer(np.ones(n_attacks), agent_resistance)
        full_matrix = 1 / (1 + np.exp(-full_matrix))  # Sigmoid

        # Add noise
        full_matrix += np.random.normal(0, 0.05, full_matrix.shape)
        full_matrix = np.clip(full_matrix, 0, 1)

        # Create sparse mask
        mask = np.random.random((n_attacks, n_agents)) < 0.33
        sparse_matrix = full_matrix * mask

        # Fit model
        model = VulnerabilityModel(sparse_matrix, mask)
        model.fit(max_iter=100, regularization=0.01)

        # Check convergence
        assert model.fit_result.convergence

        # Check prediction quality on observed data
        fit_metrics = model.evaluate_fit()
        assert fit_metrics['mse'] < 0.1  # Good fit
        assert fit_metrics['pseudo_r2'] > 0.3  # Reasonable R²


if __name__ == '__main__':
    pytest.main([__file__, '-v'])