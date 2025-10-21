"""
Cost tracking for experiments with LLM models.
Pricing estimates based on publicly available API pricing.
Note: Actual costs may vary. Optimization effectiveness not validated.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Track token usage for a single API call."""
    model: str
    input_tokens: int
    output_tokens: int
    timestamp: str
    agent_role: Optional[str] = None
    attack_type: Optional[str] = None
    chain_topology: Optional[str] = None


@dataclass
class CostEstimate:
    """Cost estimate for an experiment."""
    total_cost: float
    cost_breakdown: Dict[str, float]
    token_usage: Dict[str, int]
    evaluation_count: int
    savings_from_sparsity: float
    estimated_duration_minutes: float


class CostTracker:
    """
    Track and optimize costs for AgentRedChain experiments with 2025 models.
    """

    # October 2025 Model Pricing (per 1K tokens) - ONLY latest models
    MODEL_PRICING = {
        'gpt-5': {
            'input': 0.01,
            'output': 0.03,
            'rps': 15,  # Requests per second
            'context_window': 256000
        },
        'claude-sonnet-4.5': {
            'input': 0.003,
            'output': 0.015,
            'rps': 25,
            'context_window': 200000
        },
        'grok-4': {
            'input': 0.005,
            'output': 0.015,
            'rps': 10,
            'context_window': 32768
        },
        'grok-4-fast': {
            'input': 0.002,
            'output': 0.008,
            'rps': 20,
            'context_window': 32768
        },
        'llama-3.1-8b': {
            'input': 0.0,
            'output': 0.0,
            'rps': 5,  # Depends on hardware
            'context_window': 8192
        },
    }

    # Token estimation parameters
    TOKEN_ESTIMATES = {
        'attack_prompt': {
            'goal_hijacking': 150,
            'data_exfiltration': 200,
            'privilege_escalation': 180,
            'jailbreak_propagation': 220,
            'subtle_poisoning': 160
        },
        'agent_context': {
            'linear': 300,  # Sequential context builds up
            'star': 200,    # Central coordinator
            'hierarchical': 250  # Tree structure
        },
        'average_response': 200  # Average agent response length
    }

    def __init__(self, budget_limit: Optional[float] = None,
                 log_file: Optional[str] = None):
        """
        Initialize the cost tracker.

        Args:
            budget_limit: Optional budget limit in USD
            log_file: Optional file to log costs
        """
        self.budget_limit = budget_limit
        self.log_file = log_file
        self.usage_history: List[TokenUsage] = []
        self.total_cost = 0.0
        self.total_tokens = {'input': 0, 'output': 0}

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def estimate_experiment_cost(self,
                                n_agents: int,
                                n_attacks: int,
                                chain_topology: str,
                                coverage: float = 0.33,
                                model: str = 'claude-sonnet-4.5') -> CostEstimate:
        """
        Estimate cost for a full experiment.

        Args:
            n_agents: Number of agents in chain
            n_attacks: Number of attack scenarios
            chain_topology: Chain topology type
            coverage: Sampling coverage (sparse evaluation)
            model: Model to use

        Returns:
            Cost estimate with breakdown
        """
        if model not in self.MODEL_PRICING:
            raise ValueError(f"Unknown model: {model}")

        pricing = self.MODEL_PRICING[model]

        # Calculate number of evaluations
        total_possible = n_agents * n_attacks
        actual_evaluations = int(total_possible * coverage)

        # Estimate tokens per evaluation
        avg_attack_tokens = np.mean(list(self.TOKEN_ESTIMATES['attack_prompt'].values()))
        context_tokens = self.TOKEN_ESTIMATES['agent_context'][chain_topology]
        input_tokens_per_eval = int(avg_attack_tokens + context_tokens)
        output_tokens_per_eval = self.TOKEN_ESTIMATES['average_response']

        # Calculate total tokens
        total_input_tokens = actual_evaluations * input_tokens_per_eval
        total_output_tokens = actual_evaluations * output_tokens_per_eval

        # Calculate costs
        input_cost = (total_input_tokens / 1000) * pricing['input']
        output_cost = (total_output_tokens / 1000) * pricing['output']
        total_cost = input_cost + output_cost

        # Calculate savings from sparsity
        full_cost = total_cost / coverage
        savings = full_cost - total_cost

        # Estimate duration
        requests_per_second = pricing['rps']
        estimated_seconds = actual_evaluations / requests_per_second
        estimated_minutes = estimated_seconds / 60

        return CostEstimate(
            total_cost=round(total_cost, 2),
            cost_breakdown={
                'input_cost': round(input_cost, 2),
                'output_cost': round(output_cost, 2),
                'per_evaluation': round(total_cost / actual_evaluations, 4)
            },
            token_usage={
                'total_input': total_input_tokens,
                'total_output': total_output_tokens,
                'total': total_input_tokens + total_output_tokens
            },
            evaluation_count=actual_evaluations,
            savings_from_sparsity=round(savings, 2),
            estimated_duration_minutes=round(estimated_minutes, 1)
        )

    def track_usage(self, model: str, input_tokens: int, output_tokens: int,
                   **metadata) -> float:
        """
        Track token usage and calculate cost.

        Args:
            model: Model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            **metadata: Additional metadata (agent_role, attack_type, etc.)

        Returns:
            Cost for this usage
        """
        if model not in self.MODEL_PRICING:
            logger.warning(f"Unknown model {model}, using default pricing")
            model = 'gpt-5'  # Fallback to GPT-5

        pricing = self.MODEL_PRICING[model]
        cost = (input_tokens * pricing['input'] / 1000 +
               output_tokens * pricing['output'] / 1000)

        # Record usage
        usage = TokenUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=datetime.now().isoformat(),
            **metadata
        )
        self.usage_history.append(usage)

        # Update totals
        self.total_cost += cost
        self.total_tokens['input'] += input_tokens
        self.total_tokens['output'] += output_tokens

        # Check budget
        if self.budget_limit and self.total_cost > self.budget_limit:
            logger.warning(f"Budget limit exceeded! Total: ${self.total_cost:.2f} > ${self.budget_limit:.2f}")

        # Log to file if configured
        if self.log_file:
            self._log_usage(usage, cost)

        return cost

    def _log_usage(self, usage: TokenUsage, cost: float) -> None:
        """Log usage to file."""
        log_entry = {
            **asdict(usage),
            'cost': round(cost, 4),
            'cumulative_cost': round(self.total_cost, 2)
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def optimize_model_selection(self,
                                required_context: int,
                                budget_remaining: float) -> str:
        """
        Optimize model selection based on context and budget.

        Args:
            required_context: Required context window size
            budget_remaining: Remaining budget in USD

        Returns:
            Recommended model name
        """
        suitable_models = []

        for model_name, specs in self.MODEL_PRICING.items():
            # Check context window
            if specs['context_window'] < required_context:
                continue

            # Estimate cost for 100 evaluations
            est_cost = (500 * specs['input'] / 1000 +
                       200 * specs['output'] / 1000) * 100

            # Check if within budget
            if est_cost <= budget_remaining:
                # Score based on cost-effectiveness and speed
                score = (1 / est_cost) * specs['rps']
                suitable_models.append((model_name, score))

        if not suitable_models:
            logger.warning("No models within budget, using cheapest option")
            return 'llama-3.1-8b'

        # Sort by score and return best
        suitable_models.sort(key=lambda x: x[1], reverse=True)
        recommended = suitable_models[0][0]

        logger.info(f"Recommended model: {recommended} based on budget ${budget_remaining:.2f}")
        return recommended

    def get_cost_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive cost report.

        Returns:
            Cost report dictionary
        """
        model_breakdown = {}
        for usage in self.usage_history:
            if usage.model not in model_breakdown:
                model_breakdown[usage.model] = {
                    'count': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'cost': 0
                }

            pricing = self.MODEL_PRICING.get(usage.model, self.MODEL_PRICING['gpt-5'])
            cost = (usage.input_tokens * pricing['input'] / 1000 +
                   usage.output_tokens * pricing['output'] / 1000)

            model_breakdown[usage.model]['count'] += 1
            model_breakdown[usage.model]['input_tokens'] += usage.input_tokens
            model_breakdown[usage.model]['output_tokens'] += usage.output_tokens
            model_breakdown[usage.model]['cost'] += cost

        # Round costs
        for model_data in model_breakdown.values():
            model_data['cost'] = round(model_data['cost'], 2)

        return {
            'total_cost': round(self.total_cost, 2),
            'total_tokens': self.total_tokens,
            'evaluation_count': len(self.usage_history),
            'model_breakdown': model_breakdown,
            'average_cost_per_evaluation': round(
                self.total_cost / len(self.usage_history), 4
            ) if self.usage_history else 0,
            'budget_status': {
                'limit': self.budget_limit,
                'remaining': round(self.budget_limit - self.total_cost, 2)
                if self.budget_limit else None,
                'percentage_used': round(
                    (self.total_cost / self.budget_limit) * 100, 1
                ) if self.budget_limit else None
            }
        }

    def compare_model_costs(self,
                           n_evaluations: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Compare costs across all available models.

        Args:
            n_evaluations: Number of evaluations to compare

        Returns:
            Comparison of model costs
        """
        comparison = {}

        # Standard token estimates
        input_tokens = 500
        output_tokens = 200

        for model_name, pricing in self.MODEL_PRICING.items():
            total_cost = n_evaluations * (
                input_tokens * pricing['input'] / 1000 +
                output_tokens * pricing['output'] / 1000
            )

            time_estimate = n_evaluations / pricing['rps'] / 60  # in minutes

            comparison[model_name] = {
                'total_cost': round(total_cost, 2),
                'per_evaluation': round(total_cost / n_evaluations, 4),
                'time_minutes': round(time_estimate, 1),
                'cost_per_minute': round(total_cost / time_estimate, 2) if time_estimate > 0 else 0
            }

        return comparison

    def recommend_coverage(self,
                          budget: float,
                          n_agents: int,
                          n_attacks: int,
                          model: str = 'claude-sonnet-4.5') -> float:
        """
        Recommend optimal coverage based on budget.

        Args:
            budget: Available budget in USD
            n_agents: Number of agents
            n_attacks: Number of attacks
            model: Model to use

        Returns:
            Recommended coverage percentage
        """
        # Try different coverage levels
        coverage_options = [0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        recommendations = []

        for coverage in coverage_options:
            estimate = self.estimate_experiment_cost(
                n_agents, n_attacks, 'linear', coverage, model
            )

            if estimate.total_cost <= budget:
                # Calculate quality score (higher coverage = better)
                quality_score = coverage

                # Penalize if too close to budget
                budget_usage = estimate.total_cost / budget
                if budget_usage > 0.9:
                    quality_score *= 0.8

                recommendations.append((coverage, quality_score, estimate))

        if not recommendations:
            # Budget too low, return minimum coverage
            return 0.33

        # Sort by quality score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        best_coverage, _, best_estimate = recommendations[0]

        logger.info(f"Recommended coverage: {best_coverage:.0%} "
                   f"(cost: ${best_estimate.total_cost:.2f}, "
                   f"savings: ${best_estimate.savings_from_sparsity:.2f})")

        return best_coverage

    def save_report(self, filepath: str) -> None:
        """Save cost report to file."""
        report = self.get_cost_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Cost report saved to {filepath}")

    def reset(self) -> None:
        """Reset all tracking data."""
        self.usage_history = []
        self.total_cost = 0.0
        self.total_tokens = {'input': 0, 'output': 0}
        logger.info("Cost tracker reset")


# Singleton instance for global tracking
_tracker_instance = None


def get_tracker(budget_limit: Optional[float] = None) -> CostTracker:
    """Get or create the global cost tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = CostTracker(budget_limit=budget_limit)
    return _tracker_instance
