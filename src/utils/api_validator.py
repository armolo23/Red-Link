"""
API validation and fallback management for production deployment.
Ensures robust API operations with automatic fallback to alternative models.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from functools import wraps
import requests
try:
    from langchain.schema import BaseLanguageModel
except ImportError:
    from langchain_core.language_models import BaseLanguageModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIEndpoint:
    """Configuration for an API endpoint."""
    provider: str
    base_url: str
    health_check_url: Optional[str]
    required_env_var: str
    rate_limit_rpm: int  # Requests per minute
    timeout_seconds: int
    retry_count: int


class APIValidator:
    """
    Validates API availability and manages fallback logic for 2025 models.
    """

    ENDPOINTS = {
        'openai': APIEndpoint(
            provider='openai',
            base_url='https://api.openai.com/v1',
            health_check_url='https://api.openai.com/v1/models',
            required_env_var='OPENAI_API_KEY',
            rate_limit_rpm=500,  # GPT-5 tier limits
            timeout_seconds=60,
            retry_count=3
        ),
        'anthropic': APIEndpoint(
            provider='anthropic',
            base_url='https://api.anthropic.com/v1',
            health_check_url='https://api.anthropic.com/v1/models',
            required_env_var='ANTHROPIC_API_KEY',
            rate_limit_rpm=1000,  # Claude Sonnet 4.5 tier limits
            timeout_seconds=60,
            retry_count=3
        ),
        'xai': APIEndpoint(
            provider='xai',
            base_url='https://api.x.ai/v1',
            health_check_url='https://api.x.ai/v1/models',
            required_env_var='XAI_API_KEY',
            rate_limit_rpm=200,  # Grok 4 estimated limits
            timeout_seconds=60,
            retry_count=3
        ),
        'huggingface': APIEndpoint(
            provider='huggingface',
            base_url='https://api-inference.huggingface.co',
            health_check_url=None,  # No standard health check
            required_env_var='HUGGINGFACE_TOKEN',
            rate_limit_rpm=100,
            timeout_seconds=120,  # Longer for model loading
            retry_count=2
        )
    }

    # Fallback chain for 2025 models
    FALLBACK_CHAIN = [
        ('claude-sonnet-4.5', 'anthropic'),  # Primary: Most cost-effective
        ('gpt-5', 'openai'),                  # Secondary: High capability
        ('grok-4-fast', 'xai'),               # Tertiary: Fast alternative
        # Legacy models (gpt-4, gpt-3.5-turbo) automatically map to gpt-5
        ('llama-3.1-8b', 'huggingface'),      # Open-source last resort
    ]

    def __init__(self):
        """Initialize the API validator."""
        self.api_status: Dict[str, bool] = {}
        self.last_check: Dict[str, float] = {}
        self.request_counts: Dict[str, int] = {}
        self.request_timestamps: Dict[str, List[float]] = {}

        # Check all APIs on initialization
        self.validate_all_apis()

    def validate_all_apis(self) -> Dict[str, bool]:
        """
        Validate all configured API endpoints.

        Returns:
            Dictionary of provider -> availability status
        """
        logger.info("Validating API endpoints...")
        results = {}

        for provider, endpoint in self.ENDPOINTS.items():
            results[provider] = self.check_api_availability(provider)

        self.api_status = results
        logger.info(f"API validation complete: {results}")
        return results

    def check_api_availability(self, provider: str) -> bool:
        """
        Check if an API is available and properly configured.

        Args:
            provider: API provider name

        Returns:
            True if API is available
        """
        endpoint = self.ENDPOINTS.get(provider)
        if not endpoint:
            logger.warning(f"Unknown provider: {provider}")
            return False

        # Check environment variable
        api_key = os.getenv(endpoint.required_env_var)
        if not api_key:
            logger.warning(f"{provider}: API key not found ({endpoint.required_env_var})")
            return False

        # Special handling for OpenAI to check via SDK
        if provider == 'openai':
            return self.check_openai()

        # Skip health check if not available
        if not endpoint.health_check_url:
            logger.info(f"{provider}: No health check available, assuming available")
            return True

        # Perform health check
        try:
            headers = self._get_auth_headers(provider, api_key)
            response = requests.get(
                endpoint.health_check_url,
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"{provider}: API is available")
                return True
            elif response.status_code == 401:
                logger.warning(f"{provider}: Authentication failed - Invalid API key")
                logger.warning(f"  Please check your {endpoint.required_env_var} in the .env file")
                return False
            else:
                logger.warning(f"{provider}: Health check failed (status: {response.status_code})")
                return False

        except requests.exceptions.RequestException as e:
            logger.warning(f"{provider}: Health check failed ({str(e)})")
            return False

    def check_openai(self) -> bool:
        """Check OpenAI API availability using the SDK."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return False

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            # Try a simple API call to verify the key works
            response = client.models.list()
            logger.info("OpenAI: API is available")
            return True
        except Exception as e:
            error_msg = str(e)
            if '401' in error_msg or 'invalid' in error_msg.lower():
                logger.warning("OpenAI: Invalid API key - please check your OPENAI_API_KEY")
                logger.warning("  Get a valid key from: https://platform.openai.com/api-keys")
            elif 'quota' in error_msg.lower():
                logger.warning("OpenAI: Quota exceeded or no credits available")
            else:
                logger.warning(f"OpenAI: Health check failed ({error_msg[:100]})")
            return False

    def _get_auth_headers(self, provider: str, api_key: str) -> Dict[str, str]:
        """
        Get authentication headers for a provider.

        Args:
            provider: API provider name
            api_key: API key

        Returns:
            Headers dictionary
        """
        if provider == 'openai' or provider == 'xai':
            return {'Authorization': f'Bearer {api_key}'}
        elif provider == 'anthropic':
            return {'x-api-key': api_key, 'anthropic-version': '2023-06-01'}
        elif provider == 'huggingface':
            return {'Authorization': f'Bearer {api_key}'}
        else:
            return {}

    def get_available_model(self, preferred_model: Optional[str] = None) -> tuple[str, str]:
        """
        Get an available model, using fallback if necessary.

        Args:
            preferred_model: Preferred model name

        Returns:
            Tuple of (model_name, provider)
        """
        # Try preferred model first
        if preferred_model:
            for model, provider in self.FALLBACK_CHAIN:
                if model == preferred_model and self.api_status.get(provider, False):
                    return model, provider

        # Use fallback chain
        for model, provider in self.FALLBACK_CHAIN:
            if self.api_status.get(provider, False):
                logger.info(f"Using {model} from {provider}")
                return model, provider

        raise RuntimeError("No APIs available. Please configure at least one API key.")

    def rate_limit_check(self, provider: str) -> bool:
        """
        Check if we're within rate limits for a provider.

        Args:
            provider: API provider name

        Returns:
            True if within rate limits
        """
        endpoint = self.ENDPOINTS.get(provider)
        if not endpoint:
            return False

        # Initialize tracking if needed
        if provider not in self.request_timestamps:
            self.request_timestamps[provider] = []

        # Remove timestamps older than 1 minute
        current_time = time.time()
        self.request_timestamps[provider] = [
            t for t in self.request_timestamps[provider]
            if current_time - t < 60
        ]

        # Check if we're under the limit
        return len(self.request_timestamps[provider]) < endpoint.rate_limit_rpm

    def record_request(self, provider: str) -> None:
        """
        Record an API request for rate limiting.

        Args:
            provider: API provider name
        """
        if provider not in self.request_timestamps:
            self.request_timestamps[provider] = []

        self.request_timestamps[provider].append(time.time())

        if provider not in self.request_counts:
            self.request_counts[provider] = 0
        self.request_counts[provider] += 1

    def with_retry(self, func: Callable, provider: str, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute
            provider: API provider name
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        endpoint = self.ENDPOINTS.get(provider)
        if not endpoint:
            raise ValueError(f"Unknown provider: {provider}")

        last_error = None
        for attempt in range(endpoint.retry_count):
            try:
                # Check rate limits
                if not self.rate_limit_check(provider):
                    wait_time = 60 / endpoint.rate_limit_rpm
                    logger.info(f"Rate limit reached for {provider}, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)

                # Execute function
                result = func(*args, **kwargs)
                self.record_request(provider)
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"{provider} attempt {attempt + 1} failed: {str(e)}")

                if attempt < endpoint.retry_count - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        raise last_error

    def get_model_with_fallback(self, agent_pool, preferred_model: str = None,
                               **kwargs) -> BaseLanguageModel:
        """
        Get a model instance with automatic fallback.

        Args:
            agent_pool: AgentPool instance
            preferred_model: Preferred model name
            **kwargs: Additional model parameters

        Returns:
            Configured language model instance
        """
        model_name, provider = self.get_available_model(preferred_model)

        try:
            return agent_pool.get_agent(model_name, **kwargs)
        except Exception as e:
            logger.error(f"Failed to get {model_name}: {str(e)}")

            # Mark provider as unavailable and try fallback
            self.api_status[provider] = False

            # Recursive call with fallback
            if any(self.api_status.values()):
                logger.info("Attempting fallback...")
                return self.get_model_with_fallback(agent_pool, None, **kwargs)
            else:
                raise RuntimeError("All API providers failed")

    def estimate_experiment_cost(self, n_agents: int, n_attacks: int,
                                coverage: float = 0.33) -> Dict[str, float]:
        """
        Estimate cost for an experiment.

        Args:
            n_agents: Number of agents
            n_attacks: Number of attacks
            coverage: Sampling coverage (0 to 1)

        Returns:
            Cost estimates by provider
        """
        # Estimate tokens per evaluation
        avg_input_tokens = 500  # Attack prompt + context
        avg_output_tokens = 200  # Agent response

        total_evaluations = int(n_agents * n_attacks * coverage)
        total_input_tokens = total_evaluations * avg_input_tokens
        total_output_tokens = total_evaluations * avg_output_tokens

        costs = {}
        pricing = {
            'gpt-5': {'input': 0.01, 'output': 0.03},
            'claude-sonnet-4.5': {'input': 0.003, 'output': 0.015},
            'grok-4': {'input': 0.005, 'output': 0.015},
            'grok-4-fast': {'input': 0.002, 'output': 0.008},
            # Legacy models (gpt-4-turbo, gpt-4, gpt-3.5-turbo) map to gpt-5 pricing
            'llama-3.1-8b': {'input': 0.0, 'output': 0.0},  # Free if local
        }

        for model, prices in pricing.items():
            cost = (total_input_tokens * prices['input'] / 1000 +
                   total_output_tokens * prices['output'] / 1000)
            costs[model] = round(cost, 2)

        return costs

    def get_status_report(self) -> Dict[str, Any]:
        """
        Get comprehensive status report.

        Returns:
            Status report dictionary
        """
        return {
            'api_availability': self.api_status,
            'total_requests': self.request_counts,
            'available_models': [
                model for model, provider in self.FALLBACK_CHAIN
                if self.api_status.get(provider, False)
            ],
            'primary_model': self.get_available_model()[0] if any(self.api_status.values()) else None,
            'fallback_available': sum(self.api_status.values()) > 1
        }


def rate_limited(provider: str):
    """
    Decorator for rate-limited API calls.

    Args:
        provider: API provider name

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            validator = getattr(self, '_api_validator', None)
            if not validator:
                # Create validator if not exists
                self._api_validator = APIValidator()
                validator = self._api_validator

            return validator.with_retry(func, provider, self, *args, **kwargs)
        return wrapper
    return decorator


# Singleton instance for global use
_validator_instance = None


def get_validator() -> APIValidator:
    """Get or create the global API validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = APIValidator()
    return _validator_instance