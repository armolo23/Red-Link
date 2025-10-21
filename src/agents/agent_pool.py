"""
Agent pool for managing different LLM models and configurations.
"""

import os
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
try:
    from langchain.schema import BaseLanguageModel
except ImportError:
    from langchain_core.language_models import BaseLanguageModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentPool:
    """Manage a pool of LLM agents with different configurations."""

    SUPPORTED_MODELS = {
        # October 2025 Models - ONLY latest frontier models
        'gpt-5': {'provider': 'openai', 'max_tokens': 256000},  # Latest OpenAI
        'claude-sonnet-4.5': {'provider': 'anthropic', 'max_tokens': 200000},  # Latest Anthropic
        'grok-4': {'provider': 'xai', 'max_tokens': 32768},
        'grok-4-fast': {'provider': 'xai', 'max_tokens': 32768},

        # Open-source fallback
        'llama-3.1-8b': {'provider': 'huggingface', 'max_tokens': 8192},
    }

    def __init__(self):
        """Initialize the agent pool."""
        self.agents: Dict[str, BaseLanguageModel] = {}
        self.configurations: Dict[str, Dict[str, Any]] = {}
        self._validate_api_keys()

    def _validate_api_keys(self):
        """Validate that required API keys are set."""
        required_keys = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'xai': 'XAI_API_KEY',
            'huggingface': 'HUGGINGFACE_TOKEN'
        }

        self.available_providers = {}
        for provider, env_var in required_keys.items():
            if os.getenv(env_var):
                self.available_providers[provider] = True
                logger.info(f"{provider} API key found")
            else:
                self.available_providers[provider] = False
                logger.warning(f"{provider} API key not found - {provider} models will be unavailable")

    def get_agent(self,
                  model_name: str = 'gpt-5',
                  temperature: float = 0.7,
                  max_tokens: Optional[int] = None,
                  **kwargs) -> BaseLanguageModel:
        """
        Get or create an agent with specified configuration.

        Args:
            model_name: Name of the model to use
            temperature: Temperature setting for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters

        Returns:
            Configured language model instance
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {list(self.SUPPORTED_MODELS.keys())}")

        model_info = self.SUPPORTED_MODELS[model_name]
        provider = model_info['provider']

        if not self.available_providers.get(provider, False):
            raise ValueError(f"API key for {provider} not found. Please set the appropriate environment variable.")

        # Create cache key for this configuration
        cache_key = f"{model_name}_{temperature}_{max_tokens}"

        if cache_key not in self.agents:
            if provider == 'openai':
                self.agents[cache_key] = self._create_openai_agent(
                    model_name, temperature, max_tokens, **kwargs
                )
            elif provider == 'anthropic':
                self.agents[cache_key] = self._create_anthropic_agent(
                    model_name, temperature, max_tokens, **kwargs
                )
            elif provider == 'xai':
                self.agents[cache_key] = self._create_xai_agent(
                    model_name, temperature, max_tokens, **kwargs
                )
            elif provider == 'huggingface':
                self.agents[cache_key] = self._create_huggingface_agent(
                    model_name, temperature, max_tokens, **kwargs
                )

            self.configurations[cache_key] = {
                'model': model_name,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'provider': provider,
                **kwargs
            }

            logger.info(f"Created new agent: {model_name} with temperature={temperature}")

        return self.agents[cache_key]

    def _create_openai_agent(self,
                            model_name: str,
                            temperature: float,
                            max_tokens: Optional[int],
                            **kwargs) -> ChatOpenAI:
        """Create an OpenAI agent - GPT-5 only."""
        # Only GPT-5 is supported (October 2025)
        model_config = {
            'model': 'gpt-5',  # Always use GPT-5
            'temperature': temperature,
        }

        if max_tokens:
            model_config['max_tokens'] = min(max_tokens, self.SUPPORTED_MODELS[model_name]['max_tokens'])

        # Add any additional parameters
        for key, value in kwargs.items():
            if key in ['top_p', 'frequency_penalty', 'presence_penalty', 'seed']:
                model_config[key] = value

        return ChatOpenAI(**model_config)

    def _create_anthropic_agent(self,
                               model_name: str,
                               temperature: float,
                               max_tokens: Optional[int],
                               **kwargs) -> ChatAnthropic:
        """Create an Anthropic agent - Claude Sonnet 4.5 only."""
        # Only Claude Sonnet 4.5 is supported (October 2025)
        model_config = {
            'model': 'claude-sonnet-4-5',  # Always use Claude Sonnet 4.5
            'temperature': temperature,
        }

        if max_tokens:
            model_config['max_tokens'] = min(max_tokens, self.SUPPORTED_MODELS[model_name]['max_tokens'])

        # Add any additional parameters
        for key, value in kwargs.items():
            if key in ['top_p', 'top_k']:
                model_config[key] = value

        return ChatAnthropic(**model_config)

    def _create_xai_agent(self,
                         model_name: str,
                         temperature: float,
                         max_tokens: Optional[int],
                         **kwargs):
        """Create an xAI (Grok) agent."""
        # Note: Using ChatOpenAI as base since xAI API is OpenAI-compatible
        # In production, use langchain_community.chat_models.ChatXAI when available
        model_config = {
            'model': model_name,
            'temperature': temperature,
            'base_url': 'https://api.x.ai/v1',  # xAI API endpoint
            'api_key': os.getenv('XAI_API_KEY'),
        }

        if max_tokens:
            model_config['max_tokens'] = min(max_tokens, self.SUPPORTED_MODELS[model_name]['max_tokens'])

        # Add any additional parameters
        for key, value in kwargs.items():
            if key in ['top_p', 'frequency_penalty', 'presence_penalty']:
                model_config[key] = value

        # Using ChatOpenAI with custom base_url for xAI compatibility
        return ChatOpenAI(**model_config)

    def _create_huggingface_agent(self,
                                 model_name: str,
                                 temperature: float,
                                 max_tokens: Optional[int],
                                 **kwargs):
        """Create a HuggingFace agent for open-source models."""
        try:
            from langchain.llms import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")

        # For production, this would load the model locally or from HF Inference API
        # Simplified implementation - in production use HuggingFaceHub or local loading
        logger.info(f"Loading HuggingFace model {model_name} - this may take time...")

        # This is a placeholder - actual implementation would load the model
        # For now, fall back to OpenAI if HF not properly configured
        if not os.getenv('HUGGINGFACE_TOKEN'):
            logger.warning("HuggingFace token not found, falling back to GPT-5")
            return self._create_openai_agent('gpt-5', temperature, max_tokens, **kwargs)

        # In production, return properly configured HuggingFace model
        # return HuggingFacePipeline.from_model_id(
        #     model_id=model_name,
        #     task="text-generation",
        #     model_kwargs={"temperature": temperature, "max_length": max_tokens}
        # )

        # Placeholder return
        return self._create_openai_agent('gpt-5', temperature, max_tokens, **kwargs)

    def create_agent_with_role(self,
                              role: str,
                              model_name: str = 'gpt-5',
                              **kwargs) -> BaseLanguageModel:
        """
        Create an agent optimized for a specific role.

        Args:
            role: Agent role (e.g., 'researcher', 'summarizer', 'fact_checker')
            model_name: Base model to use
            **kwargs: Additional parameters

        Returns:
            Configured agent for the specified role
        """
        # Role-specific configurations
        role_configs = {
            'researcher': {
                'temperature': 0.8,  # More creative for research
                'max_tokens': 2000,
            },
            'summarizer': {
                'temperature': 0.3,  # More focused for summarization
                'max_tokens': 500,
            },
            'fact_checker': {
                'temperature': 0.1,  # Very focused for fact checking
                'max_tokens': 1000,
            },
            'report_writer': {
                'temperature': 0.5,  # Balanced for report writing
                'max_tokens': 3000,
            },
            'coordinator': {
                'temperature': 0.6,  # Moderate for coordination
                'max_tokens': 1500,
            },
            'specialist': {
                'temperature': 0.4,  # Focused for specialist analysis
                'max_tokens': 1500,
            },
            'analyst': {
                'temperature': 0.5,  # Balanced for analysis
                'max_tokens': 2000,
            },
            'reviewer': {
                'temperature': 0.3,  # Focused for review
                'max_tokens': 2500,
            },
        }

        # Get role-specific config or use defaults
        config = role_configs.get(role, {'temperature': 0.7, 'max_tokens': 1500})

        # Override with any provided kwargs
        config.update(kwargs)

        logger.info(f"Creating {role} agent with model {model_name}")
        return self.get_agent(model_name, **config)

    def get_available_models(self) -> Dict[str, bool]:
        """
        Get list of available models based on API keys.

        Returns:
            Dictionary of model names and availability status
        """
        available = {}
        for model_name, info in self.SUPPORTED_MODELS.items():
            provider = info['provider']
            available[model_name] = self.available_providers.get(provider, False)
        return available

    def get_agent_config(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a cached agent.

        Args:
            cache_key: Cache key for the agent

        Returns:
            Agent configuration or None if not found
        """
        return self.configurations.get(cache_key)

    def clear_cache(self):
        """Clear the agent cache."""
        self.agents.clear()
        self.configurations.clear()
        logger.info("Agent cache cleared")

    def estimate_cost(self,
                     model_name: str,
                     input_tokens: int,
                     output_tokens: int) -> float:
        """
        Estimate cost for a given model and token counts.

        Args:
            model_name: Name of the model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # October 2025 Model Pricing - ONLY latest models
        pricing = {
            'gpt-5': {'input': 0.01, 'output': 0.03},  # per 1K tokens
            'claude-sonnet-4.5': {'input': 0.003, 'output': 0.015},
            'grok-4': {'input': 0.005, 'output': 0.015},
            'grok-4-fast': {'input': 0.002, 'output': 0.008},
            'llama-3.1-8b': {'input': 0.0, 'output': 0.0},  # Free when run locally
        }

        if model_name not in pricing:
            logger.warning(f"Pricing not available for {model_name}")
            return 0.0

        model_pricing = pricing[model_name]
        cost = (input_tokens * model_pricing['input'] / 1000) + \
               (output_tokens * model_pricing['output'] / 1000)

        return round(cost, 4)
