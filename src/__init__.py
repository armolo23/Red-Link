"""
AgentRedChain - Sparse evaluation framework for red-teaming multi-agent LLM systems.
"""

__version__ = "0.1.0"
__author__ = "AgentRedChain Team"

from .agents.chain_builder import AgentChain
from .agents.agent_pool import AgentPool
from .attacks.injections import InjectionGenerator

__all__ = [
    'AgentChain',
    'AgentPool',
    'InjectionGenerator'
]