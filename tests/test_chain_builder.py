"""
Unit tests for AgentChain builder.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.chain_builder import AgentChain


class TestAgentChain:
    """Test suite for AgentChain class."""

    def test_chain_initialization(self):
        """Test chain initialization with different topologies."""
        # Test valid topologies
        for topology in ['linear', 'star', 'hierarchical']:
            chain = AgentChain(topology=topology)
            assert chain.topology == topology
            assert chain.agents == []
            assert chain.edges == []

        # Test invalid topology
        with pytest.raises(ValueError):
            chain = AgentChain(topology='invalid')

    @patch('src.agents.chain_builder.ChatOpenAI')
    def test_add_agent(self, mock_openai):
        """Test adding agents to chain."""
        chain = AgentChain(topology='linear')

        # Add agent
        idx = chain.add_agent('test_role')

        assert idx == 0
        assert len(chain.agents) == 1
        assert chain.agent_roles[0] == 'test_role'

    @patch('src.agents.chain_builder.ChatOpenAI')
    def test_build_research_pipeline(self, mock_openai):
        """Test building linear research pipeline."""
        chain = AgentChain(topology='linear')
        result = chain.build_research_pipeline()

        assert result == chain  # Should return self
        assert len(chain.agents) == 4
        assert chain.agent_roles == [
            'web researcher',
            'information summarizer',
            'fact checker',
            'report writer'
        ]
        assert len(chain.edges) == 3

    @patch('src.agents.chain_builder.ChatOpenAI')
    def test_build_consensus_system(self, mock_openai):
        """Test building star consensus system."""
        chain = AgentChain(topology='star')
        result = chain.build_consensus_system()

        assert result == chain
        assert len(chain.agents) == 4  # 1 coordinator + 3 specialists
        assert chain.agent_roles[0] == 'consensus coordinator'

    @patch('src.agents.chain_builder.ChatOpenAI')
    def test_build_hierarchical_review(self, mock_openai):
        """Test building hierarchical review system."""
        chain = AgentChain(topology='hierarchical')
        result = chain.build_hierarchical_review()

        assert result == chain
        assert len(chain.agents) == 3  # 2 juniors + 1 senior
        assert 'senior reviewer' in chain.agent_roles

    def test_save_config(self, tmp_path):
        """Test saving chain configuration."""
        chain = AgentChain(topology='linear')
        chain.agent_roles = ['role1', 'role2']
        chain.edges = [(0, 1, 'data')]

        config_path = tmp_path / 'config.json'
        chain.save_config(str(config_path))

        assert config_path.exists()

        import json
        with open(config_path) as f:
            config = json.load(f)

        assert config['topology'] == 'linear'
        assert config['num_agents'] == 0  # No actual agents created
        assert config['agent_roles'] == ['role1', 'role2']

    @patch('src.agents.chain_builder.ChatOpenAI')
    def test_load_config(self, mock_openai, tmp_path):
        """Test loading chain configuration."""
        # Create and save a config
        chain1 = AgentChain(topology='linear')
        config_path = tmp_path / 'config.json'

        import json
        config = {
            'topology': 'linear',
            'model_type': 'gpt-5',
            'num_agents': 4,
            'agent_roles': ['role1', 'role2', 'role3', 'role4'],
            'edges': [(0, 1, 'data'), (1, 2, 'data'), (2, 3, 'data')]
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

        # Load config
        chain2 = AgentChain.load_config(str(config_path))

        assert chain2.topology == 'linear'
        assert chain2.model_type == 'gpt-5'

    @patch('src.agents.chain_builder.ChatOpenAI')
    def test_execute_with_injection(self, mock_openai):
        """Test chain execution with attack injection."""
        chain = AgentChain(topology='linear')
        chain.agents = [MagicMock() for _ in range(3)]
        chain.agent_roles = ['agent1', 'agent2', 'agent3']

        # Mock agent with prompt
        for agent in chain.agents:
            agent.prompt = MagicMock()
            agent.prompt.template = "Original template"

        # Inject attack
        chain._inject_attack(1, "Attack content")

        # Check injection
        assert "Attack content" in chain.agents[1].prompt.template
        assert hasattr(chain.agents[1], '_original_template')

        # Clean injection
        chain._clean_injection(1)

        # Check cleanup
        assert chain.agents[1].prompt.template == "Original template"
        assert not hasattr(chain.agents[1], '_original_template')

    def test_injection_bounds_checking(self):
        """Test injection with invalid agent index."""
        chain = AgentChain(topology='linear')
        chain.agents = [MagicMock() for _ in range(2)]

        # Should not raise error for invalid index
        chain._inject_attack(5, "Attack")  # Out of bounds
        chain._inject_attack(-1, "Attack")  # Negative index