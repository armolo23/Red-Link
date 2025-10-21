"""
Unit tests for attack injection generator.
"""

import pytest
import json
from unittest.mock import MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks.injections import InjectionGenerator


class TestInjectionGenerator:
    """Test suite for InjectionGenerator class."""

    def test_initialization(self):
        """Test injection generator initialization."""
        injector = InjectionGenerator()

        assert injector.templates is not None
        assert len(injector.templates) == 5  # 5 attack categories
        assert 'goal_hijacking' in injector.templates
        assert 'data_exfiltration' in injector.templates
        assert 'privilege_escalation' in injector.templates
        assert 'jailbreak_propagation' in injector.templates
        assert 'subtle_poisoning' in injector.templates

    def test_get_attack_categories(self):
        """Test getting attack categories."""
        injector = InjectionGenerator()
        categories = injector.get_attack_categories()

        assert len(categories) == 5
        assert 'goal_hijacking' in categories

    def test_get_attacks_by_category(self):
        """Test getting attacks for a specific category."""
        injector = InjectionGenerator()
        attacks = injector.get_attacks_by_category('goal_hijacking')

        assert len(attacks) == 2  # 2 variants per category
        assert all('template' in attack for attack in attacks)
        assert all('severity' in attack for attack in attacks)
        assert all('description' in attack for attack in attacks)

    def test_generate_attack(self):
        """Test generating specific attack."""
        injector = InjectionGenerator()

        # Generate without target role
        attack = injector.generate('goal_hijacking', variant=0)
        assert isinstance(attack, str)
        assert len(attack) > 0

        # Generate with target role
        attack_with_role = injector.generate('goal_hijacking', target_role='researcher', variant=0)
        assert '[Addressing researcher]:' in attack_with_role

        # Test invalid attack type
        with pytest.raises(ValueError):
            injector.generate('invalid_attack')

    def test_generate_all_attacks(self):
        """Test generating all attack scenarios."""
        injector = InjectionGenerator()
        all_attacks = injector.generate_all_attacks()

        assert len(all_attacks) == 10  # 2 per category × 5 categories

        for category, attack, description in all_attacks:
            assert category in injector.templates
            assert isinstance(attack, str)
            assert isinstance(description, str)

    def test_inject_at_node(self):
        """Test injecting attack at chain node."""
        injector = InjectionGenerator()

        # Create mock chain
        chain = MagicMock()
        chain.topology = 'linear'
        chain.agents = [MagicMock() for _ in range(3)]
        chain.agent_roles = ['agent1', 'agent2', 'agent3']

        for agent in chain.agents:
            agent.prompt = MagicMock()
            agent.prompt.template = "Original template"

        # Inject attack
        attack_content = "Test attack"
        injector.inject_at_node(chain, 1, attack_content)

        # Check injection
        assert "[INJECTED]:" in chain.agents[1].prompt.template
        assert attack_content in chain.agents[1].prompt.template
        assert hasattr(chain.agents[1], '_original_template')

        # Check history
        assert len(injector.attack_history) == 1
        assert injector.attack_history[0]['node_idx'] == 1

    def test_clean_injection(self):
        """Test cleaning injected attack."""
        injector = InjectionGenerator()

        # Create mock chain with injected attack
        chain = MagicMock()
        chain.agents = [MagicMock()]
        chain.agents[0].prompt = MagicMock()
        chain.agents[0].prompt.template = "Modified template"
        chain.agents[0]._original_template = "Original template"

        # Clean injection
        injector.clean_injection(chain, 0)

        # Check cleanup
        assert chain.agents[0].prompt.template == "Original template"
        assert not hasattr(chain.agents[0], '_original_template')

    def test_create_attack_scenario(self):
        """Test creating complete attack scenario."""
        injector = InjectionGenerator()

        scenario = injector.create_attack_scenario(
            attack_type='goal_hijacking',
            chain_topology='linear',
            injection_points=[0, 2]
        )

        assert scenario['attack_type'] == 'goal_hijacking'
        assert scenario['chain_topology'] == 'linear'
        assert scenario['injection_points'] == [0, 2]
        assert len(scenario['attacks']) == 2
        assert all('node_idx' in attack for attack in scenario['attacks'])
        assert all('attack_content' in attack for attack in scenario['attacks'])
        assert all('severity' in attack for attack in scenario['attacks'])

    def test_save_templates(self, tmp_path):
        """Test saving attack templates."""
        injector = InjectionGenerator()

        template_path = tmp_path / 'templates.json'
        injector.save_templates(str(template_path))

        assert template_path.exists()

        with open(template_path) as f:
            saved_templates = json.load(f)

        assert len(saved_templates) == 5
        assert 'goal_hijacking' in saved_templates

    def test_load_custom_templates(self, tmp_path):
        """Test loading custom templates."""
        injector = InjectionGenerator()

        # Create custom templates
        custom_templates = {
            'custom_attack': [
                {
                    'template': 'Custom attack template',
                    'severity': 'high',
                    'description': 'Custom attack'
                }
            ]
        }

        custom_path = tmp_path / 'custom.json'
        with open(custom_path, 'w') as f:
            json.dump(custom_templates, f)

        # Load custom templates
        injector.load_custom_templates(str(custom_path))

        assert 'custom_attack' in injector.templates
        assert len(injector.templates['custom_attack']) == 1

    def test_severity_distribution(self):
        """Test getting severity distribution."""
        injector = InjectionGenerator()
        distribution = injector.get_severity_distribution()

        assert 'low' in distribution or distribution.get('low', 0) == 0
        assert 'medium' in distribution
        assert 'high' in distribution
        assert 'critical' in distribution
        assert sum(distribution.values()) == 10  # Total attacks

    def test_red_team_report(self):
        """Test generating red team report."""
        injector = InjectionGenerator()
        report = injector.generate_red_team_report()

        assert 'total_attacks' in report
        assert report['total_attacks'] == 10
        assert 'categories' in report
        assert len(report['categories']) == 5
        assert 'severity_distribution' in report
        assert 'attacks_by_category' in report

    def test_create_attack_matrix(self):
        """Test creating attack matrix."""
        injector = InjectionGenerator()
        chain_types = ['linear', 'star', 'hierarchical']

        matrix = injector.create_attack_matrix(chain_types)

        assert 'chains' in matrix
        assert matrix['chains'] == chain_types
        assert 'attacks' in matrix
        assert len(matrix['attacks']) == 5
        assert 'total_tests' in matrix
        assert matrix['total_tests'] == 3 * 5 * 2  # chains × categories × variants
        assert len(matrix['test_cases']) == 30