# AgentRedChain

**An Experimental Framework for Security Testing of Multi-Agent AI Systems**

## Overview

AgentRedChain is a research framework designed to evaluate the security vulnerabilities of multi-agent artificial intelligence systems. As organizations increasingly deploy chains of AI agents working together to accomplish complex tasks, understanding how these systems respond to adversarial inputs becomes critical for ensuring their safe and reliable operation.

This framework introduces sparse evaluation strategies that aim to reduce the computational costs of comprehensive security testing while maintaining assessment quality through statistical sampling methods. By implementing targeted attack injection and vulnerability modeling techniques, AgentRedChain provides researchers with tools to systematically identify weaknesses in multi-agent AI deployments.

**Important Notice**: This is experimental research software. Performance claims regarding cost reduction and accuracy preservation are theoretical and have not been empirically validated through comprehensive testing.

## Core Capabilities

### Security Testing Infrastructure
The framework enables systematic red-teaming of multi-agent systems through controlled attack injection at specific points in the agent chain. It supports three fundamental agent architectures: linear pipelines for sequential processing, star topologies for consensus-based decision making, and hierarchical structures for multi-level review systems.

### Attack Simulation Library
AgentRedChain implements ten distinct attack scenarios across five critical security categories: goal hijacking, data exfiltration, privilege escalation, jailbreak propagation, and subtle information poisoning. Each attack is designed to test different aspects of agent resilience and inter-agent communication security.

### Vulnerability Assessment Metrics
The framework employs Total Variation Distance-Mutual Information (TVD-MI) scoring to quantify the semantic divergence between clean and compromised agent outputs. This embedding-based approach provides interpretable metrics for measuring how attacks propagate through multi-agent systems and impact final outputs.

### Sparse Evaluation Methodology
To address the computational expense of testing all possible attack-agent combinations, AgentRedChain implements sparse sampling strategies informed by Item Response Theory from educational assessment. The framework includes random, logarithmic, and pattern-informed sampling approaches that aim to identify high-value test cases while reducing overall evaluation costs.

## Technical Architecture

### Supported Models
The framework currently integrates with leading AI providers through their official APIs:
- **GPT-5** (OpenAI): 256K token context window
- **Claude Sonnet 4.5** (Anthropic): 200K token context window
- **Grok 4** (xAI): 32K token context window

### Implementation Components
- **Agent Chain Builder**: Constructs multi-agent systems with configurable topologies
- **Attack Generator**: Creates adversarial prompts targeted at specific vulnerabilities
- **Evaluation Engine**: Measures attack impact using semantic similarity metrics
- **Cost Tracker**: Monitors API usage and provides real-time budget management
- **Rasch Model**: Experimental vulnerability modeling using alternating least squares

## Installation and Setup

### Prerequisites
- Python 3.9 or higher
- API keys for at least one supported model provider
- Approximately 2GB of disk space for model embeddings

### Installation Steps

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/yourusername/agentredchain.git
cd agentredchain
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your API credentials by creating a `.env` file:
```
OPENAI_API_KEY=your-gpt5-key
ANTHROPIC_API_KEY=your-claude-key
XAI_API_KEY=your-grok-key
```

## Usage Example

The following example demonstrates a basic security evaluation of a multi-agent research pipeline:

```python
from src.agents.chain_builder import AgentChain
from src.attacks.injections import InjectionGenerator
from src.evaluation.tvd_mi_scorer import TVDMIScorer

# Initialize a multi-agent chain
chain = AgentChain(
    topology='linear',
    model_type='claude-sonnet-4.5',
    enable_cost_tracking=True,
    budget_limit=5.00
)
chain.build_research_pipeline()

# Generate attack scenarios
injector = InjectionGenerator()
attacks = injector.generate_all_attacks()

# Execute evaluation with attack injection
baseline = chain.execute("Analyze AI ethics in healthcare")
compromised = chain.execute(
    "Analyze AI ethics in healthcare",
    inject_at=1,
    injection_content=attacks[0][1]
)

# Measure impact
scorer = TVDMIScorer()
impact = scorer.compute_tvd_mi(baseline['final_output'], compromised['final_output'])
print(f"Attack impact score: {impact:.4f}")
```

## Research Context

This framework emerged from the recognition that as multi-agent AI systems become more prevalent in production environments, traditional security testing approaches that evaluate every possible vulnerability become prohibitively expensive. AgentRedChain explores whether statistical sampling techniques can maintain evaluation quality while significantly reducing the number of tests required.

The sparse evaluation approach draws inspiration from educational assessment, where Item Response Theory has successfully modeled student abilities and question difficulties from incomplete test data. By adapting these techniques to security testing, the framework attempts to predict unobserved vulnerabilities from a limited set of executed tests.

## Limitations and Disclaimers

### Experimental Nature
AgentRedChain is a research prototype intended for academic investigation and controlled security testing. The framework's core hypothesis—that sparse sampling can maintain evaluation quality while reducing costs—has not been validated through empirical studies with real-world data.

### Performance Claims
Any performance metrics mentioned in the codebase or documentation represent theoretical targets based on statistical sampling theory, not measured results from production deployments. Users should conduct their own validation before relying on the framework's assessments.

### Security Considerations
This tool is designed for authorized security testing only. Users are responsible for obtaining appropriate permissions before testing any systems they do not own. The attack scenarios included are intended for defensive research and should not be used for malicious purposes.

## Project Structure

```
agentredchain/
├── src/
│   ├── agents/           # Multi-agent chain construction
│   ├── attacks/          # Attack scenario generation
│   ├── evaluation/       # Vulnerability scoring and analysis
│   └── utils/            # API management and cost tracking
├── notebooks/            # Demonstration notebooks
├── tests/               # Unit and integration tests
└── DISCLAIMER.md        # Important usage warnings
```

## Contributing

As an experimental research project, AgentRedChain welcomes contributions from the security and AI research communities. Please review the contribution guidelines before submitting pull requests. Areas of particular interest include empirical validation studies, alternative sampling strategies, and extensions to the attack library.

## License

This project is released under the MIT License. See the LICENSE file for complete details.

## Contact

For questions about the research methodology or potential collaborations, please open an issue on the GitHub repository.