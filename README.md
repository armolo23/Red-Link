# AgentRedChain

**Status: Early development. Core hypothesis unvalidated.**

An experimental framework for testing whether sparse evaluation can reduce the cost of security testing multi-agent AI systems.

## What this is

AgentRedChain is a tool I'm building to explore whether statistical sampling methods from Item Response Theory can identify vulnerabilities in multi-agent systems without exhaustive testing. The framework injects adversarial prompts at specific points in agent chains and uses Rasch modeling to attempt predicting behavior of untested combinations.

**Important: This is experimental software under active development. The sparse evaluation approach has not been validated. Performance claims are theoretical targets, not measured results.**

## Core components

### Agent architectures
Three topologies are supported. Linear pipelines for sequential processing. Star topologies for consensus-based decisions. Hierarchical structures for multi-level review.

### Attack library
Ten attack scenarios across five categories. Goal hijacking redirects objectives. Data exfiltration tests information leakage. Privilege escalation probes unauthorized access. Jailbreak propagation tracks adversarial spread. Information poisoning introduces compounding corruptions.

### Evaluation metrics
TVD-MI scoring quantifies semantic divergence between clean and compromised outputs. This embedding-based approach measures attack propagation through the system.

### Sparse sampling
The framework implements random, logarithmic, and pattern-informed sampling strategies informed by Item Response Theory. The hypothesis is that not all attack-agent combinations provide equal information. This has not been tested in practice.

## Technical details

### Supported models
Integration with three AI providers through their APIs:
- GPT-5 (OpenAI): 256K token context
- Claude Sonnet 4.5 (Anthropic): 200K token context
- Grok 4 (xAI): 32K token context

### Implementation
- Agent Chain Builder: Constructs multi-agent systems with configurable topologies
- Attack Generator: Creates adversarial prompts from templates
- Evaluation Engine: Measures impact through semantic similarity
- Cost Tracker: Monitors API usage for budget management
- Rasch Model: Experimental vulnerability modeling (untested)

## Installation

Prerequisites:
- Python 3.9 or higher
- API keys for at least one supported model provider
- Approximately 2GB disk space for embeddings

```bash
git clone https://github.com/armolo23/Red-Link.git
cd Red-Link
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
XAI_API_KEY=your-key
```

## Usage example

```python
from src.agents.chain_builder import AgentChain
from src.attacks.injections import InjectionGenerator
from src.evaluation.tvd_mi_scorer import TVDMIScorer

# Initialize chain
chain = AgentChain(
    topology='linear',
    model_type='claude-sonnet-4.5',
    enable_cost_tracking=True,
    budget_limit=5.00
)
chain.build_research_pipeline()

# Generate attacks
injector = InjectionGenerator()
attacks = injector.generate_all_attacks()

# Execute with and without attack
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

## Research questions

I'm particularly interested in how attacks propagate through agent networks, how you test for adversarial robustness when the system itself is distributed, and how you assign liability when failures emerge from interactions rather than individual components. These are empirical questions that need systematic testing infrastructure.

The sparse evaluation hypothesis is that Item Response Theory techniques can transfer from educational assessment to adversarial security testing. Whether this actually works remains an open question requiring rigorous empirical validation.

## What this is not

This is not validated research. The framework's core claim that sparse sampling maintains evaluation quality while reducing costs is theoretical. No empirical studies have verified effectiveness against real-world data.

The approach requires advance knowledge of which behaviors to test. Novel vulnerabilities from agent interactions will be missed. Only predetermined injection points are evaluated. Dynamic adversarial adaptation is unexplored.

Performance metrics in the codebase are aspirational targets, not measured results. Statistical models may not generalize across architectures or attack types. IRT assumptions may not hold in adversarial contexts.

## Validation needed

Moving from experimental tool to validated approach requires empirical studies comparing sparse evaluation against exhaustive testing across diverse architectures. Both cost reduction and vulnerability detection accuracy must be measured. Whether IRT assumptions hold in security contexts is an open question.

Traditional red-teaming enumerates comprehensively. Sparse evaluation accepts incomplete testing for reduced cost, but only if statistical models accurately predict untested scenarios. This needs testing.

## Project structure

```
agentredchain/
├── src/
│   ├── agents/       # Multi-agent chain construction
│   ├── attacks/      # Attack scenario generation
│   ├── evaluation/   # Vulnerability scoring
│   └── utils/        # API management and cost tracking
├── notebooks/        # Demonstration notebooks
└── tests/           # Unit and integration tests
```

## Security notice

This tool is for authorized security testing only. Do not use for malicious purposes. Do not deploy for production assessment without rigorous validation. Users are responsible for obtaining appropriate permissions before testing systems they do not own.

## License

MIT License. See LICENSE file for details.

## Contact

For questions about the methodology or potential collaboration, open an issue on the repository.
