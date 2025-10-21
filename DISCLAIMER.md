# IMPORTANT DISCLAIMER

## This is Experimental Research Code

**AgentRedChain** is an experimental framework for red-teaming multi-agent LLM systems.

### What This Code Does:
✅ Implements sparse evaluation strategies
✅ Integrates with real LLM APIs (Claude 4.5 Sonnet, etc.)
✅ Provides attack injection mechanisms
✅ Calculates TVD-MI scores using embeddings
✅ Implements Rasch IRT modeling

### What Has NOT Been Validated:
❌ Cost reduction effectiveness (67% claim is theoretical)
❌ Accuracy preservation of sparse sampling
❌ Rasch model convergence on real data
❌ Statistical validity of the sampling strategies
❌ Correlation maintenance between sparse and dense evaluations

### Key Points:
- All performance metrics in documentation are **theoretical targets**, not empirical results
- The framework has not been tested at scale with real attack scenarios
- Cost savings are estimated based on sampling theory, not measured results
- The Rasch model implementation has not been validated against ground truth

### For Researchers:
This codebase is intended as a starting point for research into efficient red-teaming methods. It provides infrastructure for experimentation but requires empirical validation before any claims can be made about its effectiveness.

### For Production Use:
**DO NOT** use this framework in production without:
1. Extensive empirical validation
2. Comparison against dense baseline evaluations
3. Statistical verification of sampling effectiveness
4. Security review of the attack mechanisms

## Use at Your Own Risk

This software is provided "as is" without warranty of any kind. The authors are not responsible for any damages or losses arising from its use.