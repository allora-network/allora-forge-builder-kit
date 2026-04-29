# Allora Forge Model Creation Skills

## What These Are

These are Claude Code skills that guide you through building a financial prediction model from scratch. Each skill encodes a battle-tested development methodology — not a specific model or feature set, but the **design philosophy** behind models that actually generalize to live markets.

You will produce a complete, runnable prediction pipeline: data loading, feature engineering, model training, validation, and deployment monitoring. Working code, not documentation.

## Why They Exist

Allora's decentralized intelligence network gets stronger when participants submit models that are both **performant** and **diverse**. A network of clones — even good clones — is weaker than a network of diverse models that approach prediction from different angles.

These skills solve both problems:
- **Performance floor:** They encode the methodology behind models that survive out-of-sample validation, so you avoid the common pitfalls (lookahead contamination, overfitting, optimizer gaming) that kill live performance.
- **Diversity by design:** Three different skills guide you down fundamentally different reasoning paths. Your model's architecture, features, and loss function will reflect your unique starting point — not a template copied from someone else.

## The Three Skills

### Hypothesis-Driven (`forge-hypothesis-driven`)

*"I have a theory about what moves markets."*

**For:** Builders with a clear prediction target and a theory about what drives it. You believe specific signals (momentum, volatility regimes, microstructure, cross-asset relationships) predict your target, and you want a structured way to test that belief.

**Approach:** Deductive. You start with your hypothesis, translate it into testable features organized around estimation goals, and build a pipeline designed to falsify or confirm your theory.

**Best when:** You have domain knowledge you want to encode. You can articulate why your features should predict the target before seeing any results.

### Signal Discovery (`forge-signal-discovery`)

*"I have interesting data and want to find what's predictable."*

**For:** Builders with access to interesting data sources (alternative data, on-chain metrics, order flow, social signals) who want to discover what, if anything, is predictable in that data.

**Approach:** Inductive. You start with your data, rigorously assess what information it contains, and build prediction targets and features grounded in what the data can actually support.

**Best when:** Your edge is the data itself, not a pre-existing theory. You want to explore systematically rather than guess.

### Robustness-First (`forge-robustness-first`)

*"I want maximum confidence my model generalizes."*

**For:** Builders who have been burned by overfitting, or who simply want the highest confidence that their model is real before deploying capital.

**Approach:** Adversarial. You start by defining your validation framework and quality gates, then work backwards — designing features, architecture, and loss function specifically to survive the tests you already set up.

**Best when:** You prioritize generalization over raw backtest performance. You are willing to build a simpler model if it means higher confidence that it works in production.

## Prerequisites

- **Python 3.9+**
- **ML libraries:** scikit-learn, XGBoost or LightGBM (or your preferred gradient boosting / modeling library)
- **Data access:** Market data via a public API (exchange APIs, data aggregators) or your own data sources
- **Development environment:** Claude Code or Codex with skill support

No specific frameworks or proprietary dependencies required. The skills are agnostic to your data source and modeling library.

## How to Invoke

These are Claude Code skills. Invoke them by name:

```
forge-hypothesis-driven
forge-signal-discovery
forge-robustness-first
```

The skill will guide you through an interactive workflow — asking questions, prompting decisions, and producing code artifacts at each stage. You drive the choices; the skill ensures the methodology is sound.

## What Gets Produced

Each skill produces a complete set of pipeline artifacts:

| Artifact | Description |
|----------|-------------|
| `config.yaml` | Full pipeline configuration: target, horizon, data source, features, model, loss, validation gates |
| `data_loader.py` | Data fetching, caching, and quality validation |
| `feature_engineer.py` | Feature computation with horizon-adaptive parameters and lookahead prevention |
| `model.py` | Model definition, loss function, and training loop |
| `validation.py` | Quality gates as executable pass/fail checks |
| `evaluate.py` | Pipeline orchestrator running three-stage validation |
| `evaluation_report.md` | Full results: per-stage performance, gate outcomes, diagnostics |
| `monitor.py` | Post-deployment monitoring for performance, feature drift, and staleness |

Everything is config-driven and reproducible. Someone else should be able to re-run your pipeline from the config and get the same results.

## Shared Methodology

All three skills build on the same nine principles, documented in `shared/methodology.md`. The principles cover:

1. Physics-first feature engineering
2. Horizon-adaptive parameterization
3. Lookahead prevention by architecture
4. Three-stage separation (optimize, evaluate, deploy)
5. Loss function as a modeling decision
6. Multi-threshold signal validation
7. Optimizer-gaming prevention
8. Configuration-driven experimentation
9. Problem decomposition

The skills differ in entry point, ordering, and emphasis — not in rigor. Every pipeline produced by any skill satisfies all nine principles.
