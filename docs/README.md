# Lottery Prediction Methods Documentation

A comprehensive guide to lottery number prediction methods applied to Romanian lottery games (Joker, Loto 6/49, Loto 5/40) based on scientific research, patents, and statistical analysis.

## Important Disclaimer

**No mathematical method, machine learning model, or statistical analysis can predict truly random lottery numbers with accuracy better than chance.** The methods documented here serve to:

1. Make selections more structured and interesting
2. Ensure better coverage through wheeling systems
3. Provide educational insight into probability and statistics

Expected return on lottery tickets is always negative. Play responsibly and for entertainment only.

## Documentation Index

### Core Documentation

| Document | Description |
|----------|-------------|
| [01. Introduction](01-introduction.md) | Lottery basics, odds, expected value, and responsible gambling |
| [02. Statistical Methods](02-statistical-methods.md) | Classical statistical approaches: frequency, delta, hot/cold |
| [03. Machine Learning Methods](03-machine-learning-methods.md) | Neural networks, LSTM, ensemble methods |
| [04. Wheeling Systems](04-wheeling-systems.md) | Combinatorial coverage and guarantee systems |
| [05. Loto.ro Specific](05-loto-ro-specific.md) | Romanian lottery rules, odds, and historical analysis |
| [06. Implementation Guide](06-implementation-guide.md) | Using this codebase effectively |
| [07. Backtesting Results](07-backtesting-results.md) | Strategy performance comparison |
| [08. Honest Assessment](08-honest-assessment.md) | Limitations and realistic expectations |

### References

| Document | Description |
|----------|-------------|
| [Academic Papers](references/academic-papers.md) | Research summaries and citations |
| [Patents](references/patents.md) | Patent analysis and methods |
| [External Tools](references/external-tools.md) | Third-party resources and implementations |

## Quick Navigation by Topic

### "Which strategy should I use?"
Start with [08. Honest Assessment](08-honest-assessment.md) to set expectations, then see [06. Implementation Guide](06-implementation-guide.md) for practical recommendations.

### "How do the odds work?"
See [01. Introduction](01-introduction.md) for odds calculations and [05. Loto.ro Specific](05-loto-ro-specific.md) for Romanian lottery specifics.

### "What methods exist for prediction?"
- Classical statistics: [02. Statistical Methods](02-statistical-methods.md)
- Machine learning: [03. Machine Learning Methods](03-machine-learning-methods.md)
- Coverage optimization: [04. Wheeling Systems](04-wheeling-systems.md)

### "What does the research say?"
See [references/academic-papers.md](references/academic-papers.md) for academic research summaries.

## Methods Summary

| Method | Category | Mathematical Basis | Practical Value |
|--------|----------|-------------------|-----------------|
| Frequency Analysis | Statistical | Historical distribution | Moderate |
| Hot/Cold Numbers | Statistical | Exponential decay weighting | Moderate |
| Delta Analysis | Statistical | Gap distribution | Moderate |
| Sum Constraints | Statistical | Normal distribution of sums | Moderate |
| Pair Correlation | Statistical | Co-occurrence patterns | Low |
| Skip/Gap Analysis | Statistical | Gambler's fallacy adjacent | Low |
| Softmax Regression | ML | Probability estimation | Very Low |
| MLP/LSTM | ML | Pattern recognition | Very Low |
| Ensemble Voting | ML | Model aggregation | Low |
| Wheeling Systems | Combinatorial | Covering designs | **High** |
| Chi-Square Testing | Validation | Randomness verification | Educational |

## Key Findings from Research

1. **Lotteries are random by design** - A 2025 Romanian 6/49 study found Chi-square p-value of 0.0766, confirming no significant deviation from randomness.

2. **ML cannot beat randomness** - LSTM models on Powerball/Mega Millions show predictions are "not guaranteed" because "any sign of patterns or suspicious data was not found."

3. **Markov chains don't apply** - "Not every process has the Markov Property, such as the Lottery - this week's winning numbers have no dependence to the previous week's winning numbers."

4. **Wheeling systems are the only proven approach** - They provide mathematical guarantees for smaller prizes while jackpot odds remain unchanged.

## Project Structure

```
docs/
├── README.md                    # This file
├── 01-introduction.md           # Lottery basics
├── 02-statistical-methods.md    # Statistical approaches
├── 03-machine-learning-methods.md
├── 04-wheeling-systems.md
├── 05-loto-ro-specific.md
├── 06-implementation-guide.md
├── 07-backtesting-results.md
├── 08-honest-assessment.md
└── references/
    ├── academic-papers.md
    ├── patents.md
    └── external-tools.md
```

## Contributing

This documentation is part of the `loteria-romana` project. See the [main README](../README.md) for contribution guidelines.
