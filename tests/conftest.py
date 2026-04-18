"""Test-time configuration.

Disable torch-backed strategies in shared.ensemble_blend during tests.
Rationale: the torch models (LSTM/TCN/Transformer/NF) are slow (~10 min per
walk-forward call on CI) and non-deterministic in ways that break
reproducibility tests. The rest of the suite doesn't depend on those
strategies running — frequency/bayesian/genetic etc. are enough to exercise
the blending logic.

Torch-specific tests (test_lstm_strategy, test_tcn_strategy, ...) exercise
the models directly and don't go through ensemble_blend, so they remain
unaffected.
"""
import shared.ensemble_blend as _eb

_eb._TORCH_AVAILABLE = False
