"""Shared utilities for lottery prediction strategies."""

from .math_utils import softmax, cross_entropy, relu, one_hot, matrix_multiply
from .config import JOKER_CONFIG, LOTO_649_CONFIG, GameConfig, NeuralConfig
from .strategy_base import Strategy, BaseStrategy, StrategyResult
from .neural_base import Layer, MLPModel, LSTMCell, LSTMModel, NeuralStrategyConfig
from .stats import (
    DeltaStrategy,
    HotColdStrategy,
    PairStrategy,
    SkipGapStrategy,
    SumConstraintStrategy,
    BalanceStrategy,
    compute_deltas,
    build_delta_distribution,
    compute_odd_even_distribution,
    compute_high_low_distribution,
    count_consecutives,
    compute_consecutive_distribution,
    build_position_frequency,
)
from .backtest_base import (
    PrizeTier,
    BacktestResult,
    Backtester,
    CrossValidator,
    wilson_score_interval,
    compute_max_drawdown,
)
from .ensemble import (
    EnsembleVoter,
    StrategySelector,
)
from .wheeling import (
    WheelConfig,
    WheelGenerator,
    KeyNumberWheel,
    generate_full_wheel,
    generate_balanced_wheel,
    verify_wheel_coverage,
    estimate_full_wheel_size,
)

__all__ = [
    # Math utilities
    "softmax",
    "cross_entropy",
    "relu",
    "one_hot",
    "matrix_multiply",
    # Config
    "JOKER_CONFIG",
    "LOTO_649_CONFIG",
    "GameConfig",
    "NeuralConfig",
    # Strategy
    "Strategy",
    "BaseStrategy",
    "StrategyResult",
    # Neural
    "Layer",
    "MLPModel",
    "LSTMCell",
    "LSTMModel",
    "NeuralStrategyConfig",
    # Statistical Strategies
    "DeltaStrategy",
    "HotColdStrategy",
    "PairStrategy",
    "SkipGapStrategy",
    "SumConstraintStrategy",
    "BalanceStrategy",
    # Statistical Utilities
    "compute_deltas",
    "build_delta_distribution",
    "compute_odd_even_distribution",
    "compute_high_low_distribution",
    "count_consecutives",
    "compute_consecutive_distribution",
    "build_position_frequency",
    # Backtesting
    "PrizeTier",
    "BacktestResult",
    "Backtester",
    "CrossValidator",
    "wilson_score_interval",
    "compute_max_drawdown",
    # Ensemble
    "EnsembleVoter",
    "StrategySelector",
    # Wheeling
    "WheelConfig",
    "WheelGenerator",
    "KeyNumberWheel",
    "generate_full_wheel",
    "generate_balanced_wheel",
    "verify_wheel_coverage",
    "estimate_full_wheel_size",
]
