"""Game configurations for lottery prediction."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """Configuration for a lottery game."""

    name: str
    number_pool: int  # Total numbers to choose from (e.g., 45 for Joker)
    numbers_to_pick: int  # How many numbers per draw (e.g., 5 for Joker)
    secondary_pool: int  # Secondary number pool (e.g., 20 for Joker bonus)
    secondary_count: int  # How many secondary numbers (usually 1)
    secondary_name: str  # Name for the secondary number (e.g., "Joker", "Noroc")

    @property
    def input_size(self) -> int:
        """Size of one-hot encoded input for neural networks."""
        return self.number_pool + self.secondary_pool

    @property
    def main_output_size(self) -> int:
        """Size of output for main number prediction."""
        return self.number_pool

    @property
    def secondary_output_size(self) -> int:
        """Size of output for secondary number prediction."""
        return self.secondary_pool


JOKER_CONFIG = GameConfig(
    name="Joker",
    number_pool=45,
    numbers_to_pick=5,
    secondary_pool=20,
    secondary_count=1,
    secondary_name="Joker",
)

LOTO_649_CONFIG = GameConfig(
    name="Loto 6/49",
    number_pool=49,
    numbers_to_pick=6,
    secondary_pool=10_000_000,  # Noroc is 0-9,999,999
    secondary_count=1,
    secondary_name="Noroc",
)


@dataclass
class NeuralConfig:
    """Configuration for neural network training."""

    architecture: str = "softmax"  # 'softmax', 'mlp', 'lstm'
    hidden_sizes: tuple[int, ...] = (64, 32)
    learning_rate: float = 0.1
    epochs: int = 100
    l2_lambda: float = 0.001
    sequence_length: int = 10  # For LSTM: how many past draws to use
    dropout_rate: float = 0.0  # For regularization (simulated)

    def __post_init__(self):
        if isinstance(self.hidden_sizes, list):
            self.hidden_sizes = tuple(self.hidden_sizes)


DEFAULT_NEURAL_CONFIG = NeuralConfig()


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    min_train_size: int = 50  # Minimum draws before making predictions
    n_folds: int = 5  # For cross-validation
    recent_window: int = 50  # Recent draws for performance tracking


DEFAULT_BACKTEST_CONFIG = BacktestConfig()
