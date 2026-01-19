"""
Transformer-based Sequence Model for Lottery Draw Analysis.

This module implements a Transformer architecture for analyzing lottery
draw sequences. While we don't expect to "beat" randomness, this model
serves to:

1. Provide a rigorous ML baseline
2. Verify that no learnable patterns exist (validation of randomness)
3. Demonstrate proper ML methodology for lottery analysis

Key insight: If this sophisticated model cannot beat random selection,
it provides strong evidence that simpler methods won't either.
"""

import math
from typing import Optional
from dataclasses import dataclass

# Import PyTorch - will fail gracefully if not installed
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


@dataclass
class TransformerConfig:
    """Configuration for lottery transformer model."""
    pool_size: int = 49           # Numbers to choose from
    draw_size: int = 6            # Numbers per draw
    d_model: int = 128            # Embedding dimension
    n_heads: int = 8              # Number of attention heads
    n_layers: int = 4             # Number of transformer layers
    d_ff: int = 512               # Feed-forward dimension
    dropout: float = 0.1          # Dropout rate
    max_seq_len: int = 100        # Maximum sequence length
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100


if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for sequence position information."""

        def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # Add batch dimension

            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add positional encoding to input embeddings."""
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


    class DrawEmbedding(nn.Module):
        """
        Embed lottery draws as dense vectors.

        Each draw is represented as a multi-hot encoding, then projected
        to a dense representation.
        """

        def __init__(self, pool_size: int, d_model: int):
            super().__init__()
            self.pool_size = pool_size
            self.d_model = d_model

            # Project multi-hot to dense
            self.projection = nn.Linear(pool_size, d_model)

        def forward(self, draws: torch.Tensor) -> torch.Tensor:
            """
            Embed sequence of draws.

            Args:
                draws: Tensor of shape (batch, seq_len, pool_size) with multi-hot encoding

            Returns:
                Embeddings of shape (batch, seq_len, d_model)
            """
            return self.projection(draws.float())


    class LotteryTransformer(nn.Module):
        """
        Transformer model for lottery sequence analysis.

        Architecture:
        - Draw embedding layer (multi-hot -> dense)
        - Positional encoding
        - Stack of transformer encoder layers
        - Output projection to number probabilities

        The model predicts probability distribution over next draw numbers.
        """

        def __init__(self, config: TransformerConfig):
            super().__init__()
            self.config = config

            # Embedding layers
            self.draw_embedding = DrawEmbedding(config.pool_size, config.d_model)
            self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

            # Output projection - predict probability for each number
            self.output_proj = nn.Linear(config.d_model, config.pool_size)

            # Layer norm for stability
            self.norm = nn.LayerNorm(config.d_model)

        def forward(self, draws: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Forward pass predicting next draw probabilities.

            Args:
                draws: Multi-hot encoded draws (batch, seq_len, pool_size)
                mask: Optional attention mask

            Returns:
                Logits for each number (batch, pool_size)
            """
            # Embed draws
            x = self.draw_embedding(draws)

            # Add positional encoding
            x = self.pos_encoding(x)

            # Create causal mask if not provided
            if mask is None:
                seq_len = draws.size(1)
                mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                mask = mask.to(draws.device)

            # Transform
            x = self.transformer(x, mask=mask)

            # Normalize
            x = self.norm(x)

            # Use last position for prediction
            x = x[:, -1, :]

            # Project to number probabilities
            logits = self.output_proj(x)

            return logits

        def predict_probabilities(self, draws: torch.Tensor) -> torch.Tensor:
            """Get softmax probabilities for next draw."""
            logits = self.forward(draws)
            return F.softmax(logits, dim=-1)

        def generate_picks(self, draws: torch.Tensor, num_picks: int) -> list[int]:
            """
            Generate lottery picks based on model predictions.

            Uses top-k selection based on predicted probabilities.

            Args:
                draws: Historical draws (multi-hot encoded)
                num_picks: Number of picks to generate

            Returns:
                List of predicted numbers (1-indexed)
            """
            self.eval()
            with torch.no_grad():
                probs = self.predict_probabilities(draws)
                # Get top-k indices (0-indexed)
                _, top_k = torch.topk(probs[0], num_picks)
                # Convert to 1-indexed numbers
                picks = (top_k + 1).cpu().tolist()

            return sorted(picks)


    class LotteryDataset(Dataset):
        """Dataset for lottery draw sequences."""

        def __init__(self, draws: list[list[int]], pool_size: int, seq_len: int = 50):
            """
            Initialize dataset.

            Args:
                draws: List of draws (each draw is list of numbers)
                pool_size: Total numbers in pool
                seq_len: Sequence length for training
            """
            self.draws = draws
            self.pool_size = pool_size
            self.seq_len = seq_len

            # Precompute multi-hot encodings
            self.encoded = []
            for draw in draws:
                encoding = torch.zeros(pool_size)
                for num in draw:
                    encoding[num - 1] = 1.0  # Convert to 0-indexed
                self.encoded.append(encoding)

        def __len__(self) -> int:
            return max(0, len(self.draws) - self.seq_len)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Get training sample.

            Returns:
                (input_sequence, target) where target is the next draw encoding
            """
            # Input: sequence of draws
            input_seq = torch.stack(self.encoded[idx:idx + self.seq_len])

            # Target: next draw
            target = self.encoded[idx + self.seq_len]

            return input_seq, target


    class LotteryTransformerTrainer:
        """
        Trainer for lottery transformer model.

        Implements proper training with:
        - Train/validation split
        - Early stopping
        - Learning rate scheduling
        - Gradient clipping
        """

        def __init__(self, config: TransformerConfig, device: str = "auto"):
            self.config = config

            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.model = LotteryTransformer(config).to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=0.01
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs
            )

            self.train_losses = []
            self.val_losses = []

        def train(self, draws: list[list[int]],
                 val_split: float = 0.2,
                 early_stop_patience: int = 10) -> dict:
            """
            Train the model on historical draws.

            Args:
                draws: List of historical draws
                val_split: Validation split ratio
                early_stop_patience: Epochs without improvement before stopping

            Returns:
                Training history dictionary
            """
            # Split data
            split_idx = int(len(draws) * (1 - val_split))
            train_draws = draws[:split_idx]
            val_draws = draws[split_idx:]

            # Create datasets
            train_dataset = LotteryDataset(
                train_draws, self.config.pool_size, self.config.max_seq_len
            )
            val_dataset = LotteryDataset(
                val_draws, self.config.pool_size, self.config.max_seq_len
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )

            # Loss function - multi-label binary cross entropy
            criterion = nn.BCEWithLogitsLoss()

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.config.num_epochs):
                # Training
                self.model.train()
                train_loss = 0.0

                for inputs, targets in train_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)
                self.train_losses.append(train_loss)

                # Validation
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                self.val_losses.append(val_loss)

                # Learning rate scheduling
                self.scheduler.step()

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Load best model
            if hasattr(self, 'best_state'):
                self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_state.items()})

            return {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_val_loss": best_val_loss,
                "epochs_trained": len(self.train_losses)
            }

        def evaluate(self, draws: list[list[int]]) -> dict:
            """
            Evaluate model on test draws.

            Computes various metrics comparing model predictions
            to actual draws.

            Args:
                draws: Test draws

            Returns:
                Evaluation metrics dictionary
            """
            self.model.eval()
            dataset = LotteryDataset(draws, self.config.pool_size, self.config.max_seq_len)

            if len(dataset) == 0:
                return {"error": "Insufficient test data"}

            total_correct = 0
            total_picks = 0
            top_k_hits = []

            with torch.no_grad():
                for i in range(len(dataset)):
                    inputs, target = dataset[i]
                    inputs = inputs.unsqueeze(0).to(self.device)

                    probs = self.model.predict_probabilities(inputs)[0].cpu()

                    # Get actual numbers from target
                    actual = set((target.nonzero().squeeze() + 1).tolist())
                    if isinstance(actual, int):
                        actual = {actual}

                    # Get top-k predictions (k = draw size)
                    _, top_k = torch.topk(probs, self.config.draw_size)
                    predicted = set((top_k + 1).tolist())

                    # Count hits
                    hits = len(actual & predicted)
                    total_correct += hits
                    total_picks += len(actual)
                    top_k_hits.append(hits)

            # Calculate metrics
            accuracy = total_correct / total_picks if total_picks > 0 else 0
            avg_hits = sum(top_k_hits) / len(top_k_hits) if top_k_hits else 0

            # Expected hits under random selection
            expected_random = (self.config.draw_size * self.config.draw_size) / self.config.pool_size

            return {
                "total_draws_evaluated": len(top_k_hits),
                "accuracy": accuracy,
                "average_hits_per_draw": avg_hits,
                "expected_random_hits": expected_random,
                "improvement_over_random": (avg_hits - expected_random) / expected_random if expected_random > 0 else 0,
                "hit_distribution": {
                    i: top_k_hits.count(i) for i in range(self.config.draw_size + 1)
                }
            }

        def generate_predictions(self, draws: list[list[int]], num_lines: int = 5) -> list[list[int]]:
            """
            Generate lottery predictions using trained model.

            Args:
                draws: Historical draws for context
                num_lines: Number of prediction lines to generate

            Returns:
                List of predicted number sets
            """
            self.model.eval()

            # Encode recent draws
            encoded = []
            for draw in draws[-self.config.max_seq_len:]:
                encoding = torch.zeros(self.config.pool_size)
                for num in draw:
                    encoding[num - 1] = 1.0
                encoded.append(encoding)

            input_seq = torch.stack(encoded).unsqueeze(0).to(self.device)

            predictions = []

            with torch.no_grad():
                probs = self.model.predict_probabilities(input_seq)[0].cpu()

                for i in range(num_lines):
                    # Sample from probability distribution with temperature
                    temperature = 1.0 + i * 0.2  # Increase diversity for later lines

                    if temperature > 1.0:
                        adjusted_probs = F.softmax(torch.log(probs + 1e-10) / temperature, dim=0)
                    else:
                        adjusted_probs = probs

                    # Sample without replacement
                    picks = []
                    remaining_probs = adjusted_probs.clone()

                    for _ in range(self.config.draw_size):
                        # Sample one number
                        idx = torch.multinomial(remaining_probs, 1).item()
                        picks.append(idx + 1)  # Convert to 1-indexed
                        remaining_probs[idx] = 0  # Remove from future selection
                        remaining_probs = remaining_probs / remaining_probs.sum()  # Renormalize

                    predictions.append(sorted(picks))

            return predictions

        def save_model(self, filepath: str) -> None:
            """Save model state to file."""
            torch.save({
                "model_state": self.model.state_dict(),
                "config": self.config,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses
            }, filepath)

        def load_model(self, filepath: str) -> None:
            """Load model state from file."""
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.config = checkpoint["config"]
            self.train_losses = checkpoint.get("train_losses", [])
            self.val_losses = checkpoint.get("val_losses", [])


else:
    # Stub classes when PyTorch is not available

    class LotteryTransformer:
        """Stub class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for transformer models. "
                "Install with: pip install torch"
            )


    class LotteryTransformerTrainer:
        """Stub class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for transformer training. "
                "Install with: pip install torch"
            )


def create_default_config(game: str = "loto_649") -> TransformerConfig:
    """
    Create default configuration for a specific lottery game.

    Args:
        game: One of "loto_649", "loto_540", "joker"

    Returns:
        TransformerConfig for the game
    """
    configs = {
        "loto_649": TransformerConfig(
            pool_size=49,
            draw_size=6,
            d_model=128,
            n_heads=8,
            n_layers=4
        ),
        "loto_540": TransformerConfig(
            pool_size=40,
            draw_size=5,  # Player picks 5
            d_model=96,
            n_heads=8,
            n_layers=3
        ),
        "joker": TransformerConfig(
            pool_size=45,
            draw_size=5,
            d_model=128,
            n_heads=8,
            n_layers=4
        )
    }

    return configs.get(game, configs["loto_649"])


def quick_analysis(draws: list[list[int]], game: str = "loto_649") -> dict:
    """
    Quick transformer analysis of lottery draws.

    Trains a model and evaluates whether any patterns are learnable.

    Args:
        draws: Historical draws
        game: Game type

    Returns:
        Analysis results dictionary
    """
    if not TORCH_AVAILABLE:
        return {
            "error": "PyTorch not available",
            "message": "Install PyTorch to use transformer analysis: pip install torch"
        }

    config = create_default_config(game)
    config.num_epochs = 50  # Quick training

    trainer = LotteryTransformerTrainer(config)

    # Train
    print("Training transformer model...")
    train_results = trainer.train(draws, val_split=0.2)

    # Evaluate
    print("Evaluating model...")
    eval_results = trainer.evaluate(draws[-200:])

    # Generate predictions
    predictions = trainer.generate_predictions(draws, num_lines=5)

    return {
        "training": train_results,
        "evaluation": eval_results,
        "sample_predictions": predictions,
        "conclusion": _generate_conclusion(eval_results)
    }


def _generate_conclusion(eval_results: dict) -> str:
    """Generate conclusion based on evaluation results."""
    if "error" in eval_results:
        return "Unable to evaluate model"

    improvement = eval_results.get("improvement_over_random", 0)

    if improvement < 0.05:
        return (
            "Model shows no significant improvement over random selection. "
            "This confirms the lottery draws appear to be random. "
            "No exploitable patterns detected."
        )
    elif improvement < 0.15:
        return (
            "Model shows slight improvement over random. "
            "This may be due to training set contamination or statistical noise. "
            "Further validation recommended with strict train/test separation."
        )
    else:
        return (
            "Model shows notable improvement over random. "
            "This is unexpected and warrants investigation. "
            "Verify there are no data leakage issues. "
            "If confirmed, this could indicate non-random patterns."
        )
