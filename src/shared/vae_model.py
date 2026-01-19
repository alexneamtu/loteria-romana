"""
Variational Autoencoder for Lottery Draw Anomaly Detection.

This module implements a VAE to learn the "normal" distribution of lottery
draws and detect anomalies. The key applications are:

1. Anomaly Detection: Identify draws that are statistically unusual
2. Distribution Learning: Understand the latent structure of draw patterns
3. Synthetic Generation: Generate realistic-looking draw distributions
4. Validation: Verify draws conform to expected random distributions

The VAE encodes draws into a latent space and reconstructs them. Draws with
high reconstruction error are potential anomalies worth investigating.
"""

from dataclasses import dataclass
from typing import Optional
import math

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
class VAEConfig:
    """Configuration for lottery VAE model."""
    pool_size: int = 49           # Numbers to choose from
    draw_size: int = 6            # Numbers per draw
    latent_dim: int = 16          # Latent space dimension
    hidden_dims: tuple = (64, 32) # Hidden layer dimensions
    dropout: float = 0.1          # Dropout rate
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 100
    beta: float = 1.0             # KL divergence weight (beta-VAE)


if TORCH_AVAILABLE:

    class LotteryVAE(nn.Module):
        """
        Variational Autoencoder for lottery draws.

        Architecture:
        - Encoder: Multi-hot input -> Hidden layers -> (mu, log_var)
        - Latent: Reparameterization trick for sampling
        - Decoder: Latent -> Hidden layers -> Reconstructed multi-hot

        The model learns to compress lottery draws into a low-dimensional
        latent space and reconstruct them, enabling anomaly detection.
        """

        def __init__(self, config: VAEConfig):
            super().__init__()
            self.config = config

            # Build encoder
            encoder_layers = []
            in_dim = config.pool_size
            for hidden_dim in config.hidden_dims:
                encoder_layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.BatchNorm1d(hidden_dim)
                ])
                in_dim = hidden_dim

            self.encoder = nn.Sequential(*encoder_layers)

            # Latent space projections
            self.fc_mu = nn.Linear(config.hidden_dims[-1], config.latent_dim)
            self.fc_var = nn.Linear(config.hidden_dims[-1], config.latent_dim)

            # Build decoder
            decoder_layers = []
            in_dim = config.latent_dim
            for hidden_dim in reversed(config.hidden_dims):
                decoder_layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.BatchNorm1d(hidden_dim)
                ])
                in_dim = hidden_dim

            self.decoder = nn.Sequential(*decoder_layers)
            self.output_layer = nn.Linear(config.hidden_dims[0], config.pool_size)

        def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Encode input to latent distribution parameters.

            Args:
                x: Multi-hot encoded draws (batch, pool_size)

            Returns:
                (mu, log_var) of latent distribution
            """
            h = self.encoder(x)
            mu = self.fc_mu(h)
            log_var = self.fc_var(h)
            return mu, log_var

        def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
            """
            Reparameterization trick for sampling from latent distribution.

            Args:
                mu: Mean of latent distribution
                log_var: Log variance of latent distribution

            Returns:
                Sampled latent vector
            """
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            """
            Decode latent vector to reconstruction.

            Args:
                z: Latent vector (batch, latent_dim)

            Returns:
                Reconstruction logits (batch, pool_size)
            """
            h = self.decoder(z)
            return self.output_layer(h)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Full forward pass.

            Args:
                x: Multi-hot encoded draws

            Returns:
                (reconstruction, mu, log_var)
            """
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            recon = self.decode(z)
            return recon, mu, log_var

        def loss_function(self, recon: torch.Tensor, x: torch.Tensor,
                         mu: torch.Tensor, log_var: torch.Tensor) -> tuple[torch.Tensor, dict]:
            """
            Compute VAE loss: reconstruction + KL divergence.

            Args:
                recon: Reconstructed output
                x: Original input
                mu: Latent mean
                log_var: Latent log variance

            Returns:
                (total_loss, loss_components_dict)
            """
            # Reconstruction loss (binary cross entropy)
            recon_loss = F.binary_cross_entropy_with_logits(
                recon, x, reduction='sum'
            ) / x.size(0)

            # KL divergence
            kl_loss = -0.5 * torch.sum(
                1 + log_var - mu.pow(2) - log_var.exp()
            ) / x.size(0)

            # Total loss with beta weighting
            total_loss = recon_loss + self.config.beta * kl_loss

            return total_loss, {
                'total': total_loss.item(),
                'reconstruction': recon_loss.item(),
                'kl_divergence': kl_loss.item()
            }

        def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
            """
            Compute per-sample reconstruction error for anomaly detection.

            Args:
                x: Multi-hot encoded draws (batch, pool_size)

            Returns:
                Reconstruction error per sample (batch,)
            """
            self.eval()
            with torch.no_grad():
                recon, _, _ = self.forward(x)
                # Per-sample BCE
                error = F.binary_cross_entropy_with_logits(
                    recon, x, reduction='none'
                ).sum(dim=1)
            return error

        def generate_samples(self, num_samples: int,
                            temperature: float = 1.0) -> torch.Tensor:
            """
            Generate synthetic lottery draws from latent space.

            Args:
                num_samples: Number of samples to generate
                temperature: Sampling temperature

            Returns:
                Generated multi-hot draws (num_samples, pool_size)
            """
            self.eval()
            with torch.no_grad():
                # Sample from standard normal
                z = torch.randn(num_samples, self.config.latent_dim)
                z = z * temperature
                z = z.to(next(self.parameters()).device)

                # Decode
                logits = self.decode(z)
                probs = torch.sigmoid(logits)

            return probs


    class DrawDataset(Dataset):
        """Dataset for lottery draws (VAE training)."""

        def __init__(self, draws: list[list[int]], pool_size: int):
            self.draws = draws
            self.pool_size = pool_size

            # Precompute multi-hot encodings
            self.encoded = []
            for draw in draws:
                encoding = torch.zeros(pool_size)
                for num in draw:
                    encoding[num - 1] = 1.0
                self.encoded.append(encoding)

        def __len__(self) -> int:
            return len(self.draws)

        def __getitem__(self, idx: int) -> torch.Tensor:
            return self.encoded[idx]


    class VAETrainer:
        """
        Trainer for lottery VAE model.

        Implements proper training with anomaly detection capabilities.
        """

        def __init__(self, config: VAEConfig, device: str = "auto"):
            self.config = config

            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.model = LotteryVAE(config).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate
            )

            self.train_losses = []
            self.anomaly_threshold = None

        def train(self, draws: list[list[int]],
                 val_split: float = 0.2) -> dict:
            """
            Train the VAE on historical draws.

            Args:
                draws: List of historical draws
                val_split: Validation split ratio

            Returns:
                Training history dictionary
            """
            # Split data
            split_idx = int(len(draws) * (1 - val_split))
            train_draws = draws[:split_idx]
            val_draws = draws[split_idx:]

            # Create datasets
            train_dataset = DrawDataset(train_draws, self.config.pool_size)
            val_dataset = DrawDataset(val_draws, self.config.pool_size)

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

            best_val_loss = float('inf')

            for epoch in range(self.config.num_epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                loss_components = {'reconstruction': 0, 'kl_divergence': 0}

                for batch in train_loader:
                    batch = batch.to(self.device)

                    self.optimizer.zero_grad()
                    recon, mu, log_var = self.model(batch)
                    loss, components = self.model.loss_function(recon, batch, mu, log_var)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += components['total']
                    loss_components['reconstruction'] += components['reconstruction']
                    loss_components['kl_divergence'] += components['kl_divergence']

                train_loss /= len(train_loader)
                for k in loss_components:
                    loss_components[k] /= len(train_loader)

                self.train_losses.append({
                    'total': train_loss,
                    **loss_components
                })

                # Validation
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        recon, mu, log_var = self.model(batch)
                        loss, _ = self.model.loss_function(recon, batch, mu, log_var)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_state = {k: v.cpu().clone()
                                       for k, v in self.model.state_dict().items()}

                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

            # Load best model
            if hasattr(self, 'best_state'):
                self.model.load_state_dict({k: v.to(self.device)
                                            for k, v in self.best_state.items()})

            # Compute anomaly threshold (95th percentile of training errors)
            self._compute_anomaly_threshold(train_draws)

            return {
                "train_losses": self.train_losses,
                "best_val_loss": best_val_loss,
                "epochs_trained": len(self.train_losses),
                "anomaly_threshold": self.anomaly_threshold
            }

        def _compute_anomaly_threshold(self, draws: list[list[int]],
                                      percentile: float = 95.0) -> None:
            """Compute anomaly detection threshold from training data."""
            dataset = DrawDataset(draws, self.config.pool_size)
            loader = DataLoader(dataset, batch_size=self.config.batch_size)

            all_errors = []

            self.model.eval()
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    errors = self.model.get_reconstruction_error(batch)
                    all_errors.extend(errors.cpu().tolist())

            # Set threshold at specified percentile
            all_errors.sort()
            idx = int(len(all_errors) * percentile / 100)
            self.anomaly_threshold = all_errors[min(idx, len(all_errors) - 1)]

        def detect_anomalies(self, draws: list[list[int]],
                            threshold: Optional[float] = None) -> list[dict]:
            """
            Detect anomalous draws.

            Args:
                draws: Draws to analyze
                threshold: Custom threshold (uses trained threshold if None)

            Returns:
                List of anomaly reports for each draw
            """
            if threshold is None:
                threshold = self.anomaly_threshold or float('inf')

            dataset = DrawDataset(draws, self.config.pool_size)
            loader = DataLoader(dataset, batch_size=self.config.batch_size)

            results = []
            idx = 0

            self.model.eval()
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    errors = self.model.get_reconstruction_error(batch)

                    for error in errors.cpu().tolist():
                        is_anomaly = error > threshold
                        results.append({
                            "draw_index": idx,
                            "draw": draws[idx],
                            "reconstruction_error": error,
                            "threshold": threshold,
                            "is_anomaly": is_anomaly,
                            "anomaly_score": error / threshold if threshold > 0 else 0
                        })
                        idx += 1

            return results

        def analyze_latent_space(self, draws: list[list[int]]) -> dict:
            """
            Analyze the latent space representation of draws.

            Args:
                draws: Draws to analyze

            Returns:
                Latent space analysis dictionary
            """
            dataset = DrawDataset(draws, self.config.pool_size)
            loader = DataLoader(dataset, batch_size=len(draws))

            self.model.eval()
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    mu, log_var = self.model.encode(batch)

                    mu_np = mu.cpu().numpy()
                    var_np = torch.exp(log_var).cpu().numpy()

            # Analyze latent dimensions
            dim_analysis = []
            for d in range(self.config.latent_dim):
                dim_analysis.append({
                    "dimension": d,
                    "mean": float(mu_np[:, d].mean()),
                    "std": float(mu_np[:, d].std()),
                    "variance_mean": float(var_np[:, d].mean())
                })

            # Sort by variance (most informative dimensions first)
            dim_analysis.sort(key=lambda x: -x["std"])

            return {
                "num_draws": len(draws),
                "latent_dim": self.config.latent_dim,
                "dimension_analysis": dim_analysis,
                "total_variance_explained": sum(d["std"]**2 for d in dim_analysis),
                "top_3_informative_dims": [d["dimension"] for d in dim_analysis[:3]]
            }

        def generate_synthetic_draws(self, num_draws: int,
                                    temperature: float = 1.0) -> list[list[int]]:
            """
            Generate synthetic lottery draws from the learned distribution.

            Args:
                num_draws: Number of draws to generate
                temperature: Sampling temperature (higher = more diverse)

            Returns:
                List of synthetic draws
            """
            self.model.eval()
            probs = self.model.generate_samples(num_draws, temperature)

            draws = []
            for i in range(num_draws):
                # Convert probabilities to draw
                draw_probs = probs[i].cpu()

                # Select top-k numbers by probability
                _, indices = torch.topk(draw_probs, self.config.draw_size)
                draw = sorted((indices + 1).tolist())  # 1-indexed
                draws.append(draw)

            return draws

        def compare_distributions(self, real_draws: list[list[int]],
                                 num_synthetic: int = 1000) -> dict:
            """
            Compare real draws against synthetic draws from the model.

            This validates whether the model has learned the true distribution.

            Args:
                real_draws: Real historical draws
                num_synthetic: Number of synthetic draws to generate

            Returns:
                Distribution comparison metrics
            """
            synthetic_draws = self.generate_synthetic_draws(num_synthetic)

            # Compute statistics for both distributions
            def compute_stats(draws):
                sums = [sum(d) for d in draws]
                return {
                    "mean_sum": sum(sums) / len(sums),
                    "std_sum": math.sqrt(sum((s - sum(sums)/len(sums))**2 for s in sums) / len(sums)),
                    "min_sum": min(sums),
                    "max_sum": max(sums)
                }

            real_stats = compute_stats(real_draws)
            synthetic_stats = compute_stats(synthetic_draws)

            # Compute frequency distributions
            from collections import Counter

            real_freq = Counter()
            for draw in real_draws:
                for num in draw:
                    real_freq[num] += 1

            synthetic_freq = Counter()
            for draw in synthetic_draws:
                for num in draw:
                    synthetic_freq[num] += 1

            # Normalize
            total_real = sum(real_freq.values())
            total_synthetic = sum(synthetic_freq.values())

            # Compute KL divergence approximation
            kl_div = 0.0
            for num in range(1, self.config.pool_size + 1):
                p = (real_freq.get(num, 0) + 1) / (total_real + self.config.pool_size)
                q = (synthetic_freq.get(num, 0) + 1) / (total_synthetic + self.config.pool_size)
                kl_div += p * math.log(p / q)

            return {
                "real_statistics": real_stats,
                "synthetic_statistics": synthetic_stats,
                "kl_divergence": kl_div,
                "distribution_match": kl_div < 0.1,  # Threshold for good match
                "interpretation": self._interpret_distribution_comparison(kl_div)
            }

        def _interpret_distribution_comparison(self, kl_div: float) -> str:
            """Interpret KL divergence between distributions."""
            if kl_div < 0.05:
                return ("Excellent match: Model has learned the true draw distribution. "
                        "This suggests draws follow expected random patterns.")
            elif kl_div < 0.15:
                return ("Good match: Model captures main characteristics of draws. "
                        "Minor deviations may be due to sample size or subtle patterns.")
            elif kl_div < 0.3:
                return ("Moderate match: Some discrepancies between model and reality. "
                        "May indicate patterns the model hasn't captured or overfitting.")
            else:
                return ("Poor match: Significant difference between distributions. "
                        "Either the model needs more training or draws have unusual patterns.")

        def save_model(self, filepath: str) -> None:
            """Save model state to file."""
            torch.save({
                "model_state": self.model.state_dict(),
                "config": self.config,
                "train_losses": self.train_losses,
                "anomaly_threshold": self.anomaly_threshold
            }, filepath)

        def load_model(self, filepath: str) -> None:
            """Load model state from file."""
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.config = checkpoint["config"]
            self.train_losses = checkpoint.get("train_losses", [])
            self.anomaly_threshold = checkpoint.get("anomaly_threshold")


else:
    # Stub classes when PyTorch is not available

    class LotteryVAE:
        """Stub class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for VAE models. "
                "Install with: pip install torch"
            )


    class VAETrainer:
        """Stub class when PyTorch is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for VAE training. "
                "Install with: pip install torch"
            )


def create_default_config(game: str = "loto_649") -> VAEConfig:
    """
    Create default VAE configuration for a specific lottery game.

    Args:
        game: One of "loto_649", "loto_540", "joker"

    Returns:
        VAEConfig for the game
    """
    configs = {
        "loto_649": VAEConfig(
            pool_size=49,
            draw_size=6,
            latent_dim=16,
            hidden_dims=(64, 32)
        ),
        "loto_540": VAEConfig(
            pool_size=40,
            draw_size=5,
            latent_dim=12,
            hidden_dims=(48, 24)
        ),
        "joker": VAEConfig(
            pool_size=45,
            draw_size=5,
            latent_dim=14,
            hidden_dims=(56, 28)
        )
    }

    return configs.get(game, configs["loto_649"])


def quick_anomaly_scan(draws: list[list[int]], game: str = "loto_649") -> dict:
    """
    Quick VAE-based anomaly scan of lottery draws.

    Trains a model and identifies potentially anomalous draws.

    Args:
        draws: Historical draws
        game: Game type

    Returns:
        Anomaly scan results
    """
    if not TORCH_AVAILABLE:
        return {
            "error": "PyTorch not available",
            "message": "Install PyTorch to use VAE analysis: pip install torch"
        }

    config = create_default_config(game)
    config.num_epochs = 50  # Quick training

    trainer = VAETrainer(config)

    # Train
    print("Training VAE model...")
    train_results = trainer.train(draws, val_split=0.2)

    # Detect anomalies
    print("Scanning for anomalies...")
    anomalies = trainer.detect_anomalies(draws)

    # Find actual anomalies
    flagged = [a for a in anomalies if a["is_anomaly"]]

    # Analyze latent space
    latent_analysis = trainer.analyze_latent_space(draws[-500:])

    # Compare distributions
    dist_comparison = trainer.compare_distributions(draws[-500:])

    return {
        "training": train_results,
        "total_draws_scanned": len(draws),
        "anomalies_detected": len(flagged),
        "anomaly_rate": len(flagged) / len(draws) if draws else 0,
        "top_anomalies": sorted(flagged, key=lambda x: -x["anomaly_score"])[:10],
        "latent_space": latent_analysis,
        "distribution_validation": dist_comparison,
        "conclusion": _generate_conclusion(len(flagged), len(draws), dist_comparison)
    }


def _generate_conclusion(num_anomalies: int, total: int, dist_comp: dict) -> str:
    """Generate conclusion from anomaly scan."""
    anomaly_rate = num_anomalies / total if total > 0 else 0

    lines = []

    # Anomaly assessment
    if anomaly_rate < 0.03:
        lines.append("ANOMALY SCAN: No significant anomalies detected.")
        lines.append("All draws fall within expected variance for random generation.")
    elif anomaly_rate < 0.08:
        lines.append("ANOMALY SCAN: Few outliers detected (within normal range).")
        lines.append("Some draws are unusual but likely due to natural variance.")
    else:
        lines.append("ANOMALY SCAN: Elevated number of anomalies detected.")
        lines.append("This warrants further investigation.")
        lines.append("Could indicate data quality issues or genuine patterns.")

    # Distribution assessment
    if dist_comp.get("distribution_match", False):
        lines.append("\nDISTRIBUTION: Model learned true draw distribution well.")
        lines.append("Synthetic draws match real draws, confirming expected patterns.")
    else:
        lines.append("\nDISTRIBUTION: Some mismatch between model and real draws.")
        lines.append("May indicate subtle patterns or insufficient training data.")

    return "\n".join(lines)
