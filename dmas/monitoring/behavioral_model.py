"""
GRU-based Behavioral Anomaly Detector.

Learns temporal dependencies in device communication sequences.
At inference time, reconstruction error is used as the anomaly score.

Paper reference — Section III-B-1:
  'A two-layer GRU-based RNN (hidden dimension 64, dropout 0.2) trained
   to model temporal dependencies in device communication sequences.
   Each sequence window is 20 time steps (10-second intervals).
   Trained with Adam (lr=1e-3, batch 64) for 50 epochs, 70/15/15 split.'
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, List

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure-numpy fallback model (no PyTorch required for POC demo)
# ---------------------------------------------------------------------------

class _NumpyGRUCell:
    """Single GRU cell implemented in numpy (for environments without torch)."""

    def __init__(self, input_size: int, hidden_size: int, rng: np.random.Generator):
        scale = 1.0 / math.sqrt(hidden_size)
        self.W_z = rng.uniform(-scale, scale, (hidden_size, input_size + hidden_size))
        self.W_r = rng.uniform(-scale, scale, (hidden_size, input_size + hidden_size))
        self.W_h = rng.uniform(-scale, scale, (hidden_size, input_size + hidden_size))
        self.b_z = np.zeros(hidden_size)
        self.b_r = np.zeros(hidden_size)
        self.b_h = np.zeros(hidden_size)

    def forward(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        xh = np.concatenate([x, h])
        z = _sigmoid(self.W_z @ xh + self.b_z)
        r = _sigmoid(self.W_r @ xh + self.b_r)
        h_tilde = np.tanh(self.W_h @ np.concatenate([x, r * h]) + self.b_h)
        return (1 - z) * h + z * h_tilde


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class _NumpyBehavioralModel:
    """Two-layer GRU autoencoder implemented purely in numpy."""

    def __init__(self, input_features: int, hidden_dim: int, rng: np.random.Generator):
        self.hidden_dim = hidden_dim
        self.cell1 = _NumpyGRUCell(input_features, hidden_dim, rng)
        self.cell2 = _NumpyGRUCell(hidden_dim, hidden_dim, rng)
        # Linear output head: hidden -> input_features (reconstruction)
        scale = 1.0 / math.sqrt(hidden_dim)
        self.W_out = rng.uniform(-scale, scale, (input_features, hidden_dim))
        self.b_out = np.zeros(input_features)
        self.input_features = input_features

    def forward(self, sequence: np.ndarray) -> np.ndarray:
        """sequence: (T, F) -> reconstructed last step (F,)"""
        h1 = np.zeros(self.hidden_dim)
        h2 = np.zeros(self.hidden_dim)
        for step in sequence:
            h1 = self.cell1.forward(step, h1)
            h2 = self.cell2.forward(h1, h2)
        return self.W_out @ h2 + self.b_out

    def reconstruction_error(self, sequence: np.ndarray) -> float:
        if len(sequence) == 0:
            return 0.0
        pred = self.forward(sequence)
        actual = sequence[-1]
        mse = float(np.mean((pred - actual) ** 2))
        return mse


# ---------------------------------------------------------------------------
# PyTorch model (used when torch is available)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class _TorchGRUAutoencoder(nn.Module):
        def __init__(self, input_features: int, hidden_dim: int,
                     num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.encoder = nn.GRU(
                input_features, hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.decoder = nn.Linear(hidden_dim, input_features)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.encoder(x)
            return self.decoder(out[:, -1, :])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BehavioralModel:
    """
    Wraps either a PyTorch GRU autoencoder or a pure-numpy fallback.

    Usage
    -----
    model = BehavioralModel(input_features=8, hidden_dim=64)
    # Feed a (window_size, n_features) numpy array:
    score = model.score(sequence)   # float in [0, 1]
    """

    def __init__(
        self,
        input_features: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        window_size: int = 20,
        seed: int = 42,
    ) -> None:
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self._buffer: List[np.ndarray] = []

        rng = np.random.default_rng(seed)

        if _TORCH_AVAILABLE:
            self._model = _TorchGRUAutoencoder(input_features, hidden_dim,
                                                num_layers, dropout)
            self._model.eval()
            self._backend = "torch"
        else:
            self._model = _NumpyBehavioralModel(input_features, hidden_dim, rng)
            self._backend = "numpy"

        # Baseline reconstruction error (estimated from random normal traffic)
        self._baseline_err: float = 1.0
        self._err_var: float = 1.0

    # ------------------------------------------------------------------
    # Training (simplified online update — full training outside POC scope)
    # ------------------------------------------------------------------

    def train_step(self, sequence: np.ndarray) -> None:
        """Single gradient-descent step (torch backend only)."""
        if self._backend != "torch":
            return
        import torch
        import torch.nn.functional as F
        x = torch.FloatTensor(sequence).unsqueeze(0)  # (1, T, F)
        pred = self._model(x)
        target = torch.FloatTensor(sequence[-1]).unsqueeze(0)
        loss = F.mse_loss(pred, target)
        # Lightweight in-place SGD for POC
        loss.backward()
        with torch.no_grad():
            for p in self._model.parameters():
                p -= 1e-3 * p.grad
                p.grad.zero_()

    def calibrate(self, normal_sequences: List[np.ndarray]) -> None:
        """
        Compute baseline reconstruction error on normal traffic so that
        the score() method can normalise anomaly magnitude.
        """
        if not normal_sequences:
            return
        errors = [self._reconstruction_error(s) for s in normal_sequences]
        self._baseline_err = float(np.mean(errors))
        self._err_var = float(np.var(errors)) + 1e-9

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def push(self, feature_vector: np.ndarray) -> None:
        """Push one time-step into the sliding window buffer."""
        self._buffer.append(feature_vector.astype(np.float32))
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

    def score(self, sequence: Optional[np.ndarray] = None) -> float:
        """
        Return anomaly score in [0, 1].  If `sequence` is None, use the
        internal rolling buffer.
        """
        if sequence is None:
            if len(self._buffer) < 2:
                return 0.0
            sequence = np.array(self._buffer)

        err = self._reconstruction_error(sequence)
        # Normalise using z-score relative to baseline, then sigmoid
        z = (err - self._baseline_err) / math.sqrt(self._err_var)
        return float(1.0 / (1.0 + math.exp(-z)))

    def _reconstruction_error(self, sequence: np.ndarray) -> float:
        if self._backend == "torch":
            import torch
            with torch.no_grad():
                x = torch.FloatTensor(sequence).unsqueeze(0)
                pred = self._model(x).squeeze(0).numpy()
            return float(np.mean((pred - sequence[-1]) ** 2))
        else:
            return self._model.reconstruction_error(sequence)

    def __repr__(self) -> str:
        return (f"BehavioralModel(backend={self._backend}, "
                f"hidden_dim={self.hidden_dim}, "
                f"window={self.window_size})")
