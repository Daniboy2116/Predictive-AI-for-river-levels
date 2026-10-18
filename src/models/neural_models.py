"""Redes profundas en PyTorch para predicción t+1: LSTM, CNN 1D y Transformer.

Entrada: ventanas de 7 pasos (lags escalados en orden cronológico) con forma
``(batch, 7, 1)``. Entrenamiento con Adam + MSE y *early stopping* sobre
validación (se restauran los mejores pesos). Predicciones de test desnormalizadas.
"""
from __future__ import annotations

import copy
import random
from typing import Callable

import numpy as np

from src.evaluate import evaluate_and_save
from src.preprocessing import SEED, STATIONS, StationData, load_data, make_sequences, prepare_station


def _import_torch():
    """Importación perezosa de PyTorch con mensaje claro."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Falta 'torch'. Instálalo con: pip install torch") from exc
    return torch, nn


torch, nn = _import_torch()


def set_seed(seed: int = SEED) -> None:
    """Fija semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class LSTMNet(nn.Module):
    """LSTM apilada (2 capas) + capas Dense."""

    def __init__(self, hidden: int = 64, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class CNN1DNet(nn.Module):
    """Conv1D + MaxPooling1D para patrones locales + capas Dense."""

    def __init__(self, n_steps: int = 7, filters: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(filters, filters, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(filters * (n_steps // 2), 32),
                                  nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2))).squeeze(-1)  # (B,1,T)


class TransformerNet(nn.Module):
    """Encoder Transformer (self-attention multi-cabeza) con embedding posicional aprendido."""

    def __init__(self, n_steps: int = 7, d_model: int = 32, heads: int = 4,
                 layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_steps, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=64,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        h = self.encoder(self.embed(x) + self.pos)
        return self.head(h[:, -1]).squeeze(-1)


NEURAL_MODELS: dict[str, Callable[[], nn.Module]] = {
    "LSTM": LSTMNet, "CNN1D": CNN1DNet, "Transformer": TransformerNet,
}


def train_model(model: nn.Module, d: StationData, epochs: int = 200, batch: int = 32,
                lr: float = 1e-3, patience: int = 20) -> nn.Module:
    """Entrena con train, early stopping sobre val; devuelve el modelo con mejores pesos."""
    Xtr = torch.tensor(make_sequences(d.X_train))
    ytr = torch.tensor(d.y_train.to_numpy(np.float32))
    Xva = torch.tensor(make_sequences(d.X_val))
    yva = torch.tensor(d.y_val.to_numpy(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(SEED)
    best_loss, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xtr), generator=gen).split(batch):  # barajar ventanas, no la serie
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva).item()
        if val_loss < best_loss - 1e-7:
            best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: nn.Module, d: StationData) -> np.ndarray:
    """Predicciones de test desnormalizadas (metros)."""
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(make_sequences(d.X_test))).numpy()
    return d.inverse(out)


def run(stations: list[str] | None = None, models: list[str] | None = None
        ) -> dict[str, dict[str, dict[str, float]]]:
    """Entrena y evalúa las redes para cada estación; guarda resultados en outputs/."""
    torch.set_num_threads(max(1, torch.get_num_threads()))
    df = load_data()
    results: dict[str, dict[str, dict[str, float]]] = {}
    for st in stations or STATIONS:
        d = prepare_station(df, st)
        for name in models or NEURAL_MODELS:
            set_seed()
            model = train_model(NEURAL_MODELS[name](), d)
            m = evaluate_and_save(st, name, d.y_test.index, d.raw_test.values, predict(model, d))
            results.setdefault(st, {})[name] = m
            print(f"{st:15s} {name:12s} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}",
                  flush=True)
    return results


if __name__ == "__main__":
    run()
