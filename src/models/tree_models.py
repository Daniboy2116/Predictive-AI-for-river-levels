"""Modelos de ensamblado de árboles sobre lag features.

- ``RandomForest`` (Bagging): ``sklearn.ensemble.RandomForestRegressor``.
- ``XGBoost`` (Boosting): ``xgboost.XGBRegressor`` con *early stopping* sobre validación.

Predicción a un paso (t+1) usando los lags reales t-1..t-7 del conjunto de prueba.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.evaluate import evaluate_and_save
from src.preprocessing import SEED, STATIONS, StationData, load_data, prepare_station


def build_random_forest(**kwargs: Any) -> RandomForestRegressor:
    """Crea un RandomForestRegressor con hiperparámetros por defecto razonables."""
    params = dict(n_estimators=300, max_depth=12, min_samples_leaf=2,
                  n_jobs=-1, random_state=SEED)
    params.update(kwargs)
    return RandomForestRegressor(**params)


def build_xgboost(**kwargs: Any):
    """Crea un XGBRegressor (importación perezosa de xgboost)."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Falta 'xgboost'. Instálalo con: pip install xgboost") from exc
    params = dict(n_estimators=1000, learning_rate=0.03, max_depth=5, subsample=0.9,
                  colsample_bytree=0.9, early_stopping_rounds=50,
                  random_state=SEED, n_jobs=-1)
    params.update(kwargs)
    return XGBRegressor(**params)


def train_random_forest(d: StationData) -> np.ndarray:
    """Entrena RF con train y devuelve predicciones de test desnormalizadas."""
    model = build_random_forest().fit(d.X_train, d.y_train)
    return d.inverse(model.predict(d.X_test))


def train_xgboost(d: StationData) -> np.ndarray:
    """Entrena XGBoost (early stopping con val) y devuelve predicciones de test desnormalizadas."""
    model = build_xgboost()
    model.fit(d.X_train, d.y_train, eval_set=[(d.X_val, d.y_val)], verbose=False)
    return d.inverse(model.predict(d.X_test))


TREE_MODELS = {"RandomForest": train_random_forest, "XGBoost": train_xgboost}


def run(stations: list[str] | None = None) -> dict[str, dict[str, dict[str, float]]]:
    """Entrena y evalúa los modelos de árboles para cada estación; guarda resultados."""
    df = load_data()
    results: dict[str, dict[str, dict[str, float]]] = {}
    for st in stations or STATIONS:
        d = prepare_station(df, st)
        y_true = d.raw_test.values
        for name, fn in TREE_MODELS.items():
            m = evaluate_and_save(st, name, d.y_test.index, y_true, fn(d))
            results.setdefault(st, {})[name] = m
            print(f"{st:15s} {name:13s} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")
    return results


if __name__ == "__main__":
    run()
