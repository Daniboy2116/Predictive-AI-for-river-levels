"""Métricas de evaluación y exportación de resultados.

- ``outputs/metrics.json``: ``{estacion: {modelo: {RMSE, MAE, R2}}}``
- ``outputs/predictions.csv``: formato largo ``Fecha, Estacion, Modelo, Real, Prediccion``
  (escala original, metros).

Las funciones de guardado hacen *merge* con resultados previos, de modo que
cada modelo puede ejecutarse por separado sin sobrescribir a los demás.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"
PRED_COLUMNS = ["Fecha", "Estacion", "Modelo", "Real", "Prediccion"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Raíz del error cuadrático medio."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Error absoluto medio."""
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coeficiente de determinación R²."""
    return float(r2_score(y_true, y_pred))


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Calcula RMSE, MAE y R² (en la escala recibida; usar escala original)."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"Dimensiones distintas: real={yt.shape}, pred={yp.shape}")
    return {"RMSE": rmse(yt, yp), "MAE": mae(yt, yp), "R2": r2(yt, yp)}


def save_metrics(station: str, model: str, metrics: dict[str, float],
                 path: Path = METRICS_PATH) -> None:
    """Inserta/actualiza las métricas de (estación, modelo) en el JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.setdefault(station, {})[model] = {k: round(v, 6) for k, v in metrics.items()}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_predictions(station: str, model: str, dates, y_true, y_pred,
                     path: Path = PREDICTIONS_PATH) -> None:
    """Reemplaza las predicciones de (estación, modelo) en el CSV de formato largo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame({
        "Fecha": pd.to_datetime(dates), "Estacion": station, "Modelo": model,
        "Real": np.asarray(y_true, dtype=float).ravel(),
        "Prediccion": np.asarray(y_pred, dtype=float).ravel(),
    })
    if path.exists():
        old = pd.read_csv(path, parse_dates=["Fecha"])
        old = old[~((old["Estacion"] == station) & (old["Modelo"] == model))]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(["Estacion", "Modelo", "Fecha"]).to_csv(path, index=False)


def evaluate_and_save(station: str, model: str, dates, y_true, y_pred) -> dict[str, float]:
    """Calcula métricas y persiste métricas + predicciones. Devuelve las métricas."""
    metrics = compute_metrics(y_true, y_pred)
    save_metrics(station, model, metrics)
    save_predictions(station, model, dates, y_true, y_pred)
    return metrics


def load_metrics(path: Path = METRICS_PATH) -> pd.DataFrame:
    """Carga metrics.json como DataFrame (Estacion, Modelo, RMSE, MAE, R2)."""
    if not path.exists():
        return pd.DataFrame(columns=["Estacion", "Modelo", "RMSE", "MAE", "R2"])
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [{"Estacion": s, "Modelo": m, **v} for s, ms in data.items() for m, v in ms.items()]
    return pd.DataFrame(rows)


def load_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    """Carga predictions.csv (vacío si no existe)."""
    if not path.exists():
        return pd.DataFrame(columns=PRED_COLUMNS)
    return pd.read_csv(path, parse_dates=["Fecha"])
