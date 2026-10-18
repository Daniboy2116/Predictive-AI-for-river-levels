"""Modelo de Google: TimesFM (modelo fundacional) con Prophet como alternativa.

- **TimesFM** (``timesfm``): inferencia *zero-shot* a 1 día vista. Para cada día de
  test se usa como contexto la serie real (metros) hasta t-1 (máx. ``CONTEXT_LEN`` días).
- **Prophet** (fallback): tendencia + estacionalidad anual + regresor ``lag_1``
  (nivel real del día anterior), ajustado con train + val y evaluado t+1 en test.

Se intenta TimesFM primero; si no está instalado o falla al cargar, se usa Prophet.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.evaluate import evaluate_and_save
from src.preprocessing import STATIONS, StationData, load_data, prepare_station

CONTEXT_LEN = 512
TIMESFM_REPO = "google/timesfm-1.0-200m-pytorch"


# --------------------------------------------------------------------------- TimesFM
def _load_timesfm():
    """Carga TimesFM (importación perezosa). Lanza ImportError/RuntimeError si no es posible."""
    try:
        import timesfm
    except ImportError as exc:
        raise ImportError("Falta 'timesfm' (pip install timesfm; requiere Python 3.10-3.11).") from exc
    try:
        return timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="cpu", per_core_batch_size=32, horizon_len=1),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=TIMESFM_REPO),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"No se pudo cargar TimesFM: {exc}") from exc


def predict_timesfm(model, d: StationData) -> np.ndarray:
    """Predicciones t+1 zero-shot para cada día de test usando contexto real hasta t-1."""
    values = d.raw.to_numpy(dtype=np.float32)
    positions = d.raw.index.get_indexer(d.y_test.index)
    contexts = [values[max(0, p - CONTEXT_LEN):p] for p in positions]
    point, _ = model.forecast(contexts, freq=[0] * len(contexts))
    return np.asarray(point)[:, 0]


# --------------------------------------------------------------------------- Prophet
def _import_prophet():
    """Importación perezosa de Prophet."""
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("Falta 'prophet'. Instálalo con: pip install prophet") from exc
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    return Prophet


def predict_prophet(d: StationData) -> np.ndarray:
    """Prophet con estacionalidad anual + regresor lag_1; predicción t+1 sobre test."""
    Prophet = _import_prophet()
    frame = pd.DataFrame({"ds": d.raw.index, "y": d.raw.values,
                          "lag_1": d.raw.shift(1).values}).dropna()
    fit = frame[frame["ds"] <= d.y_val.index[-1]]
    test = frame[frame["ds"].isin(d.y_test.index)]
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, changepoint_prior_scale=0.05)
    model.add_regressor("lag_1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(fit)
    return model.predict(test[["ds", "lag_1"]])["yhat"].to_numpy()


# --------------------------------------------------------------------------- Runner
def run(stations: list[str] | None = None, prefer: str = "timesfm"
        ) -> dict[str, dict[str, float]]:
    """Evalúa TimesFM (o Prophet como fallback) para cada estación y guarda resultados."""
    tfm = None
    if prefer == "timesfm":
        try:
            tfm = _load_timesfm()
        except (ImportError, RuntimeError) as exc:
            print(f"[google_model] TimesFM no disponible → usando Prophet. Motivo: {exc}")
    name = "TimesFM" if tfm is not None else "Prophet"
    df = load_data()
    results: dict[str, dict[str, float]] = {}
    for st in stations or STATIONS:
        d = prepare_station(df, st, save_scaler=False)
        y_pred = predict_timesfm(tfm, d) if tfm is not None else predict_prophet(d)
        m = evaluate_and_save(st, name, d.y_test.index, d.raw_test.values, y_pred)
        results[st] = m
        print(f"{st:15s} {name:8s} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")
    return results


if __name__ == "__main__":
    run()
