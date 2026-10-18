"""Modelo estadístico ARIMA sobre la serie original (metros, sin escalar).

Estrategia:
1. Selección de orden (p, d, q) por AIC usando SOLO train
   (``pmdarima.auto_arima`` si está disponible; si no, búsqueda en rejilla con statsmodels).
2. Reajuste de parámetros con train + val.
3. Predicción a 1 día vista sobre test: se aplican los parámetros fijos a la serie
   completa y se obtienen las predicciones one-step-ahead del filtro de Kalman
   (cada predicción usa solo observaciones hasta t-1, sin reentrenar).
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from src.evaluate import evaluate_and_save
from src.preprocessing import STATIONS, StationData, load_data, prepare_station

MODEL_NAME = "ARIMA"


def _import_arima():
    """Importación perezosa de statsmodels."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Falta 'statsmodels'. Instálalo con: pip install statsmodels") from exc
    return ARIMA


def select_order(train: pd.Series, max_p: int = 3, max_q: int = 2,
                 d: int = 1) -> tuple[int, int, int]:
    """Selecciona (p, d, q) minimizando AIC sobre train."""
    try:
        import pmdarima as pm  # opcional
        model = pm.auto_arima(train, d=d, max_p=max_p, max_q=max_q, seasonal=False,
                              stepwise=True, suppress_warnings=True, error_action="ignore")
        return tuple(model.order)  # type: ignore[return-value]
    except ImportError:
        pass
    ARIMA = _import_arima()
    best, best_aic = (1, d, 0), np.inf
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        if p == q == 0:
            continue
        try:
            aic = ARIMA(train, order=(p, d, q)).fit().aic
        except Exception:  # noqa: BLE001 - órdenes no convergentes
            continue
        if aic < best_aic:
            best, best_aic = (p, d, q), aic
    return best


def predict_one_step(d: StationData, order: tuple[int, int, int] | None = None
                     ) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Ajusta ARIMA y devuelve predicciones t+1 sobre test (metros) y el orden usado."""
    ARIMA = _import_arima()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        order = order or select_order(d.raw_train)
        fit_series = d.raw.loc[: d.y_val.index[-1]]          # train + val
        res = ARIMA(fit_series, order=order).fit()
        full = res.apply(d.raw)                              # mismos parámetros, serie completa
        pred = full.get_prediction(start=d.y_test.index[0], end=d.y_test.index[-1])
    return pred.predicted_mean.to_numpy(), order


def run(stations: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Entrena y evalúa ARIMA para cada estación; guarda resultados en outputs/."""
    df = load_data()
    results: dict[str, dict[str, float]] = {}
    for st in stations or STATIONS:
        d = prepare_station(df, st, save_scaler=False)
        y_pred, order = predict_one_step(d)
        m = evaluate_and_save(st, MODEL_NAME, d.y_test.index, d.raw_test.values, y_pred)
        results[st] = m
        print(f"{st:15s} ARIMA{order} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")
    return results


if __name__ == "__main__":
    run()
