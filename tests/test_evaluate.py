"""Pruebas de las métricas de evaluación."""
import numpy as np
from sklearn.metrics import r2_score

from src.evaluate import calcular_metricas


def test_metricas_de_una_prediccion_perfecta():
    """Si la predicción es exacta: MAE = RMSE = 0 y NSE = 1."""
    y = np.array([3.0, 5.0, 9.0, 16.0])
    metricas = calcular_metricas(y, y)
    assert metricas["MAE"] == 0.0
    assert metricas["RMSE"] == 0.0
    assert metricas["NSE"] == 1.0


def test_nse_coincide_con_r2():
    """El NSE de Nash-Sutcliffe es la misma fórmula que el R² de scikit-learn."""
    rng = np.random.default_rng(1)
    y_real = rng.normal(9, 4, 500)
    y_pred = y_real + rng.normal(0, 1, 500)
    metricas = calcular_metricas(y_real, y_pred)
    assert np.isclose(metricas["NSE"], r2_score(y_real, y_pred))


def test_predecir_la_media_da_nse_cero():
    """Predecir siempre la media observada equivale a NSE = 0."""
    y_real = np.array([2.0, 4.0, 6.0, 8.0])
    y_pred = np.full(4, y_real.mean())
    assert np.isclose(calcular_metricas(y_real, y_pred)["NSE"], 0.0)


def test_mae_y_rmse_conocidos():
    """Errores de 1 y 3 m: MAE = 2 y RMSE = raíz de 5."""
    metricas = calcular_metricas([1.0, -1.0], [2.0, -4.0])
    assert np.isclose(metricas["MAE"], 2.0)
    assert np.isclose(metricas["RMSE"], np.sqrt(5.0))
