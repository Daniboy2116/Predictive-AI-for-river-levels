"""Modelo estadístico ARIMA sobre la serie original (en metros, sin escalar).

ARIMA(p, d, q) predice el nivel de mañana combinando:
- p: cuántos niveles pasados se usan (parte autorregresiva, AR),
- d: cuántas veces se diferencia la serie para quitar la tendencia (I),
- q: cuántos errores pasados se usan (media móvil, MA).

Ejecutar:  python -m src.models.arima_model
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.evaluate import evaluar_y_guardar
from src.preprocessing import ESTACIONES, cargar_datos, preparar_estacion

# Valores de p y q que se prueban al buscar el mejor modelo (d se deja fijo en 1)
VALORES_P = [0, 1, 2, 3]
VALORES_Q = [0, 1, 2]
VALOR_D = 1


def ejecutar_arima(df: pd.DataFrame, estacion: str) -> dict:
    """Busca el mejor orden ARIMA, lo entrena y evalúa para una estación.

    Args:
        df: DataFrame completo devuelto por ``cargar_datos``.
        estacion: nombre de la estación.

    Returns:
        Diccionario con las métricas (RMSE, MAE, R2) en metros.
    """
    # statsmodels avisa a menudo de problemas de convergencia en órdenes de prueba;
    # los silenciamos para que la salida sea legible.
    warnings.filterwarnings("ignore")

    # === PASO 1: PREPARACIÓN DE DATOS ===
    # [JUSTIFICACIÓN]: ARIMA no usa una matriz X de lags ni un vector y. Trabaja
    # directamente sobre la serie temporal completa y en metros (sin escalar),
    # porque el propio modelo calcula internamente sus "lags". Solo tomamos de
    # preparar_estacion las fechas de corte para respetar el split 70/15/15.
    datos = preparar_estacion(df, estacion)
    serie_completa = datos["serie"]
    serie_train = serie_completa[serie_completa.index <= datos["fecha_fin_train"]]
    serie_train_val = serie_completa[serie_completa.index <= datos["fecha_fin_val"]]
    fechas_test = datos["fechas_test"]

    # === PASO 2: DEFINICIÓN / CONFIGURACIÓN DEL MODELO ===
    # [JUSTIFICACIÓN]: en ARIMA "definir el modelo" significa elegir el orden (p, d, q).
    # Probamos varias combinaciones SOLO con train y nos quedamos con la de menor AIC
    # (criterio que premia el buen ajuste y penaliza la complejidad).
    mejor_orden = (1, VALOR_D, 0)
    mejor_aic = float("inf")
    for p in VALORES_P:
        for q in VALORES_Q:
            if p == 0 and q == 0:
                continue  # un modelo sin parte AR ni MA no tiene sentido aquí
            orden = (p, VALOR_D, q)
            # [JUSTIFICACIÓN]: con algunos órdenes el cálculo interno de ARIMA es
            # numéricamente inestable y lanza un error. Ese candidato simplemente
            # se descarta y se sigue probando el resto (no es un "fallback" de librería).
            try:
                resultado_prueba = ARIMA(serie_train, order=orden).fit()
            except (np.linalg.LinAlgError, ValueError):
                print(estacion.ljust(15), "orden", orden, "descartado (no converge)")
                continue
            if resultado_prueba.aic < mejor_aic:
                mejor_aic = resultado_prueba.aic
                mejor_orden = orden
    print(estacion.ljust(15), "orden ARIMA elegido:", mejor_orden)

    modelo = ARIMA(serie_train_val, order=mejor_orden)

    # === PASO 3: ENTRENAMIENTO ===
    # Una vez elegido el orden, se reajustan los parámetros con train + val.
    modelo_entrenado = modelo.fit()

    # === PASO 4: PREDICCIÓN (YA EN METROS) ===
    # [JUSTIFICACIÓN]: no hay que invertir ninguna escala porque ARIMA se entrenó en metros.
    # Para predecir a 1 día vista en test, aplicamos los MISMOS parámetros a la serie
    # completa: así cada predicción del día t usa solo los datos reales hasta t-1
    # (sin volver a entrenar).
    modelo_serie_completa = modelo_entrenado.apply(serie_completa)
    prediccion = modelo_serie_completa.get_prediction(start=fechas_test[0], end=fechas_test[-1])
    y_pred = prediccion.predicted_mean.to_numpy()
    y_real = datos["y_test_metros"]

    metricas = evaluar_y_guardar(estacion, "ARIMA", fechas_test, y_real, y_pred)
    return metricas


def ejecutar() -> None:
    """Entrena y evalúa ARIMA en todas las estaciones."""
    df = cargar_datos()
    for estacion in ESTACIONES:
        ejecutar_arima(df, estacion)


if __name__ == "__main__":
    ejecutar()
