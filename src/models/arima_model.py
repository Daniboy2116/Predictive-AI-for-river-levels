"""Modelo estadístico SARIMAX: anomalía ARMA + estaciones aguas arriba como exógenas.

Nivel(t) = climatología(t) + anomalía(t), y la anomalía se modela como:

    anomalía(t) = β · [anomalías aguas arriba en t - 15] + u(t),   u(t) ~ ARMA(p, q)

- La "S" (estacionalidad) la aporta la climatología armónica ajustada con train.
- La "X" son las anomalías de las otras estaciones con 15 días de retraso: así,
  para pronosticar hasta 15 días, todas las exógenas futuras ya se conocen en t.
- ARMA(p, q): p anomalías pasadas (AR) y q errores pasados (MA).

Ejecutar:  python -m src.models.arima_model
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from src.configuracion import ESTACIONES, ESTACIONES_OBJETIVO, FECHA_FIN_TRAIN, HORIZONTE_PRINCIPAL, HORIZONTES
from src.evaluate import armar_predicciones, evaluar_y_guardar
from src.preprocessing import ajustar_climatologias, cargar_datos, evaluar_climatologia, preparar_datos

# Órdenes ARMA que se prueban (d = 0 porque la anomalía es estacionaria: prueba ADF)
VALORES_P = [1, 2, 3]
VALORES_Q = [0, 1]
REZAGO_EXOGENAS = HORIZONTE_PRINCIPAL


def ejecutar_sarimax(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    """Busca el mejor orden, entrena SARIMAX y pronostica 1..15 días desde cada día de validación y prueba.

    Returns:
        Métricas de prueba por horizonte (ver ``evaluar_y_guardar``).
    """
    # === PASO 1: PREPARACIÓN DE DATOS ===
    # [JUSTIFICACIÓN]: SARIMAX no usa la tabla de 79 variables. Trabaja sobre la serie
    # temporal de la anomalía (en metros, sin escalar) y recibe aparte las exógenas.
    climatologias = ajustar_climatologias(niveles, observados)
    anomalias = pd.DataFrame(index=niveles.index)
    for nombre in ESTACIONES:
        anomalias[nombre] = niveles[nombre] - evaluar_climatologia(climatologias[nombre], niveles.index)

    otras_estaciones = []
    for nombre in ESTACIONES:
        if nombre != estacion:
            otras_estaciones.append(nombre)
    exogenas = anomalias[otras_estaciones].shift(REZAGO_EXOGENAS)
    exogenas.columns = [nombre + "_t-" + str(REZAGO_EXOGENAS) for nombre in otras_estaciones]

    # Los primeros días no tienen exógenas (no hay datos 15 días antes): se descartan
    anomalia_objetivo = anomalias[estacion].iloc[REZAGO_EXOGENAS:]
    exogenas = exogenas.iloc[REZAGO_EXOGENAS:]
    en_train = anomalia_objetivo.index <= pd.Timestamp(FECHA_FIN_TRAIN)

    # Prueba de Dickey-Fuller aumentada: p < 0.05 indica serie estacionaria (no hace falta diferenciar)
    valor_p_adf = adfuller(anomalia_objetivo[en_train].to_numpy(), maxlag=30, result_object=True).pvalue
    print(estacion.ljust(15), "ADF p-valor = %.2e -> d = 0" % valor_p_adf)

    # === PASO 2: DEFINICIÓN / CONFIGURACIÓN DEL MODELO ===
    # [JUSTIFICACIÓN]: en SARIMAX "definir el modelo" es elegir el orden (p, 0, q).
    # Probamos varias combinaciones SOLO con train y nos quedamos con la de menor AIC.
    mejor_orden = (1, 0, 0)
    mejor_aic = float("inf")
    for p in VALORES_P:
        for q in VALORES_Q:
            orden = (p, 0, q)
            # statsmodels avisa de problemas de convergencia en órdenes de prueba;
            # los silenciamos SOLO dentro de este bloque para que la salida sea legible.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                resultado_prueba = SARIMAX(anomalia_objetivo[en_train], exog=exogenas[en_train],
                                           order=orden, trend="n").fit(disp=False)
            if resultado_prueba.aic < mejor_aic:
                mejor_aic = resultado_prueba.aic
                mejor_orden = orden
    print(estacion.ljust(15), "orden SARIMAX elegido:", mejor_orden)
    modelo = SARIMAX(anomalia_objetivo[en_train], exog=exogenas[en_train], order=mejor_orden, trend="n")

    # === PASO 3: ENTRENAMIENTO ===
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        modelo_entrenado = modelo.fit(disp=False)

    # === PASO 4: PREDICCIÓN E INVERSIÓN DE ESCALA ===
    # [JUSTIFICACIÓN]: aplicamos los MISMOS parámetros a la serie completa (sin reentrenar)
    # y, desde cada día de emisión t, pedimos el pronóstico dinámico de t+1 a t+15: cada
    # paso usa sus propias predicciones, nunca datos reales posteriores a t.
    modelo_serie_completa = modelo_entrenado.apply(anomalia_objetivo, exog=exogenas)

    tablas_por_horizonte = {}
    origenes = set()
    for horizonte in HORIZONTES:
        datos = preparar_datos(niveles, observados, estacion, horizonte)
        tablas_por_horizonte[horizonte] = datos
        origenes.update(datos["tabla_val"].index)
        origenes.update(datos["tabla_test"].index)

    maximo_horizonte = max(HORIZONTES)
    ultima_posicion = len(anomalia_objetivo) - 1
    pronosticos = {}
    for origen in sorted(origenes):
        posicion = anomalia_objetivo.index.get_loc(origen)
        # Al final de la serie no hay 15 días por delante; esos pasos nunca se evalúan
        fin = min(posicion + maximo_horizonte, ultima_posicion)
        prediccion = modelo_serie_completa.get_prediction(start=posicion + 1, end=fin, dynamic=0)
        valores = np.full(maximo_horizonte, np.nan)
        valores[:fin - posicion] = prediccion.predicted_mean.to_numpy()
        pronosticos[origen] = valores
    pronosticos = pd.DataFrame.from_dict(pronosticos, orient="index", columns=range(1, maximo_horizonte + 1))

    # Inversión: nivel(t + h) = climatología(t + h) + anomalía pronosticada
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        datos = tablas_por_horizonte[horizonte]
        for conjunto in ["val", "test"]:
            tabla = datos["tabla_" + conjunto]
            anomalia_predicha = pronosticos.loc[tabla.index, horizonte].to_numpy()
            fechas_objetivo = pd.DatetimeIndex(tabla["fecha_objetivo"])
            nivel_predicho = evaluar_climatologia(climatologias[estacion], fechas_objetivo) + anomalia_predicha
            resultado = armar_predicciones(tabla, nivel_predicho, estacion, "SARIMAX", horizonte)
            if conjunto == "val":
                predicciones_val.append(resultado)
            else:
                predicciones_test.append(resultado)

    return evaluar_y_guardar(estacion, "SARIMAX", pd.concat(predicciones_val), pd.concat(predicciones_test))


def ejecutar() -> None:
    """Entrena y evalúa SARIMAX en todas las estaciones objetivo."""
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_sarimax(niveles, observados, estacion)


if __name__ == "__main__":
    ejecutar()
