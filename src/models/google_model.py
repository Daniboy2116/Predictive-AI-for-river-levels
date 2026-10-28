import logging

import numpy as np
import pandas as pd
from prophet import Prophet
from timesfm import ForecastConfig
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

from src.configuracion import ESTACIONES, ESTACIONES_OBJETIVO, HORIZONTES
from src.evaluate import armar_predicciones, evaluar_y_guardar
from src.preprocessing import cargar_datos, preparar_datos

registro_cmdstanpy = logging.getLogger("cmdstanpy")
registro_cmdstanpy.addHandler(logging.NullHandler())
registro_cmdstanpy.setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

REGRESORES_PROPHET = []
for _estacion in ESTACIONES:
    REGRESORES_PROPHET.append(_estacion + "_rezago_0")
    REGRESORES_PROPHET.append(_estacion + "_tendencia_7")

LONGITUD_CONTEXTO_TIMESFM = 128
MAXIMO_HORIZONTE_TIMESFM = 15


def tabla_prophet(tabla: pd.DataFrame) -> pd.DataFrame:
    """Convierte una tabla de preparar_datos al formato requerido por Prophet (ds, y + regresores)."""
    formato = pd.DataFrame()
    formato["ds"] = pd.to_datetime(tabla["fecha_objetivo"]).to_numpy()
    formato["y"] = tabla["nivel_objetivo"].to_numpy()
    for regresor in REGRESORES_PROPHET:
        formato[regresor] = tabla[regresor].to_numpy()
    return formato


def ejecutar_prophet(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    """Entrena y evalúa Prophet con regresores aguas arriba para una estación en todos los horizontes.

    Returns:
        Métricas de prueba por horizonte (ver evaluar_y_guardar).
    """
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === Preparación de datos ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)
        entrenamiento = tabla_prophet(datos["tabla_train"])

        # === Definición del modelo ===
        modelo = Prophet(
            growth="flat",
            yearly_seasonality=10,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        for regresor in REGRESORES_PROPHET:
            modelo.add_regressor(regresor)

        # === Entrenamiento ===
        # [JUSTIFICACIÓN]: se entrena exclusivamente con datos de train (2009-2016).
        modelo.fit(entrenamiento)

        # === Evaluación ===
        for conjunto in ["val", "test", "futuro"]:
            tabla = datos["tabla_" + conjunto]
            prediccion = modelo.predict(tabla_prophet(tabla).drop(columns=["y"]))
            nivel_predicho = prediccion["yhat"].to_numpy()
            resultado = armar_predicciones(tabla, nivel_predicho, estacion, "Prophet", horizonte)
            if conjunto == "val":
                predicciones_val.append(resultado)
            else:
                predicciones_test.append(resultado)

    return evaluar_y_guardar(
        estacion,
        "Prophet",
        pd.concat(predicciones_val),
        pd.concat(predicciones_test),
    )


def ejecutar_timesfm(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    """Ejecuta inferencia zero-shot con Google TimesFM para una estación en todos los horizontes.

    Returns:
        Métricas de prueba por horizonte (ver evaluar_y_guardar).
    """
    tablas_por_horizonte = {}
    dias_emision_unicos = set()
    for horizonte in HORIZONTES:
        datos = preparar_datos(niveles, observados, estacion, horizonte)
        tablas_por_horizonte[horizonte] = datos
        dias_emision_unicos.update(datos["tabla_val"].index)
        dias_emision_unicos.update(datos["tabla_test"].index)
        dias_emision_unicos.update(datos["tabla_futuro"].index)

    dias_emision_ordenados = sorted(dias_emision_unicos)
    ventanas_contexto = []
    for dia in dias_emision_ordenados:
        posicion = niveles.index.get_loc(dia)
        inicio = max(0, posicion - LONGITUD_CONTEXTO_TIMESFM + 1)
        serie_contexto = niveles[estacion].iloc[inicio:posicion + 1].to_numpy(dtype=np.float32)
        ventanas_contexto.append(serie_contexto)

    modelo = TimesFM_2p5_200M_torch(torch_compile=False)
    configuracion = ForecastConfig(
        max_context=LONGITUD_CONTEXTO_TIMESFM,
        max_horizon=MAXIMO_HORIZONTE_TIMESFM,
        per_core_batch_size=32,
        normalize_inputs=True,
    )
    modelo.compile(configuracion)

    pronosticos_matriz, _ = modelo.forecast(horizon=MAXIMO_HORIZONTE_TIMESFM, inputs=ventanas_contexto)
    pronosticos_df = pd.DataFrame(
        pronosticos_matriz,
        index=dias_emision_ordenados,
        columns=range(1, MAXIMO_HORIZONTE_TIMESFM + 1),
    )

    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        datos = tablas_por_horizonte[horizonte]
        for conjunto in ["val", "test", "futuro"]:
            tabla = datos["tabla_" + conjunto]
            nivel_predicho = pronosticos_df.loc[tabla.index, horizonte].to_numpy()
            resultado = armar_predicciones(tabla, nivel_predicho, estacion, "TimesFM", horizonte)
            if conjunto == "val":
                predicciones_val.append(resultado)
            else:
                predicciones_test.append(resultado)

    return evaluar_y_guardar(
        estacion,
        "TimesFM",
        pd.concat(predicciones_val),
        pd.concat(predicciones_test),
    )


def ejecutar() -> None:
    """Entrena y evalúa Prophet en todas las estaciones objetivo."""
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_prophet(niveles, observados, estacion)


if __name__ == "__main__":
    ejecutar()
