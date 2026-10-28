import numpy as np
import pandas as pd
from timesfm import ForecastConfig
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

from src.configuracion import ESTACIONES_OBJETIVO, HORIZONTES
from src.evaluate import armar_predicciones, evaluar_y_guardar
from src.preprocessing import cargar_datos, preparar_datos

LONGITUD_CONTEXTO = 128
MAXIMO_HORIZONTE = 15


def ejecutar_timesfm(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    # === Preparación de datos ===
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
        inicio = max(0, posicion - LONGITUD_CONTEXTO + 1)
        serie_contexto = niveles[estacion].iloc[inicio:posicion + 1].to_numpy(dtype=np.float32)
        ventanas_contexto.append(serie_contexto)

    # === Definición del modelo ===
    modelo = TimesFM_2p5_200M_torch(torch_compile=False)
    configuracion = ForecastConfig(
        max_context=LONGITUD_CONTEXTO,
        max_horizon=MAXIMO_HORIZONTE,
        per_core_batch_size=32,
        normalize_inputs=True,
    )
    modelo.compile(configuracion)

    # === Entrenamiento ===
    pass

    # === Evaluación ===
    pronosticos_matriz, _ = modelo.forecast(horizon=MAXIMO_HORIZONTE, inputs=ventanas_contexto)
    pronosticos_df = pd.DataFrame(
        pronosticos_matriz,
        index=dias_emision_ordenados,
        columns=range(1, MAXIMO_HORIZONTE + 1),
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
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_timesfm(niveles, observados, estacion)


if __name__ == "__main__":
    ejecutar()
