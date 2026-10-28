"""Google TimesFM (Time-series Foundation Model de Google Research) para pronóstico zero-shot.

TimesFM es el modelo fundacional de series temporales lanzado por Google en 2024.
Está basado en una arquitectura Transformer decoder-only con procesamiento por parches (patches)
preentrenada sobre más de 100 mil millones de puntos temporales reales y sintéticos.

Para pronosticar de 1 a 15 días:
- Se extrae una ventana de contexto de niveles observados en la estación hasta el día de emisión t.
- TimesFM procesa la ventana y genera de forma directa (zero-shot) el pronóstico multi-horizonte
  a t+1, t+2, ..., t+15 días.
- No requiere entrenamiento supervisado de parámetros locales con gradientes: transfiere el
  conocimiento de dinámica temporal adquirido en su corpus fundacional.

Ejecutar:  python -m src.models.google_model
"""
import numpy as np
import pandas as pd
from timesfm import ForecastConfig
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

from src.configuracion import ESTACIONES_OBJETIVO, HORIZONTES, SEMILLA
from src.evaluate import armar_predicciones, evaluar_y_guardar
from src.preprocessing import cargar_datos, preparar_datos

# Hiperparámetros de inferencia de TimesFM
LONGITUD_CONTEXTO = 128  # días de historia previa de nivel que ve el Transformer
MAXIMO_HORIZONTE = 15    # pronóstico simultáneo a 1..15 días


def ejecutar_timesfm(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    """Ejecuta inferencia zero-shot con Google TimesFM para una estación en todos los horizontes.

    Returns:
        Métricas de prueba por horizonte (ver ``evaluar_y_guardar``).
    """
    # === PASO 1: PREPARACIÓN DE DATOS ===
    # [JUSTIFICACIÓN]: TimesFM recibe ventanas 1D con la historia continua de niveles
    # observados hasta el día de emisión t. Para cada día de validación, prueba y futuro
    # tomamos los últimos LONGITUD_CONTEXTO días de la estación objetivo.
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

    # === PASO 2: DEFINICIÓN DEL MODELO ===
    modelo = TimesFM_2p5_200M_torch(torch_compile=False)
    configuracion = ForecastConfig(
        max_context=LONGITUD_CONTEXTO,
        max_horizon=MAXIMO_HORIZONTE,
        per_core_batch_size=32,
        normalize_inputs=True,
    )
    modelo.compile(configuracion)

    # === PASO 3: ENTRENAMIENTO / CARGA DEL MODELO FUNDACIONAL ===
    # [JUSTIFICACIÓN]: TimesFM es un modelo fundacional zero-shot preentrenado por Google Research
    # en más de 100 mil millones de puntos temporales. No requiere optimización supervisada de
    # pesos con los datos de train: opera de forma zero-shot directamente sobre las secuencias.
    # Los pesos de la arquitectura procesan la serie preservando la memoria temporal y los ciclos.

    # === PASO 4: PREDICCIÓN E INVERSIÓN DE ESCALA ===
    # [JUSTIFICACIÓN]: el modelo pronostica directamente los 15 horizontes a la vez en metros.
    # No requiere escalado ni invertir escala porque opera en la magnitud física real del río.
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
    """Entrena y evalúa Google TimesFM en todas las estaciones objetivo."""
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_timesfm(niveles, observados, estacion)


if __name__ == "__main__":
    ejecutar()
