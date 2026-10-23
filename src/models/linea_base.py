"""Modelos de referencia (líneas base): lo mínimo que cualquier modelo debe superar.

- Persistencia: "dentro de h días el río estará igual que hoy".
- Climatología armónica: "el río estará en su nivel típico de esa fecha".
- Anomalía amortiguada: "el nivel típico de esa fecha más una fracción de la
  desviación de hoy", porque las anomalías se van apagando con el tiempo.

Un modelo de IA solo aporta valor si mejora claramente estas referencias.

Ejecutar:  python -m src.models.linea_base
"""
import numpy as np
import pandas as pd

from src.configuracion import ESTACIONES_OBJETIVO, HORIZONTES
from src.evaluate import armar_predicciones, evaluar_y_guardar
from src.preprocessing import cambio_a_nivel, cargar_datos, preparar_datos


def ejecutar_persistencia(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    """Persistencia: ŷ(t + h) = y(t)."""
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === PASO 1: PREPARACIÓN DE DATOS ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)

        # === PASO 2: DEFINICIÓN DEL MODELO ===
        # El cambio pronosticado es siempre cero.

        # === PASO 3: ENTRENAMIENTO ===
        # [JUSTIFICACIÓN]: la persistencia no tiene parámetros, no hay nada que entrenar.

        # === PASO 4: PREDICCIÓN (YA EN METROS) ===
        for conjunto in ["val", "test"]:
            tabla = datos["tabla_" + conjunto]
            cambio_predicho = np.zeros(len(tabla))
            nivel_predicho = cambio_a_nivel(tabla["nivel_actual"], cambio_predicho)
            resultado = armar_predicciones(tabla, nivel_predicho, estacion, "Persistencia", horizonte)
            if conjunto == "val":
                predicciones_val.append(resultado)
            else:
                predicciones_test.append(resultado)

    return evaluar_y_guardar(estacion, "Persistencia", pd.concat(predicciones_val), pd.concat(predicciones_test))


def ejecutar_climatologia(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    """Climatología armónica: ŷ(t + h) = climatología(t + h)."""
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === PASO 1: PREPARACIÓN DE DATOS ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)
        columna_anomalia = estacion + "_anomalia"

        # === PASO 2: DEFINICIÓN DEL MODELO ===
        # climatología(t + h) - y(t) = [climatología(t + h) - climatología(t)] - anomalía(t)

        # === PASO 3: ENTRENAMIENTO ===
        # [JUSTIFICACIÓN]: la climatología ya se ajustó con train dentro de preparar_datos
        # (regresión armónica con K = 2); aquí solo se usa.

        # === PASO 4: PREDICCIÓN (YA EN METROS) ===
        for conjunto in ["val", "test"]:
            tabla = datos["tabla_" + conjunto]
            cambio_predicho = tabla["climatologia_cambio_esperado"] - tabla[columna_anomalia]
            nivel_predicho = cambio_a_nivel(tabla["nivel_actual"], cambio_predicho)
            resultado = armar_predicciones(tabla, nivel_predicho, estacion, "Climatologia", horizonte)
            if conjunto == "val":
                predicciones_val.append(resultado)
            else:
                predicciones_test.append(resultado)

    return evaluar_y_guardar(estacion, "Climatologia", pd.concat(predicciones_val), pd.concat(predicciones_test))


def ejecutar_anomalia_amortiguada(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    """Anomalía amortiguada: ŷ(t + h) = climatología(t + h) + φ_h · anomalía(t)."""
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === PASO 1: PREPARACIÓN DE DATOS ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)
        columna_anomalia = estacion + "_anomalia"
        tabla_train = datos["tabla_train"]

        # === PASO 2: DEFINICIÓN DEL MODELO ===
        # La anomalía de dentro de h días es una fracción φ_h de la de hoy (0 < φ_h < 1).

        # === PASO 3: ENTRENAMIENTO ===
        # φ_h es la pendiente (sin intercepto) entre la anomalía de hoy y la de t + h en train.
        anomalia_hoy = tabla_train[columna_anomalia]
        anomalia_futura = anomalia_hoy + tabla_train["cambio"] - tabla_train["climatologia_cambio_esperado"]
        phi = float((anomalia_hoy * anomalia_futura).sum() / (anomalia_hoy ** 2).sum())

        # === PASO 4: PREDICCIÓN (YA EN METROS) ===
        for conjunto in ["val", "test"]:
            tabla = datos["tabla_" + conjunto]
            cambio_predicho = tabla["climatologia_cambio_esperado"] - (1.0 - phi) * tabla[columna_anomalia]
            nivel_predicho = cambio_a_nivel(tabla["nivel_actual"], cambio_predicho)
            resultado = armar_predicciones(tabla, nivel_predicho, estacion, "AnomaliaAmortiguada", horizonte)
            if conjunto == "val":
                predicciones_val.append(resultado)
            else:
                predicciones_test.append(resultado)

    return evaluar_y_guardar(estacion, "AnomaliaAmortiguada", pd.concat(predicciones_val),
                             pd.concat(predicciones_test))


def ejecutar() -> None:
    """Evalúa las tres líneas base en todas las estaciones objetivo."""
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_persistencia(niveles, observados, estacion)
        ejecutar_climatologia(niveles, observados, estacion)
        ejecutar_anomalia_amortiguada(niveles, observados, estacion)


if __name__ == "__main__":
    ejecutar()
