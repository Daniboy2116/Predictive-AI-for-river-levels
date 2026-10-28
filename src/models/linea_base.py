import numpy as np
import pandas as pd

from src.configuracion import ESTACIONES_OBJETIVO, HORIZONTES
from src.evaluate import armar_predicciones, evaluar_y_guardar
from src.preprocessing import cambio_a_nivel, cargar_datos, preparar_datos


def _generar_predicciones(datos: dict, cambio_predicho_dict: dict, estacion: str, nombre_modelo: str, horizonte: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_val, pred_test = [], []
    for conjunto in ["val", "test", "futuro"]:
        tabla = datos["tabla_" + conjunto]
        cambio = cambio_predicho_dict[conjunto]
        nivel = cambio_a_nivel(tabla["nivel_actual"], cambio)
        res = armar_predicciones(tabla, nivel, estacion, nombre_modelo, horizonte)
        if conjunto == "val":
            pred_val.append(res)
        else:
            pred_test.append(res)
    return pd.concat(pred_val), pd.concat(pred_test)


def ejecutar_persistencia(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    val_totales, test_totales = [], []
    for horizonte in HORIZONTES:
        # === Preparación de datos ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)

        # === Definición del modelo ===
        pass

        # === Entrenamiento ===
        pass

        # === Evaluación ===
        cambios = {c: np.zeros(len(datos["tabla_" + c])) for c in ["val", "test", "futuro"]}
        val_h, test_h = _generar_predicciones(datos, cambios, estacion, "Persistencia", horizonte)
        val_totales.append(val_h)
        test_totales.append(test_h)

    return evaluar_y_guardar(estacion, "Persistencia", pd.concat(val_totales), pd.concat(test_totales))


def ejecutar_climatologia(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    val_totales, test_totales = [], []
    for horizonte in HORIZONTES:
        # === Preparación de datos ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)
        col_anomalia = estacion + "_anomalia"

        # === Definición del modelo ===
        pass

        # === Entrenamiento ===
        pass

        # === Evaluación ===
        cambios = {}
        for c in ["val", "test", "futuro"]:
            tabla = datos["tabla_" + c]
            cambios[c] = tabla["climatologia_cambio_esperado"] - tabla[col_anomalia]

        val_h, test_h = _generar_predicciones(datos, cambios, estacion, "Climatologia", horizonte)
        val_totales.append(val_h)
        test_totales.append(test_h)

    return evaluar_y_guardar(estacion, "Climatologia", pd.concat(val_totales), pd.concat(test_totales))


def ejecutar_anomalia_amortiguada(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    val_totales, test_totales = [], []
    for horizonte in HORIZONTES:
        # === Preparación de datos ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)
        col_anomalia = estacion + "_anomalia"
        tabla_train = datos["tabla_train"]

        # === Definición del modelo ===
        pass

        # === Entrenamiento ===
        anomalia_hoy = tabla_train[col_anomalia]
        anomalia_futura = anomalia_hoy + tabla_train["cambio"] - tabla_train["climatologia_cambio_esperado"]
        phi = float((anomalia_hoy * anomalia_futura).sum() / (anomalia_hoy ** 2).sum())

        # === Evaluación ===
        cambios = {}
        for c in ["val", "test", "futuro"]:
            tabla = datos["tabla_" + c]
            cambios[c] = tabla["climatologia_cambio_esperado"] - (1.0 - phi) * tabla[col_anomalia]

        val_h, test_h = _generar_predicciones(datos, cambios, estacion, "AnomaliaAmortiguada", horizonte)
        val_totales.append(val_h)
        test_totales.append(test_h)

    return evaluar_y_guardar(estacion, "AnomaliaAmortiguada", pd.concat(val_totales), pd.concat(test_totales))


def ejecutar() -> None:
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_persistencia(niveles, observados, estacion)
        ejecutar_climatologia(niveles, observados, estacion)
        ejecutar_anomalia_amortiguada(niveles, observados, estacion)


if __name__ == "__main__":
    ejecutar()
