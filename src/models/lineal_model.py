import pandas as pd
from sklearn.linear_model import Ridge

from src.configuracion import ESTACIONES_OBJETIVO, HORIZONTES
from src.evaluate import armar_predicciones, calcular_metricas, evaluar_y_guardar
from src.preprocessing import cambio_a_nivel, cargar_datos, preparar_datos

VALORES_ALPHA = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def seleccionar_mejor_alpha(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> float:
    mejor_alpha = VALORES_ALPHA[0]
    mejor_mae = float("inf")
    for alpha in VALORES_ALPHA:
        candidato = Ridge(alpha=alpha)
        candidato.fit(x_train, y_train)
        mae_val = calcular_metricas(y_val, candidato.predict(x_val))["MAE"]
        if mae_val < mejor_mae:
            mejor_mae = mae_val
            mejor_alpha = alpha
    return mejor_alpha


def ejecutar_ridge(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === Preparación de datos ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)

        # === Definición del modelo ===
        mejor_alpha = seleccionar_mejor_alpha(datos["x_train"], datos["y_train"], datos["x_val"], datos["y_val"])
        modelo = Ridge(alpha=mejor_alpha)

        # === Entrenamiento ===
        modelo.fit(datos["x_train"], datos["y_train"])

        # === Evaluación ===
        for conjunto in ["val", "test", "futuro"]:
            cambio_predicho = modelo.predict(datos["x_" + conjunto])
            tabla = datos["tabla_" + conjunto]
            nivel_predicho = cambio_a_nivel(tabla["nivel_actual"], cambio_predicho)
            resultado = armar_predicciones(tabla, nivel_predicho, estacion, "Ridge", horizonte)
            if conjunto == "val":
                predicciones_val.append(resultado)
            else:
                predicciones_test.append(resultado)

    return evaluar_y_guardar(estacion, "Ridge", pd.concat(predicciones_val), pd.concat(predicciones_test))


def ejecutar() -> None:
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_ridge(niveles, observados, estacion)


if __name__ == "__main__":
    ejecutar()
