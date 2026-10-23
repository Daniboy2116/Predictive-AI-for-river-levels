"""Regresión Ridge: modelo lineal con penalización L2 sobre las 79 variables.

Predice el cambio y(t + h) - y(t) como una suma ponderada de las variables. La
penalización (alpha) encoge los pesos para que el modelo no se sobreajuste
cuando hay variables muy correlacionadas (rezagos vecinos, estaciones vecinas).

Es el modelo "sencillo pero fuerte": si un modelo complejo no le gana, la
complejidad no se justifica.

Ejecutar:  python -m src.models.lineal_model
"""
import pandas as pd
from sklearn.linear_model import Ridge

from src.configuracion import ESTACIONES_OBJETIVO, HORIZONTES
from src.evaluate import armar_predicciones, calcular_metricas, evaluar_y_guardar
from src.preprocessing import cambio_a_nivel, cargar_datos, preparar_datos

# Valores de alpha que se prueban; se elige el de menor MAE en validación
VALORES_ALPHA = [0.1, 1.0, 10.0, 100.0, 1000.0]


def ejecutar_ridge(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    """Entrena y evalúa Ridge para una estación en todos los horizontes.

    Returns:
        Métricas de prueba por horizonte (ver ``evaluar_y_guardar``).
    """
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === PASO 1: PREPARACIÓN DE DATOS ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)

        # === PASO 2: DEFINICIÓN DEL MODELO ===
        # [JUSTIFICACIÓN]: probamos varios alpha y nos quedamos con el que mejor
        # funciona en validación (búsqueda de hiperparámetros sin tocar prueba).
        mejor_alpha = VALORES_ALPHA[0]
        mejor_mae = float("inf")
        for alpha in VALORES_ALPHA:
            candidato = Ridge(alpha=alpha)
            candidato.fit(datos["x_train"], datos["y_train"])
            mae_val = calcular_metricas(datos["y_val"], candidato.predict(datos["x_val"]))["MAE"]
            if mae_val < mejor_mae:
                mejor_mae = mae_val
                mejor_alpha = alpha
        modelo = Ridge(alpha=mejor_alpha)

        # === PASO 3: ENTRENAMIENTO ===
        modelo.fit(datos["x_train"], datos["y_train"])

        # === PASO 4: PREDICCIÓN E INVERSIÓN DE ESCALA ===
        # El modelo predice el cambio en metros; el nivel es y(t) + cambio.
        for conjunto in ["val", "test"]:
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
    """Entrena y evalúa Ridge en todas las estaciones objetivo."""
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_ridge(niveles, observados, estacion)


if __name__ == "__main__":
    ejecutar()
