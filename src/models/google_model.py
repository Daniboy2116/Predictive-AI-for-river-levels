"""Modelo de Google: Prophet (tendencia + estacionalidad anual + nivel del día anterior).

Prophet descompone la serie en tendencia y estacionalidad. Además le añadimos como
"regresor externo" el nivel real del día anterior (``lag_1``), para que pueda
predecir a 1 día vista igual que el resto de modelos.

Ejecutar:  python -m src.models.google_model
"""
import logging

import pandas as pd
from prophet import Prophet

from src.evaluate import evaluar_y_guardar
from src.preprocessing import ESTACIONES, cargar_datos, preparar_estacion

# Prophet (y su motor cmdstanpy) imprimen muchos mensajes; mostramos solo avisos importantes
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def ejecutar_prophet(df: pd.DataFrame, estacion: str) -> dict:
    """Entrena y evalúa Prophet para una estación.

    Args:
        df: DataFrame completo devuelto por ``cargar_datos``.
        estacion: nombre de la estación.

    Returns:
        Diccionario con las métricas (RMSE, MAE, R2) en metros.
    """
    # === PASO 1: PREPARACIÓN DE DATOS ===
    # [JUSTIFICACIÓN]: Prophet no acepta matrices X / y. Exige un DataFrame con dos
    # columnas obligatorias: "ds" (fecha) e "y" (valor a predecir), más las columnas
    # de regresores extra. Trabaja en metros, por lo que no se escala.
    datos = preparar_estacion(df, estacion)
    serie = datos["serie"]

    tabla = pd.DataFrame()
    tabla["ds"] = serie.index
    tabla["y"] = serie.to_numpy()
    tabla["lag_1"] = serie.shift(1).to_numpy()   # nivel real del día anterior
    tabla = tabla.dropna()                        # el primer día no tiene lag_1

    # Prophet no usa early stopping, así que entrena con train + val juntos
    tabla_entrenamiento = tabla[tabla["ds"] <= datos["fecha_fin_val"]]
    tabla_test = tabla[tabla["ds"].isin(datos["fechas_test"])]

    # === PASO 2: DEFINICIÓN DEL MODELO ===
    modelo = Prophet(
        yearly_seasonality=True,        # ciclo anual de crecida / estiaje
        weekly_seasonality=False,       # el río no depende del día de la semana
        daily_seasonality=False,        # tenemos un dato por día, no por hora
        changepoint_prior_scale=0.05,   # flexibilidad de la tendencia
    )
    modelo.add_regressor("lag_1")

    # === PASO 3: ENTRENAMIENTO ===
    modelo.fit(tabla_entrenamiento)

    # === PASO 4: PREDICCIÓN (YA EN METROS) ===
    # [JUSTIFICACIÓN]: no hay que invertir ninguna escala porque Prophet se entrenó en metros.
    prediccion = modelo.predict(tabla_test[["ds", "lag_1"]])
    y_pred = prediccion["yhat"].to_numpy()   # "yhat" es la predicción central de Prophet
    y_real = datos["y_test_metros"]

    metricas = evaluar_y_guardar(estacion, "Prophet", datos["fechas_test"], y_real, y_pred)
    return metricas


def ejecutar() -> None:
    """Entrena y evalúa Prophet en todas las estaciones."""
    df = cargar_datos()
    for estacion in ESTACIONES:
        ejecutar_prophet(df, estacion)


if __name__ == "__main__":
    ejecutar()
