"""Modelos de ensamblado de árboles: Random Forest y XGBoost.

Ambos usan como entradas los niveles escalados de los 7 días anteriores
(lag_1 ... lag_7) y predicen el nivel del día siguiente (t+1).

- Random Forest (Bagging): muchos árboles independientes; se promedian sus predicciones.
- XGBoost (Boosting): árboles que se construyen uno tras otro corrigiendo los errores
  del anterior.

Ejecutar:  python -m src.models.tree_models
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.evaluate import evaluar_y_guardar
from src.preprocessing import ESTACIONES, SEMILLA, cargar_datos, invertir_escala, preparar_estacion


def ejecutar_random_forest(df: pd.DataFrame, estacion: str) -> dict:
    """Entrena y evalúa un Random Forest para una estación.

    Args:
        df: DataFrame completo devuelto por ``cargar_datos``.
        estacion: nombre de la estación.

    Returns:
        Diccionario con las métricas (RMSE, MAE, R2) en metros.
    """
    # === PASO 1: PREPARACIÓN DE DATOS ===
    datos = preparar_estacion(df, estacion)
    x_train = datos["x_train"]
    y_train = datos["y_train"]
    x_test = datos["x_test"]

    # === PASO 2: DEFINICIÓN DEL MODELO ===
    modelo = RandomForestRegressor(
        n_estimators=300,       # número de árboles
        max_depth=12,           # profundidad máxima de cada árbol
        min_samples_leaf=2,     # mínimo de muestras en cada hoja
        n_jobs=-1,              # usar todos los núcleos del procesador
        random_state=SEMILLA,
    )

    # === PASO 3: ENTRENAMIENTO ===
    # [JUSTIFICACIÓN]: Random Forest no usa el conjunto de validación porque no tiene
    # "early stopping": entrena todos sus árboles de una vez solo con train.
    modelo.fit(x_train, y_train)

    # === PASO 4: PREDICCIÓN E INVERSIÓN DE ESCALA ===
    y_pred_escalado = modelo.predict(x_test)
    y_pred = invertir_escala(datos["escalador"], y_pred_escalado)
    y_real = datos["y_test_metros"]

    metricas = evaluar_y_guardar(estacion, "RandomForest", datos["fechas_test"], y_real, y_pred)
    return metricas


def ejecutar_xgboost(df: pd.DataFrame, estacion: str) -> dict:
    """Entrena y evalúa un XGBoost para una estación.

    Args:
        df: DataFrame completo devuelto por ``cargar_datos``.
        estacion: nombre de la estación.

    Returns:
        Diccionario con las métricas (RMSE, MAE, R2) en metros.
    """
    # === PASO 1: PREPARACIÓN DE DATOS ===
    datos = preparar_estacion(df, estacion)
    x_train = datos["x_train"]
    y_train = datos["y_train"]
    x_val = datos["x_val"]
    y_val = datos["y_val"]
    x_test = datos["x_test"]

    # === PASO 2: DEFINICIÓN DEL MODELO ===
    modelo = XGBRegressor(
        n_estimators=1000,          # máximo de árboles (se detiene antes si no mejora)
        learning_rate=0.03,         # cuánto corrige cada árbol nuevo
        max_depth=5,
        subsample=0.9,              # % de filas que ve cada árbol
        colsample_bytree=0.9,       # % de columnas (lags) que ve cada árbol
        early_stopping_rounds=50,   # parar si 50 árboles seguidos no mejoran en val
        random_state=SEMILLA,
        n_jobs=-1,
    )

    # === PASO 3: ENTRENAMIENTO ===
    # [JUSTIFICACIÓN]: a diferencia de Random Forest, XGBoost recibe el conjunto de
    # validación (eval_set) para aplicar "early stopping" y evitar el sobreajuste.
    modelo.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    # === PASO 4: PREDICCIÓN E INVERSIÓN DE ESCALA ===
    y_pred_escalado = modelo.predict(x_test)
    y_pred = invertir_escala(datos["escalador"], y_pred_escalado)
    y_real = datos["y_test_metros"]

    metricas = evaluar_y_guardar(estacion, "XGBoost", datos["fechas_test"], y_real, y_pred)
    return metricas


def ejecutar() -> None:
    """Entrena y evalúa ambos modelos de árboles en todas las estaciones."""
    df = cargar_datos()
    for estacion in ESTACIONES:
        ejecutar_random_forest(df, estacion)
        ejecutar_xgboost(df, estacion)


if __name__ == "__main__":
    ejecutar()
