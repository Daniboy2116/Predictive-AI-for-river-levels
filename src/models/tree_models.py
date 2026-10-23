"""Modelos de ensamblado de árboles: Random Forest, XGBoost y LightGBM.

Los tres usan las 79 variables del día t y predicen el cambio y(t + h) - y(t).

- Random Forest (Bagging): muchos árboles independientes; se promedian sus predicciones.
- XGBoost y LightGBM (Boosting): árboles que se construyen uno tras otro corrigiendo
  los errores de los anteriores. LightGBM es el mejor modelo del análisis del PDF.

[JUSTIFICACIÓN] general: los árboles no saben extrapolar (nunca predicen fuera del
rango visto en train). Por eso predecimos el CAMBIO del nivel y no el nivel: un
cambio de +1 m en 15 días ya ocurrió muchas veces en train, aunque el nivel de
partida sea un récord.

Ejecutar:  python -m src.models.tree_models
"""
import lightgbm
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.configuracion import ESTACIONES_OBJETIVO, HORIZONTES, SEMILLA
from src.evaluate import armar_predicciones, evaluar_y_guardar
from src.preprocessing import cambio_a_nivel, cargar_datos, preparar_datos


def guardar_conjunto(modelo, datos: dict, estacion: str, nombre_modelo: str, horizonte: int,
                     predicciones_val: list, predicciones_test: list) -> None:
    """PASO 4 común: predice validación y prueba, reconstruye el nivel y lo añade a las listas."""
    for conjunto in ["val", "test"]:
        cambio_predicho = modelo.predict(datos["x_" + conjunto])
        tabla = datos["tabla_" + conjunto]
        nivel_predicho = cambio_a_nivel(tabla["nivel_actual"], cambio_predicho)
        resultado = armar_predicciones(tabla, nivel_predicho, estacion, nombre_modelo, horizonte)
        if conjunto == "val":
            predicciones_val.append(resultado)
        else:
            predicciones_test.append(resultado)


def ejecutar_random_forest(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    """Entrena y evalúa un Random Forest para una estación en todos los horizontes."""
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === PASO 1: PREPARACIÓN DE DATOS ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)

        # === PASO 2: DEFINICIÓN DEL MODELO ===
        modelo = RandomForestRegressor(
            n_estimators=200,       # número de árboles
            max_features=0.33,      # cada división mira 1/3 de las variables (árboles más distintos)
            min_samples_leaf=5,     # mínimo de muestras en cada hoja
            n_jobs=-1,              # usar todos los núcleos del procesador
            random_state=SEMILLA,
        )

        # === PASO 3: ENTRENAMIENTO ===
        # [JUSTIFICACIÓN]: Random Forest no usa validación porque no tiene "early stopping":
        # entrena todos sus árboles de una vez solo con train.
        modelo.fit(datos["x_train"], datos["y_train"])

        # === PASO 4: PREDICCIÓN E INVERSIÓN DE ESCALA ===
        guardar_conjunto(modelo, datos, estacion, "RandomForest", horizonte, predicciones_val, predicciones_test)

    return evaluar_y_guardar(estacion, "RandomForest", pd.concat(predicciones_val), pd.concat(predicciones_test))


def ejecutar_xgboost(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    """Entrena y evalúa XGBoost para una estación en todos los horizontes."""
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === PASO 1: PREPARACIÓN DE DATOS ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)

        # === PASO 2: DEFINICIÓN DEL MODELO ===
        modelo = XGBRegressor(
            n_estimators=2000,          # máximo de árboles (se detiene antes si no mejora)
            learning_rate=0.03,         # cuánto corrige cada árbol nuevo
            max_depth=6,
            subsample=0.8,              # % de filas que ve cada árbol
            colsample_bytree=0.8,       # % de variables que ve cada árbol
            objective="reg:absoluteerror",  # optimiza directamente el MAE
            early_stopping_rounds=100,  # parar si 100 árboles seguidos no mejoran en val
            random_state=SEMILLA,
            n_jobs=-1,
        )

        # === PASO 3: ENTRENAMIENTO ===
        # [JUSTIFICACIÓN]: XGBoost recibe el conjunto de validación (eval_set) para aplicar
        # "early stopping" y evitar el sobreajuste.
        modelo.fit(datos["x_train"], datos["y_train"], eval_set=[(datos["x_val"], datos["y_val"])], verbose=False)

        # === PASO 4: PREDICCIÓN E INVERSIÓN DE ESCALA ===
        guardar_conjunto(modelo, datos, estacion, "XGBoost", horizonte, predicciones_val, predicciones_test)

    return evaluar_y_guardar(estacion, "XGBoost", pd.concat(predicciones_val), pd.concat(predicciones_test))


def ejecutar_lightgbm(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    """Entrena y evalúa LightGBM para una estación en todos los horizontes."""
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === PASO 1: PREPARACIÓN DE DATOS ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)

        # === PASO 2: DEFINICIÓN DEL MODELO ===
        modelo = LGBMRegressor(
            n_estimators=3000,          # máximo de árboles (se detiene antes si no mejora)
            learning_rate=0.02,
            num_leaves=31,              # complejidad de cada árbol
            min_child_samples=40,       # mínimo de muestras por hoja
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            objective="l1",             # optimiza directamente el MAE
            random_state=SEMILLA,
            n_jobs=-1,
            verbose=-1,
        )

        # === PASO 3: ENTRENAMIENTO ===
        # [JUSTIFICACIÓN]: como XGBoost, usa validación para detenerse ("early stopping").
        parada_temprana = lightgbm.early_stopping(stopping_rounds=150, verbose=False)
        modelo.fit(datos["x_train"], datos["y_train"], eval_X=datos["x_val"], eval_y=datos["y_val"],
                   callbacks=[parada_temprana])

        # === PASO 4: PREDICCIÓN E INVERSIÓN DE ESCALA ===
        guardar_conjunto(modelo, datos, estacion, "LightGBM", horizonte, predicciones_val, predicciones_test)

    return evaluar_y_guardar(estacion, "LightGBM", pd.concat(predicciones_val), pd.concat(predicciones_test))


def ejecutar() -> None:
    """Entrena y evalúa los tres modelos de árboles en todas las estaciones objetivo."""
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_random_forest(niveles, observados, estacion)
        ejecutar_xgboost(niveles, observados, estacion)
        ejecutar_lightgbm(niveles, observados, estacion)


if __name__ == "__main__":
    ejecutar()
