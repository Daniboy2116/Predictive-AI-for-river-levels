import lightgbm
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.configuracion import ESTACIONES_OBJETIVO, HORIZONTES, SEMILLA
from src.evaluate import armar_predicciones, evaluar_y_guardar
from src.preprocessing import cambio_a_nivel, cargar_datos, preparar_datos


def evaluar_modelo_arbol(modelo, datos: dict, estacion: str, nombre_modelo: str, horizonte: int,
                         predicciones_val: list, predicciones_test: list) -> None:
    for conjunto in ["val", "test", "futuro"]:
        cambio_predicho = modelo.predict(datos["x_" + conjunto])
        tabla = datos["tabla_" + conjunto]
        nivel_predicho = cambio_a_nivel(tabla["nivel_actual"], cambio_predicho)
        resultado = armar_predicciones(tabla, nivel_predicho, estacion, nombre_modelo, horizonte)
        if conjunto == "val":
            predicciones_val.append(resultado)
        else:
            predicciones_test.append(resultado)


def ejecutar_random_forest(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === Preparación de datos ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)

        # === Definición del modelo ===
        modelo = RandomForestRegressor(
            n_estimators=200,
            max_features=0.33,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=SEMILLA,
        )

        # === Entrenamiento ===
        modelo.fit(datos["x_train"], datos["y_train"])

        # === Evaluación ===
        evaluar_modelo_arbol(modelo, datos, estacion, "RandomForest", horizonte, predicciones_val, predicciones_test)

    return evaluar_y_guardar(estacion, "RandomForest", pd.concat(predicciones_val), pd.concat(predicciones_test))


def ejecutar_xgboost(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === Preparación de datos ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)

        # === Definición del modelo ===
        modelo = XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:absoluteerror",
            early_stopping_rounds=100,
            random_state=SEMILLA,
            n_jobs=-1,
        )

        # === Entrenamiento ===
        modelo.fit(datos["x_train"], datos["y_train"], eval_set=[(datos["x_val"], datos["y_val"])], verbose=False)

        # === Evaluación ===
        evaluar_modelo_arbol(modelo, datos, estacion, "XGBoost", horizonte, predicciones_val, predicciones_test)

    return evaluar_y_guardar(estacion, "XGBoost", pd.concat(predicciones_val), pd.concat(predicciones_test))


def ejecutar_lightgbm(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === Preparación de datos ===
        datos = preparar_datos(niveles, observados, estacion, horizonte)

        # === Definición del modelo ===
        modelo = LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            objective="l1",
            random_state=SEMILLA,
            n_jobs=-1,
            verbose=-1,
        )

        # === Entrenamiento ===
        parada_temprana = lightgbm.early_stopping(stopping_rounds=150, verbose=False)
        modelo.fit(datos["x_train"], datos["y_train"], eval_X=datos["x_val"], eval_y=datos["y_val"],
                   callbacks=[parada_temprana])

        # === Evaluación ===
        evaluar_modelo_arbol(modelo, datos, estacion, "LightGBM", horizonte, predicciones_val, predicciones_test)

    return evaluar_y_guardar(estacion, "LightGBM", pd.concat(predicciones_val), pd.concat(predicciones_test))


def ejecutar() -> None:
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_random_forest(niveles, observados, estacion)
        ejecutar_xgboost(niveles, observados, estacion)
        ejecutar_lightgbm(niveles, observados, estacion)


if __name__ == "__main__":
    ejecutar()
