"""Contribución de cada grupo de variables (análisis de ablación, Figura 8 del PDF).

Se reentrena el modelo quitando un grupo de variables cada vez y se mide cuánto
empeora el MAE a 15 días en prueba. Cuanto más empeora, más información aporta
ese grupo.

Grupos:
- ciclicas: calendario y cambio climatológico esperado (7 variables).
- propias: rezagos, tendencias, medias y anomalía de la estación objetivo (18).
- otras_estaciones: lo mismo para las otras tres estaciones (54).

Resultado: ``outputs/contribucion_variables.json`` ->
``{estacion: {modelo: {"Modelo completo": MAE, "Sin ciclicas": MAE, ...}}}`` (en metros).

Ejecutar:  python -m src.contribucion
"""
import json

import lightgbm
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from src.configuracion import CARPETA_SALIDA, ESTACIONES_OBJETIVO, HORIZONTE_PRINCIPAL, SEMILLA
from src.evaluate import calcular_metricas
from src.preprocessing import cambio_a_nivel, cargar_datos, grupos_de_variables, preparar_datos

RUTA_CONTRIBUCION = CARPETA_SALIDA / "contribucion_variables.json"
NOMBRES_CONFIGURACION = {
    "completo": "Modelo completo",
    "otras_estaciones": "Sin otras estaciones",
    "ciclicas": "Sin cíclicas",
    "propias": "Sin variables propias",
}


def crear_modelo(nombre_modelo: str):
    """Devuelve un modelo nuevo (mismos hiperparámetros que en lineal_model y tree_models)."""
    if nombre_modelo == "Ridge":
        return Ridge(alpha=10.0)
    return LGBMRegressor(n_estimators=3000, learning_rate=0.02, num_leaves=31, min_child_samples=40,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, objective="l1",
                         random_state=SEMILLA, n_jobs=-1, verbose=-1)


def mae_sin_grupo(datos: dict, variables: list, nombre_modelo: str) -> float:
    """Entrena con las ``variables`` indicadas y devuelve el MAE de prueba (días observados)."""
    modelo = crear_modelo(nombre_modelo)
    if nombre_modelo == "LightGBM":
        parada_temprana = lightgbm.early_stopping(stopping_rounds=150, verbose=False)
        modelo.fit(datos["x_train"][variables], datos["y_train"], eval_X=datos["x_val"][variables],
                   eval_y=datos["y_val"], callbacks=[parada_temprana])
    else:
        modelo.fit(datos["x_train"][variables], datos["y_train"])

    tabla = datos["tabla_test"]
    nivel_predicho = cambio_a_nivel(tabla["nivel_actual"], modelo.predict(datos["x_test"][variables]))
    observados = tabla["observado"].to_numpy()
    metricas = calcular_metricas(tabla["nivel_objetivo"].to_numpy()[observados], nivel_predicho[observados])
    return metricas["MAE"]


def ejecutar() -> None:
    """Calcula la contribución de los grupos para Ridge y LightGBM a 15 días."""
    niveles, observados = cargar_datos()
    resultados = {}
    for estacion in ESTACIONES_OBJETIVO:
        datos = preparar_datos(niveles, observados, estacion, HORIZONTE_PRINCIPAL)
        grupos = grupos_de_variables(datos["variables"], estacion)
        resultados[estacion] = {}
        for nombre_modelo in ["Ridge", "LightGBM"]:
            resultados[estacion][nombre_modelo] = {}
            for configuracion, etiqueta in NOMBRES_CONFIGURACION.items():
                variables = []
                for variable in datos["variables"]:
                    if configuracion == "completo" or variable not in grupos[configuracion]:
                        variables.append(variable)
                mae = mae_sin_grupo(datos, variables, nombre_modelo)
                resultados[estacion][nombre_modelo][etiqueta] = round(mae, 6)
            print(estacion.ljust(15), nombre_modelo.ljust(9), "MAE h=15 (cm):",
                  pd.Series(resultados[estacion][nombre_modelo]).mul(100).round(1).to_dict(), flush=True)

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    with open(RUTA_CONTRIBUCION, "w", encoding="utf-8") as archivo:
        json.dump(resultados, archivo, indent=2, ensure_ascii=False)


def cargar_contribucion() -> dict:
    """Lee el resultado del análisis (diccionario vacío si todavía no existe)."""
    if not RUTA_CONTRIBUCION.exists():
        return {}
    with open(RUTA_CONTRIBUCION, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


if __name__ == "__main__":
    ejecutar()
