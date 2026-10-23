"""Métricas de evaluación y guardado de resultados.

Archivos generados (siempre en metros, escala original):
- ``outputs/metrics.json``     -> ``{estacion: {modelo: {"RMSE", "MAE", "R2"}}}``
- ``outputs/predictions.csv``  -> columnas ``Fecha, Estacion, Modelo, Real, Prediccion``

Al guardar se conservan los resultados de los otros modelos, así cada script
de ``src/models`` puede ejecutarse por separado sin borrar el trabajo de los demás.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_SALIDA = CARPETA_PROYECTO / "outputs"
RUTA_METRICAS = CARPETA_SALIDA / "metrics.json"
RUTA_PREDICCIONES = CARPETA_SALIDA / "predictions.csv"


def calcular_metricas(y_real, y_pred) -> dict:
    """Calcula RMSE, MAE y R² entre valores reales y predichos (ambos en metros).

    Args:
        y_real: valores observados.
        y_pred: valores predichos por el modelo.

    Returns:
        Diccionario ``{"RMSE": ..., "MAE": ..., "R2": ...}``.
    """
    y_real = np.asarray(y_real, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)

    metricas = {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}
    return metricas


def guardar_metricas(estacion: str, nombre_modelo: str, metricas: dict) -> None:
    """Añade (o reemplaza) las métricas de un modelo/estación en ``metrics.json``."""
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    # 1) Leer el archivo existente (si no existe empezamos con un diccionario vacío)
    if RUTA_METRICAS.exists():
        with open(RUTA_METRICAS, "r", encoding="utf-8") as archivo:
            todas_las_metricas = json.load(archivo)
    else:
        todas_las_metricas = {}

    # 2) Insertar las nuevas métricas redondeadas
    if estacion not in todas_las_metricas:
        todas_las_metricas[estacion] = {}
    todas_las_metricas[estacion][nombre_modelo] = {
        "RMSE": round(metricas["RMSE"], 6),
        "MAE": round(metricas["MAE"], 6),
        "R2": round(metricas["R2"], 6),
    }

    # 3) Escribir el archivo actualizado
    with open(RUTA_METRICAS, "w", encoding="utf-8") as archivo:
        json.dump(todas_las_metricas, archivo, indent=2, ensure_ascii=False)


def guardar_predicciones(estacion: str, nombre_modelo: str, fechas, y_real, y_pred) -> None:
    """Añade (o reemplaza) las predicciones de un modelo/estación en ``predictions.csv``."""
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    nuevas = pd.DataFrame({
        "Fecha": pd.to_datetime(fechas),
        "Estacion": estacion,
        "Modelo": nombre_modelo,
        "Real": np.asarray(y_real, dtype=float).ravel(),
        "Prediccion": np.asarray(y_pred, dtype=float).ravel(),
    })

    if RUTA_PREDICCIONES.exists():
        anteriores = pd.read_csv(RUTA_PREDICCIONES, parse_dates=["Fecha"])
        # Quitamos las filas antiguas de este mismo modelo y estación
        mismas_filas = (anteriores["Estacion"] == estacion) & (anteriores["Modelo"] == nombre_modelo)
        anteriores = anteriores[~mismas_filas]
        nuevas = pd.concat([anteriores, nuevas], ignore_index=True)

    nuevas = nuevas.sort_values(["Estacion", "Modelo", "Fecha"])
    nuevas.to_csv(RUTA_PREDICCIONES, index=False)


def evaluar_y_guardar(estacion: str, nombre_modelo: str, fechas, y_real, y_pred) -> dict:
    """Calcula las métricas, guarda métricas y predicciones, e imprime un resumen.

    Returns:
        Diccionario con las métricas calculadas.
    """
    metricas = calcular_metricas(y_real, y_pred)
    guardar_metricas(estacion, nombre_modelo, metricas)
    guardar_predicciones(estacion, nombre_modelo, fechas, y_real, y_pred)

    print(
        estacion.ljust(15), nombre_modelo.ljust(13),
        "RMSE = %.4f" % metricas["RMSE"],
        "MAE = %.4f" % metricas["MAE"],
        "R2 = %.4f" % metricas["R2"],
        flush=True,
    )
    return metricas


def cargar_metricas() -> pd.DataFrame:
    """Lee ``metrics.json`` como tabla con columnas Estacion, Modelo, RMSE, MAE, R2."""
    if not RUTA_METRICAS.exists():
        return pd.DataFrame(columns=["Estacion", "Modelo", "RMSE", "MAE", "R2"])

    with open(RUTA_METRICAS, "r", encoding="utf-8") as archivo:
        todas_las_metricas = json.load(archivo)

    filas = []
    for estacion in todas_las_metricas:
        for nombre_modelo in todas_las_metricas[estacion]:
            valores = todas_las_metricas[estacion][nombre_modelo]
            filas.append({
                "Estacion": estacion,
                "Modelo": nombre_modelo,
                "RMSE": valores["RMSE"],
                "MAE": valores["MAE"],
                "R2": valores["R2"],
            })
    return pd.DataFrame(filas)


def cargar_predicciones() -> pd.DataFrame:
    """Lee ``predictions.csv`` (devuelve una tabla vacía si todavía no existe)."""
    if not RUTA_PREDICCIONES.exists():
        return pd.DataFrame(columns=["Fecha", "Estacion", "Modelo", "Real", "Prediccion"])
    return pd.read_csv(RUTA_PREDICCIONES, parse_dates=["Fecha"])
