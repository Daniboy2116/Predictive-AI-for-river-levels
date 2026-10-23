"""Métricas de evaluación y guardado de resultados.

Archivos generados (niveles siempre en metros, escala original):
- ``outputs/predictions.csv``            -> predicciones de PRUEBA (2020-2025) y el pronóstico
  operativo desde el último dato (``Real`` vacío y ``Observado = False``).
- ``outputs/predicciones_validacion.csv`` -> predicciones de VALIDACIÓN (2015-2019), para el ensamble.
  Ambos con columnas ``Fecha, Estacion, Modelo, Horizonte, Real, Prediccion, Observado``
  (``Fecha`` es la fecha objetivo t + h).
- ``outputs/metrics.json`` -> ``{estacion: {modelo: {horizonte: {"MAE", "RMSE", "NSE"}}}}``.

Las métricas se calculan solo con días observados (``Observado == True``).
Al guardar se conservan los resultados de los otros modelos, así cada script de
``src/models`` puede ejecutarse por separado sin borrar el trabajo de los demás.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.configuracion import CARPETA_SALIDA, HORIZONTES_RESUMEN

RUTA_METRICAS = CARPETA_SALIDA / "metrics.json"
RUTA_PREDICCIONES = CARPETA_SALIDA / "predictions.csv"
RUTA_PREDICCIONES_VALIDACION = CARPETA_SALIDA / "predicciones_validacion.csv"
COLUMNAS_PREDICCIONES = ["Fecha", "Estacion", "Modelo", "Horizonte", "Real", "Prediccion", "Observado"]


def calcular_metricas(y_real, y_pred) -> dict:
    """Calcula MAE, RMSE y NSE entre valores reales y predichos (ambos en metros).

    NSE (Nash-Sutcliffe) = 1 - Σ(y - ŷ)² / Σ(y - ȳ)². Es la métrica clásica en
    hidrología: 1 es perfecto y 0 equivale a predecir siempre la media.
    (Es la misma fórmula que el R² de scikit-learn.)

    Returns:
        Diccionario ``{"MAE": ..., "RMSE": ..., "NSE": ...}``.
    """
    y_real = np.asarray(y_real, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    nse = 1.0 - np.sum((y_real - y_pred) ** 2) / np.sum((y_real - y_real.mean()) ** 2)

    metricas = {"MAE": float(mae), "RMSE": float(rmse), "NSE": float(nse)}
    return metricas


def armar_predicciones(tabla: pd.DataFrame, nivel_predicho, estacion: str, nombre_modelo: str,
                       horizonte: int) -> pd.DataFrame:
    """Junta en una tabla las predicciones de un conjunto (validación o prueba) y un horizonte.

    Args:
        tabla: ``tabla_val`` o ``tabla_test`` de ``preparar_datos`` (aporta fechas, nivel real y máscara).
        nivel_predicho: nivel pronosticado en metros, en el mismo orden que ``tabla``.
        estacion: estación objetivo.
        nombre_modelo: nombre con el que se guardará el modelo.
        horizonte: días de anticipación.

    Returns:
        DataFrame con las columnas ``COLUMNAS_PREDICCIONES``.
    """
    predicciones = pd.DataFrame({
        "Fecha": pd.to_datetime(tabla["fecha_objetivo"]).to_numpy(),
        "Estacion": estacion,
        "Modelo": nombre_modelo,
        "Horizonte": horizonte,
        "Real": tabla["nivel_objetivo"].to_numpy(),
        "Prediccion": np.asarray(nivel_predicho, dtype=float).ravel(),
        "Observado": tabla["observado"].to_numpy(),
    })
    return predicciones


def _reemplazar_en_csv(ruta, nuevas: pd.DataFrame) -> None:
    """Sustituye en el CSV las filas de los mismos (estación, modelo) por ``nuevas``."""
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    if ruta.exists():
        anteriores = pd.read_csv(ruta, parse_dates=["Fecha"])
        claves_nuevas = nuevas[["Estacion", "Modelo"]].drop_duplicates()
        for _, clave in claves_nuevas.iterrows():
            mismas = (anteriores["Estacion"] == clave["Estacion"]) & (anteriores["Modelo"] == clave["Modelo"])
            anteriores = anteriores[~mismas]
        nuevas = pd.concat([anteriores, nuevas], ignore_index=True)

    nuevas = nuevas.sort_values(["Estacion", "Modelo", "Horizonte", "Fecha"])
    nuevas["Real"] = nuevas["Real"].round(4)
    nuevas["Prediccion"] = nuevas["Prediccion"].round(4)
    nuevas.to_csv(ruta, index=False)


def guardar_metricas(estacion: str, nombre_modelo: str, metricas_por_horizonte: dict) -> None:
    """Añade (o reemplaza) las métricas de un modelo/estación en ``metrics.json``."""
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    if RUTA_METRICAS.exists():
        with open(RUTA_METRICAS, "r", encoding="utf-8") as archivo:
            todas_las_metricas = json.load(archivo)
    else:
        todas_las_metricas = {}

    if estacion not in todas_las_metricas:
        todas_las_metricas[estacion] = {}
    todas_las_metricas[estacion][nombre_modelo] = {}
    for horizonte, metricas in metricas_por_horizonte.items():
        redondeadas = {}
        for nombre, valor in metricas.items():
            redondeadas[nombre] = round(valor, 6)
        todas_las_metricas[estacion][nombre_modelo][str(horizonte)] = redondeadas

    with open(RUTA_METRICAS, "w", encoding="utf-8") as archivo:
        json.dump(todas_las_metricas, archivo, indent=2, ensure_ascii=False)


def evaluar_y_guardar(estacion: str, nombre_modelo: str, predicciones_val: pd.DataFrame,
                      predicciones_test: pd.DataFrame) -> dict:
    """Calcula las métricas de prueba por horizonte, guarda todo e imprime un resumen.

    Args:
        estacion: estación objetivo.
        nombre_modelo: nombre del modelo.
        predicciones_val: salida de :func:`armar_predicciones` (todos los horizontes) en validación.
        predicciones_test: lo mismo en prueba.

    Returns:
        ``{horizonte: {"MAE", "RMSE", "NSE"}}`` calculadas en prueba con días observados.
    """
    metricas_por_horizonte = {}
    for horizonte in sorted(predicciones_test["Horizonte"].unique()):
        del_horizonte = predicciones_test[(predicciones_test["Horizonte"] == horizonte) & predicciones_test["Observado"]]
        metricas_por_horizonte[int(horizonte)] = calcular_metricas(del_horizonte["Real"], del_horizonte["Prediccion"])

    guardar_metricas(estacion, nombre_modelo, metricas_por_horizonte)
    _reemplazar_en_csv(RUTA_PREDICCIONES, predicciones_test)
    _reemplazar_en_csv(RUTA_PREDICCIONES_VALIDACION, predicciones_val)

    # Resumen: MAE en centímetros en algunos horizontes
    partes = []
    for horizonte, metricas in metricas_por_horizonte.items():
        if horizonte in HORIZONTES_RESUMEN:
            partes.append("h" + str(horizonte) + " " + ("%.1f" % (100 * metricas["MAE"])))
    print(estacion.ljust(15), nombre_modelo.ljust(22), "MAE (cm):", " | ".join(partes), flush=True)
    return metricas_por_horizonte


def cargar_metricas() -> pd.DataFrame:
    """Lee ``metrics.json`` como tabla y añade la mejora del MAE respecto a la persistencia.

    Returns:
        DataFrame con columnas ``Estacion, Modelo, Horizonte, MAE, RMSE, NSE, Mejora_vs_persistencia``
        (la mejora en %, positiva si el modelo es mejor que la persistencia).
    """
    columnas = ["Estacion", "Modelo", "Horizonte", "MAE", "RMSE", "NSE", "Mejora_vs_persistencia"]
    if not RUTA_METRICAS.exists():
        return pd.DataFrame(columns=columnas)

    with open(RUTA_METRICAS, "r", encoding="utf-8") as archivo:
        todas_las_metricas = json.load(archivo)

    filas = []
    for estacion in todas_las_metricas:
        for nombre_modelo in todas_las_metricas[estacion]:
            for horizonte in todas_las_metricas[estacion][nombre_modelo]:
                valores = todas_las_metricas[estacion][nombre_modelo][horizonte]
                filas.append({
                    "Estacion": estacion,
                    "Modelo": nombre_modelo,
                    "Horizonte": int(horizonte),
                    "MAE": valores["MAE"],
                    "RMSE": valores["RMSE"],
                    "NSE": valores["NSE"],
                })
    tabla = pd.DataFrame(filas)

    # Mejora relativa respecto a la persistencia de la misma estación y horizonte
    persistencia = tabla[tabla["Modelo"] == "Persistencia"][["Estacion", "Horizonte", "MAE"]]
    persistencia = persistencia.rename(columns={"MAE": "MAE_persistencia"})
    tabla = tabla.merge(persistencia, on=["Estacion", "Horizonte"], how="left")
    tabla["Mejora_vs_persistencia"] = 100.0 * (1.0 - tabla["MAE"] / tabla["MAE_persistencia"])
    return tabla[columnas]


def cargar_predicciones(conjunto: str = "prueba") -> pd.DataFrame:
    """Lee las predicciones de ``"prueba"`` o ``"validacion"`` (tabla vacía si no existen)."""
    if conjunto == "validacion":
        ruta = RUTA_PREDICCIONES_VALIDACION
    else:
        ruta = RUTA_PREDICCIONES
    if not ruta.exists():
        return pd.DataFrame(columns=COLUMNAS_PREDICCIONES)
    return pd.read_csv(ruta, parse_dates=["Fecha"])
