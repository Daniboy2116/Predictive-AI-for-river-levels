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
    y_real = np.asarray(y_real, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    nse = 1.0 - np.sum((y_real - y_pred) ** 2) / np.sum((y_real - y_real.mean()) ** 2)

    return {"MAE": float(mae), "RMSE": float(rmse), "NSE": float(nse)}


def armar_predicciones(tabla: pd.DataFrame, nivel_predicho, estacion: str, nombre_modelo: str,
                       horizonte: int) -> pd.DataFrame:
    return pd.DataFrame({
        "Fecha": pd.to_datetime(tabla["fecha_objetivo"]).to_numpy(),
        "Estacion": estacion,
        "Modelo": nombre_modelo,
        "Horizonte": horizonte,
        "Real": tabla["nivel_objetivo"].to_numpy(),
        "Prediccion": np.asarray(nivel_predicho, dtype=float).ravel(),
        "Observado": tabla["observado"].to_numpy(),
    })


def _reemplazar_en_csv(ruta, nuevas: pd.DataFrame) -> None:
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
        redondeadas = {nombre: round(valor, 6) for nombre, valor in metricas.items()}
        todas_las_metricas[estacion][nombre_modelo][str(horizonte)] = redondeadas

    with open(RUTA_METRICAS, "w", encoding="utf-8") as archivo:
        json.dump(todas_las_metricas, archivo, indent=2, ensure_ascii=False)


def evaluar_y_guardar(estacion: str, nombre_modelo: str, predicciones_val: pd.DataFrame,
                      predicciones_test: pd.DataFrame) -> dict:
    metricas_por_horizonte = {}
    for horizonte in sorted(predicciones_test["Horizonte"].unique()):
        del_horizonte = predicciones_test[(predicciones_test["Horizonte"] == horizonte) & predicciones_test["Observado"]]
        metricas_por_horizonte[int(horizonte)] = calcular_metricas(del_horizonte["Real"], del_horizonte["Prediccion"])

    guardar_metricas(estacion, nombre_modelo, metricas_por_horizonte)
    _reemplazar_en_csv(RUTA_PREDICCIONES, predicciones_test)
    _reemplazar_en_csv(RUTA_PREDICCIONES_VALIDACION, predicciones_val)

    partes = []
    for horizonte, metricas in metricas_por_horizonte.items():
        if horizonte in HORIZONTES_RESUMEN:
            partes.append(f"h{horizonte} {100 * metricas['MAE']:.1f}")
    print(estacion.ljust(15), nombre_modelo.ljust(22), "MAE (cm):", " | ".join(partes), flush=True)
    return metricas_por_horizonte


def cargar_metricas() -> pd.DataFrame:
    columnas = ["Estacion", "Modelo", "Horizonte", "MAE", "RMSE", "NSE", "Mejora_vs_persistencia"]
    if not RUTA_METRICAS.exists():
        return pd.DataFrame(columns=columnas)

    with open(RUTA_METRICAS, "r", encoding="utf-8") as archivo:
        todas_las_metricas = json.load(archivo)

    filas = []
    for estacion in todas_las_metricas:
        for nombre_modelo in todas_las_metricas[estacion]:
            for horizonte, valores in todas_las_metricas[estacion][nombre_modelo].items():
                filas.append({
                    "Estacion": estacion,
                    "Modelo": nombre_modelo,
                    "Horizonte": int(horizonte),
                    "MAE": valores["MAE"],
                    "RMSE": valores["RMSE"],
                    "NSE": valores["NSE"],
                })
    tabla = pd.DataFrame(filas)

    persistencia = tabla[tabla["Modelo"] == "Persistencia"][["Estacion", "Horizonte", "MAE"]]
    persistencia = persistencia.rename(columns={"MAE": "MAE_persistencia"})
    tabla = tabla.merge(persistencia, on=["Estacion", "Horizonte"], how="left")
    tabla["Mejora_vs_persistencia"] = 100.0 * (1.0 - tabla["MAE"] / tabla["MAE_persistencia"])
    return tabla[columnas]


def cargar_predicciones(conjunto: str = "prueba") -> pd.DataFrame:
    ruta = RUTA_PREDICCIONES_VALIDACION if conjunto == "validacion" else RUTA_PREDICCIONES
    if not ruta.exists():
        return pd.DataFrame(columns=COLUMNAS_PREDICCIONES)
    return pd.read_csv(ruta, parse_dates=["Fecha"])
