"""Módulo central de lógica, validación y métricas para el Simulador de Backtesting Histórico.

Contiene exclusivamente funciones de datos, cálculo y evaluación sin dependencias de UI.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.configuracion import (
    CARPETA_SALIDA, ESTACIONES, ESTACIONES_OBJETIVO, HORIZONTE_BACKTEST,
    MODO_PARTICION_BACKTEST, TEMPORADAS_HIDROLOGICAS, obtener_rangos_particion,
)
from src.inference import REGISTRO_MODELOS, obtener_modelo_backtest
from src.preprocessing import cargar_datos

ARCHIVO_EVENTOS_BACKTEST = CARPETA_SALIDA / "backtest_events.csv"


class BacktestValidationError(ValueError):
    """Excepción específica para errores de validación en simulaciones históricas."""
    pass


def cargar_datos_backtest() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los niveles corregidos e interpolados junto con la máscara booleana de observaciones reales."""
    return cargar_datos()


def clasificar_temporada(fecha: pd.Timestamp) -> dict:
    """Clasifica una fecha en una temporada hidrológica del río Orinoco."""
    fecha = pd.Timestamp(fecha)
    mes = fecha.month
    for temp in TEMPORADAS_HIDROLOGICAS:
        if mes in temp["meses"]:
            return {
                "nombre": temp["nombre"],
                "codigo": temp["codigo"],
                "mes": mes,
                "descripcion": temp["descripcion"],
                "color": temp["color"],
                "dia_del_anio": int(getattr(fecha, "day_of_year", getattr(fecha, "dayofyear", 1))),
            }
    # Fallback por seguridad
    return {
        "nombre": "Transición",
        "codigo": "transicion",
        "mes": mes,
        "descripcion": "Periodo de transición estacional.",
        "color": "#facc15",
        "dia_del_anio": int(getattr(fecha, "day_of_year", getattr(fecha, "dayofyear", 1))),
    }


def validar_fecha_origen(
    fecha_origen: pd.Timestamp,
    inicio_holdout: pd.Timestamp,
    fin_holdout: pd.Timestamp,
    horizonte_maximo: int = HORIZONTE_BACKTEST,
    lookback: int = 60,
    niveles: Optional[pd.DataFrame] = None,
    observados: Optional[pd.DataFrame] = None,
    estaciones: Optional[List[str]] = None,
) -> None:
    """Valida que t0 y su horizonte futuro pertenezcan al conjunto de prueba."""
    t0 = pd.Timestamp(fecha_origen)
    inicio_h = pd.Timestamp(inicio_holdout)
    fin_h = pd.Timestamp(fin_holdout)

    if t0 < inicio_h:
        raise BacktestValidationError(
            f"La fecha de origen seleccionada ({t0.strftime('%Y-%m-%d')}) es anterior al inicio de prueba ({inicio_h.strftime('%Y-%m-%d')})."
        )

    ultimo_origen_valido = fin_h - pd.Timedelta(days=horizonte_maximo)
    if t0 > ultimo_origen_valido:
        raise BacktestValidationError(
            f"La fecha de origen ({t0.strftime('%Y-%m-%d')}) no cuenta con {horizonte_maximo} días posteriores dentro de prueba. "
            f"El último origen válido permitido es {ultimo_origen_valido.strftime('%Y-%m-%d')}."
        )

    if niveles is not None:
        inicio_lookback = t0 - pd.Timedelta(days=lookback)
        if inicio_lookback < niveles.index.min():
            raise BacktestValidationError(
                f"No hay suficientes días históricos previos a {t0.strftime('%Y-%m-%d')} para construir la ventana de lookback ({lookback} días)."
            )

    if observados is not None and estaciones is not None and t0 in observados.index:
        for est in estaciones:
            if est in observados.columns and not bool(observados.loc[t0, est]):
                raise BacktestValidationError(
                    f"El nivel en la fecha origen t₀ ({t0.strftime('%Y-%m-%d')}) no fue observado empíricamente en la estación {est}."
                )


def calcular_metricas_evento(
    real: pd.Series,
    prediccion: pd.Series,
    observados: Optional[pd.Series] = None,
) -> dict:
    """Calcula MAE, RMSE y R² locales únicamente sobre los días efectivamente observados."""
    real_s = pd.Series(real)
    pred_s = pd.Series(prediccion)

    mascara_valida = real_s.notna() & pred_s.notna()
    if observados is not None:
        obs_s = pd.Series(observados).astype(bool)
        mascara_valida = mascara_valida & obs_s

    y_true = real_s[mascara_valida].to_numpy(dtype=float)
    y_pred = pred_s[mascara_valida].to_numpy(dtype=float)
    dias_evaluados = int(len(y_true))

    if dias_evaluados < 2:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "dias_evaluados": dias_evaluados,
            "MAE_1_6": np.nan,
            "MAE_7_14": np.nan,
            "advertencia": "Se requieren al menos 2 observaciones válidas para calcular R² y métricas de error.",
        }

    errores = y_pred - y_true
    mae = float(np.mean(np.abs(errores)))
    rmse = float(np.sqrt(np.mean(errores ** 2)))
    
    # Manejo robusto de varianza cero para R2
    if np.var(y_true) < 1e-9:
        r2 = np.nan
    else:
        r2 = float(r2_score(y_true, y_pred))

    # Segmentación por tramo de horizonte
    indices = np.arange(1, len(real_s) + 1)[mascara_valida.to_numpy()]
    mask_1_6 = indices <= 6
    mask_7_14 = indices >= 7

    mae_1_6 = float(np.mean(np.abs(errores[mask_1_6]))) if np.any(mask_1_6) else np.nan
    mae_7_14 = float(np.mean(np.abs(errores[mask_7_14]))) if np.any(mask_7_14) else np.nan

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4) if not np.isnan(r2) else np.nan,
        "dias_evaluados": dias_evaluados,
        "MAE_1_6": round(mae_1_6, 4) if not np.isnan(mae_1_6) else np.nan,
        "MAE_7_14": round(mae_7_14, 4) if not np.isnan(mae_7_14) else np.nan,
        "advertencia": None,
    }


def simular_evento(
    nombre_modelo: Union[str, List[str]],
    niveles: pd.DataFrame,
    observados: pd.DataFrame,
    fecha_origen: pd.Timestamp,
    lookback: int = 60,
    estaciones: Optional[Union[str, List[str]]] = None,
    horizonte_maximo: int = HORIZONTE_BACKTEST,
    modo_split: str = MODO_PARTICION_BACKTEST,
    inicio_holdout: Optional[pd.Timestamp] = None,
    inicio_prueba: Optional[pd.Timestamp] = None,
    guardar_en_disco: bool = True,
) -> dict:
    """Ejecuta una simulación histórica completa de 1 a horizonte_maximo días para uno o varios modelos."""
    t0 = pd.Timestamp(fecha_origen)
    if estaciones is None:
        estaciones = ESTACIONES_OBJETIVO
    elif isinstance(estaciones, str):
        estaciones = [estaciones]

    if inicio_holdout is None:
        inicio_train, inicio_h, fin_h = obtener_rangos_particion(
            niveles, observados, modo=modo_split,
        )
    else:
        inicio_train = pd.Timestamp(inicio_holdout)
        inicio_h = pd.Timestamp(inicio_prueba) if inicio_prueba is not None else inicio_train
        fin_h = niveles.index.max()

    validar_fecha_origen(
        fecha_origen=t0,
        inicio_holdout=inicio_h,
        fin_holdout=fin_h,
        horizonte_maximo=horizonte_maximo,
        lookback=lookback,
        niveles=niveles,
        observados=observados,
        estaciones=estaciones,
    )

    temporada = clasificar_temporada(t0)
    modelos_lista = [nombre_modelo] if isinstance(nombre_modelo, str) else list(nombre_modelo)
    if not modelos_lista:
        raise BacktestValidationError("Debes especificar al menos un modelo para simular.")

    # Instanciar o recuperar todos los modelos solicitados
    modelos_instancias = {
        m_nom: obtener_modelo_backtest(m_nom, niveles, observados, inicio_train, horizonte_maximo)
        for m_nom in modelos_lista
    }

    # Ventana histórica de historia previa [t0 - lookback, t0]
    inicio_historia = t0 - pd.Timedelta(days=lookback)
    fechas_historia = niveles.loc[inicio_historia:t0].index

    filas_historia = []
    filas_gt = []
    tablas_prediccion = []
    metricas_por_estacion = []

    fechas_futuras = [t0 + pd.Timedelta(days=h) for h in range(1, horizonte_maximo + 1)]

    for est in estaciones:
        # Historia observada (una sola vez por estación)
        for f in fechas_historia:
            obs = bool(observados.loc[f, est])
            filas_historia.append({
                "Fecha": f,
                "Estacion": est,
                "Nivel": float(niveles.loc[f, est]) if obs else np.nan,
                "Observado": obs,
            })

        # Ground Truth futuro (una sola vez por estación)
        for h, f in enumerate(fechas_futuras, start=1):
            obs = bool(observados.loc[f, est]) if f in observados.index else False
            val_real = float(niveles.loc[f, est]) if (obs and f in niveles.index) else np.nan
            filas_gt.append({
                "Fecha": f,
                "Estacion": est,
                "Horizonte": h,
                "Real": val_real,
                "Observado": obs,
            })

        # Predicciones por cada modelo seleccionado
        for m_nom, modelo in modelos_instancias.items():
            df_pred_est = modelo.predecir_evento(niveles, observados, t0, horizonte_maximo, est)
            df_pred_est["Modelo"] = m_nom

            reales_serie = []
            obs_serie = []
            for f in df_pred_est["Fecha"]:
                obs = bool(observados.loc[f, est]) if f in observados.index else False
                val = float(niveles.loc[f, est]) if (obs and f in niveles.index) else np.nan
                reales_serie.append(val)
                obs_serie.append(obs)

            df_pred_est["Real"] = reales_serie
            df_pred_est["Observado"] = obs_serie

            df_pred_est["Error"] = np.where(df_pred_est["Observado"], df_pred_est["Prediccion"] - df_pred_est["Real"], np.nan)
            df_pred_est["ErrorAbsoluto"] = np.where(df_pred_est["Observado"], np.abs(df_pred_est["Error"]), np.nan)

            tablas_prediccion.append(df_pred_est)

            # Métricas de evento locales
            met_est = calcular_metricas_evento(df_pred_est["Real"], df_pred_est["Prediccion"], df_pred_est["Observado"])
            met_est["Modelo"] = m_nom
            met_est["Estacion"] = est
            metricas_por_estacion.append(met_est)

    df_predicciones = pd.concat(tablas_prediccion, ignore_index=True)
    df_historia = pd.DataFrame(filas_historia).drop_duplicates(subset=["Fecha", "Estacion"])
    df_gt = pd.DataFrame(filas_gt).drop_duplicates(subset=["Fecha", "Estacion", "Horizonte"])
    df_metricas = pd.DataFrame(metricas_por_estacion)

    # Persistir evento puntual en outputs/backtest_events.csv
    if guardar_en_disco:
        CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
        registro_disco = df_predicciones.copy()
        registro_disco["FechaOrigen"] = t0
        registro_disco["Lookback"] = lookback
        registro_disco["Split"] = modo_split
        columnas_orden = [
            "Modelo", "Estacion", "FechaOrigen", "Lookback", "Horizonte",
            "Fecha", "Real", "Prediccion", "Error", "ErrorAbsoluto", "Observado", "Split",
        ]
        columnas_existentes = [c for c in columnas_orden if c in registro_disco.columns]
        registro_disco = registro_disco[columnas_existentes]
        if ARCHIVO_EVENTOS_BACKTEST.exists():
            registro_disco.to_csv(ARCHIVO_EVENTOS_BACKTEST, mode="a", header=False, index=False)
        else:
            registro_disco.to_csv(ARCHIVO_EVENTOS_BACKTEST, mode="w", header=True, index=False)

    return {
        "fecha_origen": t0,
        "fecha_inicio_historia": inicio_historia,
        "fecha_fin_historia": t0,
        "temporada": temporada,
        "modelo": ", ".join(modelos_lista) if len(modelos_lista) > 1 else modelos_lista[0],
        "modelos": modelos_lista,
        "lookback": lookback,
        "predicciones": df_predicciones,
        "historia": df_historia,
        "ground_truth": df_gt,
        "metricas": df_metricas,
    }
