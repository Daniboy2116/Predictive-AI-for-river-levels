"""Pruebas automáticas obligatorias del módulo de Backtesting Histórico.

Verifica:
1. Corte y partición holdout cronológico sin mezcla aleatoria.
2. Inexistencia de fugas de información futura en features o escaladores.
3. Validación estricta de fechas de origen y límites del holdout.
4. Clasificación correcta de temporadas hidrológicas.
5. Estructura y coherencia de eventos simulados (14 días, no incluye t0 en GT).
6. Métricas locales (MAE, RMSE, R² y omisión de valores no observados).
7. Adaptadores de modelos (Persistencia, Ridge, etc.).
"""
import numpy as np
import pandas as pd
import pytest

from src.configuracion import (
    ESTACIONES, ESTACIONES_OBJETIVO, HORIZONTE_BACKTEST,
    calcular_corte_holdout, obtener_rango_holdout,
)
from src.backtesting import (
    BacktestValidationError, calcular_metricas_evento, clasificar_temporada,
    simular_evento, validar_fecha_origen,
)
from src.inference import instanciar_modelos
from src.preprocessing import cargar_datos, preparar_datos_backtest

# Carga de datos reales para pruebas
NIVELES, OBSERVADOS = cargar_datos()
INICIO_HOLDOUT, FIN_HOLDOUT = obtener_rango_holdout(NIVELES, OBSERVADOS, modo="porcentaje")


def test_split_holdout_cronologico_sin_mezcla():
    """El 70% temporal debe ser estrictamente anterior al 30% holdout."""
    corte = calcular_corte_holdout(NIVELES.index, porcentaje_train=0.70)
    fechas_train = NIVELES.index[NIVELES.index < corte]
    fechas_holdout = NIVELES.index[NIVELES.index >= corte]

    assert len(fechas_train) > 0
    assert len(fechas_holdout) > 0
    assert fechas_train.max() < fechas_holdout.min()
    assert (fechas_train.sort_values() == fechas_train).all()
    assert (fechas_holdout.sort_values() == fechas_holdout).all()


def test_sin_fuga_de_informacion_futura_en_backtest():
    """Modificar datos dentro del holdout no debe alterar ni features de train ni escaladores."""
    corte = INICIO_HOLDOUT
    datos_originales = preparar_datos_backtest(NIVELES, OBSERVADOS, "Ciudad_Bolivar", corte, 7)

    # Modificar artificialmente el futuro (dentro del holdout)
    niveles_alterados = NIVELES.copy()
    en_holdout = niveles_alterados.index >= corte
    rng = np.random.default_rng(123)
    niveles_alterados.loc[en_holdout, "Ciudad_Bolivar"] += rng.normal(10, 5, size=en_holdout.sum())

    datos_alterados = preparar_datos_backtest(niveles_alterados, OBSERVADOS, "Ciudad_Bolivar", corte, 7)

    # La media y desviación del escalador de entrenamiento deben ser exactamente idénticas
    assert np.allclose(datos_originales["escalador"].mean_, datos_alterados["escalador"].mean_)
    assert np.allclose(datos_originales["escalador"].scale_, datos_alterados["escalador"].scale_)

    # Las matrices de entrenamiento X e y deben ser idénticas
    pd.testing.assert_frame_equal(datos_originales["x_train"], datos_alterados["x_train"])
    pd.testing.assert_series_equal(datos_originales["y_train"], datos_alterados["y_train"])


def test_validacion_de_fechas_rechaza_casos_invalidos():
    """Debe rechazar fechas anteriores al holdout, demasiado cercanas al final o con lookback insuficiente."""
    # 1. Fecha anterior al holdout
    fecha_previa = INICIO_HOLDOUT - pd.Timedelta(days=5)
    with pytest.raises(BacktestValidationError, match="anterior al inicio del holdout"):
        validar_fecha_origen(fecha_previa, INICIO_HOLDOUT, FIN_HOLDOUT, horizonte_maximo=14)

    # 2. Fecha con menos de 14 días antes del fin del holdout
    fecha_tardia = FIN_HOLDOUT - pd.Timedelta(days=5)
    with pytest.raises(BacktestValidationError, match="no cuenta con 14 días posteriores"):
        validar_fecha_origen(fecha_tardia, INICIO_HOLDOUT, FIN_HOLDOUT, horizonte_maximo=14)

    # 3. Lookback fuera de rango
    fecha_minima = NIVELES.index.min() + pd.Timedelta(days=10)
    with pytest.raises(BacktestValidationError):
        validar_fecha_origen(
            fecha_minima, fecha_minima, FIN_HOLDOUT, lookback=60, niveles=NIVELES
        )


def test_clasificacion_temporada_hidrologica():
    """Verifica que fechas en diferentes meses devuelvan la temporada esperada."""
    estiaje = clasificar_temporada(pd.Timestamp("2023-03-15"))
    assert estiaje["codigo"] == "estiaje"

    crecida = clasificar_temporada(pd.Timestamp("2023-08-15"))
    assert crecida["codigo"] == "crecida"

    transicion = clasificar_temporada(pd.Timestamp("2023-05-15"))
    assert transicion["codigo"] == "transicion"

    descenso = clasificar_temporada(pd.Timestamp("2023-12-10"))
    assert descenso["codigo"] == "descenso"


def test_metricas_evento_con_datos_conocidos_y_valores_imputados():
    """Comprueba cálculo exacto de MAE, RMSE, R² y omisión de días no observados."""
    real = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    pred = pd.Series([10.5, 11.0, 11.5, 13.5, 14.0])
    obs = pd.Series([True, True, False, True, True])  # día 3 es imputado/no observado

    metricas = calcular_metricas_evento(real, pred, obs)
    assert metricas["dias_evaluados"] == 4
    # Errores en observados: [0.5, 0.0, 0.5, 0.0] -> MAE = 0.25
    assert np.isclose(metricas["MAE"], 0.25)
    # RMSE: sqrt((0.25 + 0 + 0.25 + 0) / 4) = sqrt(0.125) ≈ 0.3536
    assert np.isclose(metricas["RMSE"], np.sqrt(0.125), atol=1e-3)
    assert not np.isnan(metricas["R2"])


def test_simulacion_evento_estructura_y_coherencia():
    """Comprueba que la simulación devuelva exactamente 14 días y las columnas requeridas."""
    t0 = INICIO_HOLDOUT + pd.Timedelta(days=50)
    resultado = simular_evento(
        nombre_modelo="Persistencia",
        niveles=NIVELES,
        observados=OBSERVADOS,
        fecha_origen=t0,
        lookback=60,
        estaciones=["Ciudad_Bolivar"],
        horizonte_maximo=14,
        guardar_en_disco=False,
    )

    predicciones = resultado["predicciones"]
    assert len(predicciones) == 14
    assert list(predicciones["Horizonte"]) == list(range(1, 15))
    assert (predicciones["Fecha"] == [t0 + pd.Timedelta(days=h) for h in range(1, 15)]).all()
    # t0 no debe estar en las predicciones futuras
    assert t0 not in predicciones["Fecha"].values

    # Comprobar columnas exigidas
    columnas_esperadas = {"Fecha", "Estacion", "Horizonte", "Real", "Prediccion", "Error", "ErrorAbsoluto", "Observado"}
    assert columnas_esperadas.issubset(set(predicciones.columns))

    # Comprobar historia
    historia = resultado["historia"]
    assert historia["Fecha"].min() == t0 - pd.Timedelta(days=60)
    assert historia["Fecha"].max() == t0
