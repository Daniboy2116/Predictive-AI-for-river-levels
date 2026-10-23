"""Pruebas del preprocesamiento: control de calidad, fugas de información y split.

Ejecutar:  python -m pytest
"""
import numpy as np
import pandas as pd

from src.configuracion import ESTACIONES, FECHA_FIN_TRAIN, FECHA_FIN_VAL
from src.preprocessing import (
    ajustar_climatologias, cambio_a_nivel, cargar_datos, construir_tabla, grupos_de_variables, leer_excel,
    nombres_variables, preparar_datos, preparar_ventanas, revisar_calidad,
)

# Los datos se cargan una sola vez para todas las pruebas
CRUDOS = leer_excel()
NIVELES, OBSERVADOS = cargar_datos()


def test_indice_diario_continuo():
    """Tras leer el Excel no faltan fechas: las ausentes aparecen como filas con NaN."""
    diferencias = CRUDOS.index.to_series().diff().dropna()
    assert (diferencias == pd.Timedelta(days=1)).all()
    assert list(CRUDOS.columns) == ESTACIONES


def test_control_de_calidad_detecta_los_saltos_simulados():
    """Solo se confirman los dos cambios de referencia de Ciudad Bolívar."""
    revision = revisar_calidad(CRUDOS)
    confirmados = revision[revision["Confirmado"]]
    assert set(confirmados["Estacion"]) == {"Ciudad_Bolivar"}

    fechas = set(confirmados["Fecha"].dt.strftime("%Y-%m-%d"))
    assert fechas == {"1987-05-15", "2019-01-01"}

    salto_1987 = confirmados[confirmados["Fecha"] == "1987-05-15"]["Salto"].mean()
    salto_2019 = confirmados[confirmados["Fecha"] == "2019-01-01"]["Salto"].mean()
    assert abs(salto_1987 - (-2.20)) < 0.3
    assert abs(salto_2019 - 1.00) < 0.3


def test_correccion_elimina_los_saltos():
    """Después de corregir no queda ningún salto confirmado."""
    corregidos_sin_imputar = NIVELES.where(OBSERVADOS)
    revision = revisar_calidad(corregidos_sin_imputar)
    assert not revision["Confirmado"].any()


def test_imputacion_sin_nan_y_mascara_correcta():
    """La imputación no deja NaN y la máscara marca como no observados los NaN originales."""
    assert NIVELES.isna().sum().sum() == 0
    assert (OBSERVADOS == CRUDOS.notna()).all().all()


def test_sin_fuga_de_informacion_futura():
    """Cambiar los datos posteriores a t0 no debe alterar ninguna variable de días <= t0."""
    t0 = pd.Timestamp("2021-06-30")
    rng = np.random.default_rng(0)
    alterados = NIVELES.copy()
    posteriores = alterados.index > t0
    alterados.loc[posteriores] = alterados.loc[posteriores] + rng.normal(0, 3, size=alterados.loc[posteriores].shape)

    climatologias_originales = ajustar_climatologias(NIVELES, OBSERVADOS)
    climatologias_alteradas = ajustar_climatologias(alterados, OBSERVADOS)
    for estacion in ESTACIONES:
        assert np.allclose(climatologias_originales[estacion], climatologias_alteradas[estacion])

    tabla_original = construir_tabla(NIVELES, OBSERVADOS, climatologias_originales, "Ciudad_Bolivar", 15)
    tabla_alterada = construir_tabla(alterados, OBSERVADOS, climatologias_alteradas, "Ciudad_Bolivar", 15)
    variables = nombres_variables(tabla_original)
    hasta_t0 = tabla_original.index <= t0
    pd.testing.assert_frame_equal(tabla_original.loc[hasta_t0, variables], tabla_alterada.loc[hasta_t0, variables])


def test_split_por_fecha_objetivo():
    """Train termina en 2014, validación cubre 2015-2019 y prueba empieza en 2020."""
    datos = preparar_datos(NIVELES, OBSERVADOS, "Palua", 15)
    fin_train = pd.Timestamp(FECHA_FIN_TRAIN)
    fin_val = pd.Timestamp(FECHA_FIN_VAL)

    assert datos["tabla_train"]["fecha_objetivo"].max() <= fin_train
    assert datos["tabla_val"]["fecha_objetivo"].min() > fin_train
    assert datos["tabla_val"]["fecha_objetivo"].max() <= fin_val
    assert datos["tabla_test"]["fecha_objetivo"].min() > fin_val
    # El objetivo de train nunca usa un día de validación
    ultimo_origen_train = datos["tabla_train"].index.max()
    assert ultimo_origen_train + pd.Timedelta(days=15) <= fin_train


def test_escalador_ajustado_solo_con_train():
    """El Z-Score deja train con media 0 y desviación 1 (validación y prueba no se usan para ajustar)."""
    datos = preparar_datos(NIVELES, OBSERVADOS, "Ciudad_Bolivar", 15)
    medias = datos["x_train"].mean()
    desviaciones = datos["x_train"].std(ddof=0)
    assert np.allclose(medias, 0, atol=1e-8)
    assert np.allclose(desviaciones, 1, atol=1e-6)
    assert np.allclose(datos["escalador"].mean_, datos["tabla_train"][datos["variables"]].mean())


def test_79_variables_en_tres_grupos():
    """7 cíclicas + 18 propias + 54 de las otras estaciones = 79 variables."""
    datos = preparar_datos(NIVELES, OBSERVADOS, "Ciudad_Bolivar", 15)
    assert len(datos["variables"]) == 79
    grupos = grupos_de_variables(datos["variables"], "Ciudad_Bolivar")
    assert len(grupos["ciclicas"]) == 7
    assert len(grupos["propias"]) == 18
    assert len(grupos["otras_estaciones"]) == 54


def test_reconstruccion_del_nivel():
    """Nivel actual + cambio real = nivel objetivo."""
    datos = preparar_datos(NIVELES, OBSERVADOS, "Palua", 7)
    tabla = datos["tabla_test"]
    reconstruido = cambio_a_nivel(tabla["nivel_actual"], datos["y_test"])
    assert np.allclose(reconstruido, tabla["nivel_objetivo"])


def test_ventanas_terminan_en_el_dia_de_emision():
    """La última fila de cada ventana es el día t y los objetivos de train no pasan de 2014."""
    datos = preparar_ventanas(NIVELES, OBSERVADOS, "Ciudad_Bolivar", 60, 15)
    assert datos["ventanas"].shape[1:] == (60, 10)

    posicion = 5000
    origen = datos["origenes"][posicion]
    columna = datos["canales"].index("Ciudad_Bolivar_nivel")
    escalador = datos["escalador_canales"]
    nivel_recuperado = datos["ventanas"][posicion, -1, columna] * escalador.scale_[columna] + escalador.mean_[columna]
    assert np.isclose(nivel_recuperado, NIVELES.loc[origen, "Ciudad_Bolivar"], atol=1e-4)

    ultimo_origen_train = datos["origenes"][datos["en_train"]].max()
    assert ultimo_origen_train + pd.Timedelta(days=15) <= pd.Timestamp(FECHA_FIN_TRAIN)
    assert not (datos["en_train"] & datos["en_val"]).any()


def test_pronostico_operativo_futuro():
    """Los últimos h días de emisión quedan en 'futuro': variables completas y objetivo desconocido."""
    ultimo_dia = NIVELES.index.max()
    for horizonte in [1, 7, 15]:
        datos = preparar_datos(NIVELES, OBSERVADOS, "Ciudad_Bolivar", horizonte)
        futuro = datos["tabla_futuro"]
        assert len(futuro) == horizonte
        assert futuro.index.max() == ultimo_dia
        assert (futuro["fecha_objetivo"] > ultimo_dia).all()
        assert futuro["nivel_objetivo"].isna().all()
        assert not futuro["observado"].any()
        assert not datos["x_futuro"].isna().any().any()
        # Prueba sigue teniendo objetivo real en todas sus filas
        assert datos["tabla_test"]["nivel_objetivo"].notna().all()
        assert datos["tabla_test"]["fecha_objetivo"].max() == ultimo_dia
