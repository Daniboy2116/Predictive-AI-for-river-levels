"""Configuración central del proyecto: rutas, estaciones, horizontes y fechas de corte."""
from pathlib import Path
import pandas as pd
import numpy as np

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
# Serie auditada (data/auditoria/INFORME_AUDITORIA.md): hojas por estación de "DATOS RIO ORINOCO.xlsx",
# sin imputados ni días aberrantes. No se usa la hoja "Entrenar" (Ciudad Bolívar corrida un día en 2020-2021).
# Empieza en 2009 porque Ayacucho no tiene datos reales entre 2007-01-01 y 2008-11-30.
RUTA_DATOS = CARPETA_PROYECTO / "data" / "niveles_reales.xlsx"
HOJA_DATOS = "Niveles"
COLUMNAS_EXCEL = {
    "Ayacucho": "Ayacucho",
    "Caicara": "Caicara",
    "Ciudad_Bolivar": "Ciudad_Bolivar",
    "Palua": "Palua",
}
RUTA_DATOS_SIMULADOS = CARPETA_PROYECTO / "data" / "datos_rios.xlsx"
HOJA_DATOS_SIMULADOS = "Niveles"
CARPETA_SALIDA = CARPETA_PROYECTO / "outputs"
CARPETA_ESCALADORES = CARPETA_SALIDA / "escaladores"
CARPETA_MODELOS = CARPETA_SALIDA / "modelos"
ARCHIVO_METADATOS_MODELOS = CARPETA_MODELOS / "metadata.json"

ESTACIONES = ["Ayacucho", "Caicara", "Ciudad_Bolivar", "Palua"]
ESTACIONES_OBJETIVO = ["Ciudad_Bolivar", "Palua"]

NOMBRES_ESTACIONES = {
    "Ayacucho": "Puerto Ayacucho",
    "Caicara": "Caicara del Orinoco",
    "Ciudad_Bolivar": "Ciudad Bolívar",
    "Palua": "Palúa",
}

UMBRALES_ALERTA = {"Ciudad_Bolivar": 16.5}

# Los supuestos cambios de Ciudad Bolívar (1987-05-15 y 2019-01-01) son pulsos de 7-10 días, no escalones:
# frente a Caicara (t-4) la diferencia anual no cambia. Ver data/auditoria/INFORME_AUDITORIA.md, anexo D.
CAMBIOS_DE_REFERENCIA = []

VECINAS_CONTROL = {
    "Ayacucho": [("Caicara", 14), ("Ciudad_Bolivar", 18)],
    "Caicara": [("Ciudad_Bolivar", 4), ("Palua", 4)],
    "Ciudad_Bolivar": [("Palua", 0), ("Caicara", -4)],
    "Palua": [("Ciudad_Bolivar", 0), ("Caicara", -4)],
}

SEMILLA = 42
HORIZONTE_PRINCIPAL = 15
HORIZONTE_MAXIMO = 15
HORIZONTES = list(range(1, HORIZONTE_MAXIMO + 1))
HORIZONTES_RESUMEN = [1, 3, 5, 7, 10, 15]

FECHA_FIN_TRAIN = "2016-12-31"
FECHA_FIN_VAL = "2019-12-31"

ARMONICOS_CALENDARIO = 3
ARMONICOS_CLIMATOLOGIA = 2
REZAGOS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]
VENTANAS_TENDENCIA = [1, 3, 7, 15, 30]
VENTANAS_MEDIA_MOVIL = [7, 30]

# --------------------------------------------------------------------------
# Configuración del Backtesting Interactivo
# --------------------------------------------------------------------------
PORCENTAJE_TRAIN_BACKTEST = 0.70
PORCENTAJE_VALIDACION_BACKTEST = 0.15
PORCENTAJE_PRUEBA_BACKTEST = 0.15
HORIZONTE_BACKTEST = 14
LOOKBACKS_BACKTEST = [30, 60, 90]
MODO_PARTICION_BACKTEST = "porcentaje"  # El simulador usa estrictamente 70/15/15.

TEMPORADAS_HIDROLOGICAS = [
    {
        "nombre": "Estiaje / Aguas Bajas",
        "codigo": "estiaje",
        "meses": [1, 2, 3, 4],
        "color": "#38bdf8",
        "descripcion": "Periodo climatológico de niveles mínimos del río Orinoco.",
    },
    {
        "nombre": "Transición / Ascenso",
        "codigo": "transicion",
        "meses": [5, 6, 10, 11],
        "color": "#facc15",
        "descripcion": "Periodo de transición estacional con caudales intermedios.",
    },
    {
        "nombre": "Crecida / Aguas Altas",
        "codigo": "crecida",
        "meses": [7, 8, 9],
        "color": "#ef4444",
        "descripcion": "Periodo de crecida anual con máximos caudales e inundación.",
    },
    {
        "nombre": "Transición / Descenso",
        "codigo": "descenso",
        "meses": [12],
        "color": "#a78bfa",
        "descripcion": "Periodo de inicio de descenso hacia los niveles de estiaje.",
    },
]


def calcular_corte_holdout(
    fechas: pd.DatetimeIndex,
    porcentaje_train: float = PORCENTAJE_TRAIN_BACKTEST,
) -> pd.Timestamp:
    """Calcula un corte cronológico simple para compatibilidad con utilidades antiguas."""
    fechas_ordenadas = pd.DatetimeIndex(fechas).sort_values().drop_duplicates()
    if len(fechas_ordenadas) == 0:
        raise ValueError("El índice de fechas está vacío.")
    indice_corte = int(len(fechas_ordenadas) * porcentaje_train)
    indice_corte = min(max(1, indice_corte), len(fechas_ordenadas) - 1)
    return fechas_ordenadas[indice_corte]


def calcular_cortes_backtest(
    fechas: pd.DatetimeIndex,
    porcentaje_train: float = PORCENTAJE_TRAIN_BACKTEST,
    porcentaje_validacion: float = PORCENTAJE_VALIDACION_BACKTEST,
    porcentaje_prueba: float = PORCENTAJE_PRUEBA_BACKTEST,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Calcula los inicios de validación y prueba con una partición cronológica 70/15/15."""
    suma_porcentajes = porcentaje_train + porcentaje_validacion + porcentaje_prueba
    if not np.isclose(suma_porcentajes, 1.0):
        raise ValueError("Los porcentajes de train, validación y prueba deben sumar 1.")

    fechas_ordenadas = pd.DatetimeIndex(fechas).sort_values().drop_duplicates()
    if len(fechas_ordenadas) < 3:
        raise ValueError("Se necesitan al menos tres fechas para crear la partición 70/15/15.")

    indice_validacion = max(1, int(len(fechas_ordenadas) * porcentaje_train))
    indice_prueba = max(
        indice_validacion + 1,
        int(len(fechas_ordenadas) * (porcentaje_train + porcentaje_validacion)),
    )
    indice_prueba = min(indice_prueba, len(fechas_ordenadas) - 1)
    return fechas_ordenadas[indice_validacion], fechas_ordenadas[indice_prueba]


def obtener_rangos_particion(
    niveles: pd.DataFrame,
    observados: pd.DataFrame,
    modo: str = MODO_PARTICION_BACKTEST,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Devuelve inicio de validación, inicio de prueba y fin de una partición 70/15/15."""
    if modo != "porcentaje":
        raise ValueError("El simulador interactivo solo admite la partición porcentual 70/15/15.")

    fechas_comunes = observados.all(axis=1)
    fechas_validas = niveles.index[fechas_comunes]
    if len(fechas_validas) == 0:
        fechas_validas = niveles.index

    inicio_validacion, inicio_prueba = calcular_cortes_backtest(fechas_validas)
    return inicio_validacion, inicio_prueba, niveles.index.max()


def obtener_rango_holdout(
    niveles: pd.DataFrame,
    observados: pd.DataFrame,
    modo: str = MODO_PARTICION_BACKTEST,
    porcentaje_train: float = PORCENTAJE_TRAIN_BACKTEST,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Devuelve el rango de prueba (último 15 %) del simulador 70/15/15."""
    _, inicio_prueba, fin_prueba = obtener_rangos_particion(niveles, observados, modo)
    return inicio_prueba, fin_prueba
