"""Configuración central del proyecto: rutas, estaciones, horizontes y fechas de corte."""
from pathlib import Path
import pandas as pd

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = CARPETA_PROYECTO / "data" / "DATOS RIO ORINOCO.xlsx"
HOJA_DATOS = "Entrenar"
COLUMNAS_EXCEL = {
    "Puerto Ayacucho": "Ayacucho",
    "Caicara el mini mini": "Caicara",
    "Ciudad Bolívar": "Ciudad_Bolivar",
    "Palúa": "Palua",
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

CAMBIOS_DE_REFERENCIA = [
    ("Ciudad_Bolivar", "1987-05-15", -2.20),
    ("Ciudad_Bolivar", "2019-01-01", 1.00),
]

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
HORIZONTE_BACKTEST = 14
LOOKBACKS_BACKTEST = [30, 60, 90]
MODO_PARTICION_BACKTEST = "porcentaje"  # Opciones: "porcentaje", "fechas_actuales"

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
    """Calcula el corte cronológico del train/holdout (por defecto 70/30) sin mezclar aleatoriamente.

    El 70% inicial temporal corresponde a entrenamiento y el 30% final al holdout.
    """
    fechas_ordenadas = pd.DatetimeIndex(fechas).sort_values().drop_duplicates()
    if len(fechas_ordenadas) == 0:
        raise ValueError("El índice de fechas está vacío.")
    indice_corte = int(len(fechas_ordenadas) * porcentaje_train)
    indice_corte = min(max(1, indice_corte), len(fechas_ordenadas) - 1)
    return fechas_ordenadas[indice_corte]


def obtener_rango_holdout(
    niveles: pd.DataFrame,
    observados: pd.DataFrame,
    modo: str = MODO_PARTICION_BACKTEST,
    porcentaje_train: float = PORCENTAJE_TRAIN_BACKTEST,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Devuelve el inicio y fin válidos del conjunto holdout para backtesting.

    El cálculo se realiza preferentemente sobre las fechas comunes observadas en las cuatro estaciones.
    """
    if modo == "fechas_actuales":
        inicio = pd.Timestamp(FECHA_FIN_VAL) + pd.Timedelta(days=1)
        fin = niveles.index.max()
        return inicio, fin

    fechas_comunes = observados.all(axis=1)
    fechas_validas = niveles.index[fechas_comunes]
    if len(fechas_validas) == 0:
        fechas_validas = niveles.index

    inicio = calcular_corte_holdout(fechas_validas, porcentaje_train)
    fin = niveles.index.max()
    return inicio, fin
