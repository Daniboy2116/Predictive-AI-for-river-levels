from pathlib import Path

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
