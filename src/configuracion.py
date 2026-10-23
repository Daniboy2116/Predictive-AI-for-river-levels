"""Configuración central del proyecto: rutas, estaciones, horizontes y fechas de corte.

Todos los módulos leen sus parámetros de aquí, así un cambio (por ejemplo, otro
horizonte de pronóstico o nuevas fechas de corte) se hace en un único lugar.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = CARPETA_PROYECTO / "data" / "datos_rios.xlsx"
CARPETA_SALIDA = CARPETA_PROYECTO / "outputs"
CARPETA_ESCALADORES = CARPETA_SALIDA / "escaladores"

# --------------------------------------------------------------------------
# Estaciones (ordenadas de aguas arriba a aguas abajo)
# --------------------------------------------------------------------------
ESTACIONES = ["Ayacucho", "Caicara", "Ciudad_Bolivar", "Palua"]

# Estaciones que queremos pronosticar (las demás solo aportan información)
ESTACIONES_OBJETIVO = ["Ciudad_Bolivar", "Palua"]

# Nombres para mostrar en gráficas y en el dashboard
NOMBRES_ESTACIONES = {
    "Ayacucho": "Puerto Ayacucho",
    "Caicara": "Caicara del Orinoco",
    "Ciudad_Bolivar": "Ciudad Bolívar",
    "Palua": "Palúa",
}

# Umbral de alerta por crecida (m), cuando se conoce
UMBRALES_ALERTA = {"Ciudad_Bolivar": 16.5}

# --------------------------------------------------------------------------
# Control de calidad
# --------------------------------------------------------------------------
# Cambios de referencia (cero de la regla) CONFIRMADOS: (estación, fecha, salto en m).
# El salto es lo que cambió la lectura a partir de esa fecha. Al corregir se lleva
# toda la serie a la referencia más reciente (la que usan hoy los umbrales de alerta).
CAMBIOS_DE_REFERENCIA = [
    ("Ciudad_Bolivar", "1987-05-15", -2.20),
    ("Ciudad_Bolivar", "2019-01-01", 1.00),
]

# Para detectar saltos se compara cada estación con DOS vecinas muy correlacionadas:
# estación -> [(vecina, días que tarda la onda de la estación a la vecina), ...]
# Un salto solo se confirma si aparece frente a ambas (si solo aparece frente a una,
# probablemente el problema está en esa vecina).
VECINAS_CONTROL = {
    "Ayacucho": [("Caicara", 14), ("Ciudad_Bolivar", 18)],
    "Caicara": [("Ciudad_Bolivar", 4), ("Palua", 4)],
    "Ciudad_Bolivar": [("Palua", 0), ("Caicara", -4)],
    "Palua": [("Ciudad_Bolivar", 0), ("Caicara", -4)],
}

# --------------------------------------------------------------------------
# Diseño experimental
# --------------------------------------------------------------------------
SEMILLA = 42                        # semilla fija para resultados reproducibles
HORIZONTE_PRINCIPAL = 15            # días de anticipación del pronóstico
HORIZONTE_MAXIMO = 15               # se pronostica cada día desde t+1 hasta t+15
HORIZONTES = list(range(1, HORIZONTE_MAXIMO + 1))   # 1, 2, ..., 15: todos se evalúan y se guardan
HORIZONTES_RESUMEN = [1, 3, 5, 7, 10, 15]           # los que se imprimen en consola

# Partición cronológica según la FECHA OBJETIVO (t + h)
FECHA_FIN_TRAIN = "2014-12-31"      # train: 1974-2014
FECHA_FIN_VAL = "2019-12-31"        # validación: 2015-2019; prueba: 2020-2025

# --------------------------------------------------------------------------
# Variables de entrada
# --------------------------------------------------------------------------
ARMONICOS_CALENDARIO = 3                     # seno/coseno con k = 1, 2, 3
ARMONICOS_CLIMATOLOGIA = 2                   # K = 2 armónicos para la climatología (Tabla 1 del PDF)
REZAGOS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]  # y(t - L)
VENTANAS_TENDENCIA = [1, 3, 7, 15, 30]       # y(t) - y(t - w)
VENTANAS_MEDIA_MOVIL = [7, 30]               # medias móviles terminadas en t
