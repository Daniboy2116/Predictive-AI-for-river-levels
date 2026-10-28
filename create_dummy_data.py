"""Genera un dataset simulado de niveles del Orinoco en data/datos_rios.xlsx.

Mientras no tengamos los archivos reales, simulamos datos con el mismo formato y
con las propiedades descritas en el análisis exploratorio (PDF):

1. Ciclo anual: crecida de abril a agosto (máximo ~agosto) y recesión lenta hasta
   el estiaje (mínimo ~marzo).
2. Anomalías con memoria larga: la lluvia de la cuenca pasa por "embalses lineales"
   (suelo, acuíferos, planicies inundables) y produce variaciones suaves que duran
   semanas o meses, más un ciclo interanual tipo El Niño / La Niña.
3. Propagación aguas abajo: lo que ocurre en Ayacucho llega a Caicara ~14 días
   después y a Ciudad Bolívar ~18 días después; Ciudad Bolívar y Palúa suben a la vez.
4. Imperfecciones reales: días sin dato, huecos de varias semanas, fechas que no
   aparecen en el archivo y dos cambios de referencia en Ciudad Bolívar.

Ejecutar:  python create_dummy_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42
CARPETA_DATOS = Path(__file__).resolve().parent / "data"
RUTA_SALIDA = CARPETA_DATOS / "datos_rios.xlsx"
FECHA_INICIO = "1974-01-01"
FECHA_FIN = "2025-12-31"

# Ciclo anual medio: (nivel medio m, amplitud máx-mín m, día del año del máximo, día del mínimo)
CICLO_ANUAL = {
    "Ayacucho": (7.5, 10.9, 203, 58),
    "Caicara": (9.0, 11.8, 221, 76),
    "Ciudad_Bolivar": (9.6, 13.1, 225, 80),
    "Palua": (7.4, 9.6, 225, 80),
}

# Cambios del cero de la regla que "ensucian" la serie de Ciudad Bolívar: (fecha, salto m)
CAMBIOS_DE_REFERENCIA_SIMULADOS = [("1987-05-15", -2.20), ("2019-01-01", 1.00)]

# Días sin dato aislados (fracción) y un hueco largo en Ayacucho
FRACCION_FALTANTES = {"Ayacucho": 0.003, "Caicara": 0.002, "Ciudad_Bolivar": 0.001, "Palua": 0.001}
HUECO_LARGO = ("Ayacucho", "1993-06-01", "1993-08-31")
FECHAS_AUSENTES = 310   # días que ni siquiera aparecen como fila en el Excel


def forma_anual(dia_del_anio: np.ndarray, dia_maximo: int, dia_minimo: int) -> np.ndarray:
    """Hidrograma anual normalizado entre -1 (estiaje) y +1 (crecida).

    La rama de crecida (del mínimo al máximo) y la de recesión (del máximo al
    mínimo del año siguiente) son medios cosenos de distinta duración, por eso
    el río sube más rápido de lo que baja.
    """
    duracion_crecida = dia_maximo - dia_minimo
    duracion_recesion = 365.25 - duracion_crecida

    dias_desde_minimo = np.mod(dia_del_anio - dia_minimo, 365.25)
    dias_desde_maximo = np.mod(dia_del_anio - dia_maximo, 365.25)

    rama_crecida = -np.cos(np.pi * dias_desde_minimo / duracion_crecida)
    rama_recesion = np.cos(np.pi * dias_desde_maximo / duracion_recesion)

    en_crecida = dias_desde_minimo <= duracion_crecida
    forma = np.where(en_crecida, rama_crecida, rama_recesion)
    return forma


def embalse_lineal(entrada: np.ndarray, tiempo_residencia: float) -> np.ndarray:
    """Embalse lineal: la salida se vacía poco a poco (memoria exponencial).

    salida(t) = phi * salida(t-1) + (1 - phi) * entrada(t),  con phi = exp(-1 / tiempo_residencia)
    """
    phi = np.exp(-1.0 / tiempo_residencia)
    salida = np.zeros(len(entrada))
    salida[0] = entrada[0]
    for i in range(1, len(entrada)):
        salida[i] = phi * salida[i - 1] + (1.0 - phi) * entrada[i]
    return salida


def quitar_ciclo_y_estandarizar(serie: np.ndarray, dia_del_anio: np.ndarray, desviacion: float) -> np.ndarray:
    """Resta el promedio de cada día del año y reescala a la desviación pedida (anomalía)."""
    tabla = pd.DataFrame({"valor": serie, "dia": dia_del_anio})
    media_por_dia = tabla.groupby("dia")["valor"].transform("mean")
    anomalia = serie - media_por_dia.to_numpy()
    anomalia = anomalia / anomalia.std() * desviacion
    return anomalia


def transito(serie: np.ndarray, desfase: float, dispersion: float) -> np.ndarray:
    """Traslada una onda aguas abajo: retraso de `desfase` días y suavizado gaussiano.

    Cada valor de salida es un promedio ponderado de la serie entre t-desfase-3σ y
    t-desfase+3σ (solo pasado), así la onda llega más tarde y algo más aplanada.
    """
    retardos = np.arange(0, int(desfase + 4 * dispersion) + 1)
    pesos = np.exp(-0.5 * ((retardos - desfase) / dispersion) ** 2)
    pesos = pesos / pesos.sum()
    salida = np.convolve(serie, pesos)[:len(serie)]
    # Los primeros días no tienen suficiente pasado; repetimos el primer valor válido
    salida[:len(retardos)] = salida[len(retardos)]
    return salida


def lluvia_simulada(fechas: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Eventos de lluvia diarios: más frecuentes en la temporada húmeda (mayo-agosto)."""
    dia_del_anio = fechas.dayofyear.to_numpy()
    temporada_humeda = 0.5 + 0.5 * np.cos(2 * np.pi * (dia_del_anio - 170) / 365.25)
    probabilidad = 0.03 + 0.25 * temporada_humeda
    hay_evento = rng.random(len(fechas)) < probabilidad
    intensidad = rng.exponential(1.0, len(fechas))
    return hay_evento * intensidad


def simular_niveles(rng: np.random.Generator) -> pd.DataFrame:
    """Simula los niveles reales (sin errores de medición) de las cuatro estaciones."""
    fechas = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")
    dia_del_anio = fechas.dayofyear.to_numpy()

    # --- 1) Fuentes de variabilidad (anomalías en metros) ---
    # Adelanto o retraso de la temporada de lluvias (en días): cambia de un año a otro
    ruido_temporada = rng.normal(0, 1, len(fechas))
    desfase_temporada = embalse_lineal(embalse_lineal(ruido_temporada, 90), 90)
    desfase_temporada = desfase_temporada / desfase_temporada.std() * 7.0

    # Cuenca alta del Orinoco: lluvia filtrada por dos embalses (rápido + lento)
    lluvia_alta = lluvia_simulada(fechas, rng)
    cuenca_alta = embalse_lineal(embalse_lineal(lluvia_alta, 10), 60)
    cuenca_alta = quitar_ciclo_y_estandarizar(cuenca_alta, dia_del_anio, 0.7)

    # Ciclo interanual tipo El Niño / La Niña: ruido muy suavizado
    ruido_lento = rng.normal(0, 1, len(fechas))
    enos = embalse_lineal(embalse_lineal(ruido_lento, 180), 180)
    enos = quitar_ciclo_y_estandarizar(enos, dia_del_anio, 0.5)

    # Afluentes: Apure (llega entre Ayacucho y Caicara), Caura y Caroní (regulado por Guri)
    lluvia_apure = 0.5 * lluvia_alta + 0.5 * lluvia_simulada(fechas, rng)
    apure = quitar_ciclo_y_estandarizar(embalse_lineal(embalse_lineal(lluvia_apure, 6), 35),
                                        dia_del_anio, 0.7)
    caura = quitar_ciclo_y_estandarizar(embalse_lineal(embalse_lineal(lluvia_simulada(fechas, rng), 15), 40),
                                        dia_del_anio, 1.0)
    caroni = quitar_ciclo_y_estandarizar(embalse_lineal(embalse_lineal(rng.normal(0, 1, len(fechas)), 45), 45),
                                         dia_del_anio, 0.5)

    # --- 2) Propagación aguas abajo (del desfase de la temporada y de las anomalías) ---
    desfases = {}
    desfases["Ayacucho"] = desfase_temporada
    desfases["Caicara"] = transito(desfase_temporada, 14, 3)
    desfases["Ciudad_Bolivar"] = transito(desfase_temporada, 18, 3)
    desfases["Palua"] = desfases["Ciudad_Bolivar"]

    anomalias = {}
    anomalias["Ayacucho"] = cuenca_alta + enos
    anomalias["Caicara"] = 0.8 * transito(anomalias["Ayacucho"], 14, 3) + 0.8 * transito(apure, 4, 2)
    anomalias["Ciudad_Bolivar"] = 0.95 * transito(anomalias["Caicara"], 4, 1.5) + caura
    anomalias["Palua"] = 0.72 * anomalias["Ciudad_Bolivar"] + caroni

    # --- 3) Nivel = ciclo anual (desplazado) + anomalía (más pequeña en aguas bajas) ---
    niveles = pd.DataFrame(index=fechas)
    for estacion, (media, amplitud, dia_maximo, dia_minimo) in CICLO_ANUAL.items():
        dia_desplazado = dia_del_anio - desfases[estacion]
        forma = forma_anual(dia_desplazado, dia_maximo, dia_minimo)
        ciclo = media + 0.5 * amplitud * forma
        anomalia = anomalias[estacion] * (0.8 + 0.2 * forma)
        ruido_medicion = rng.normal(0, 0.015, len(fechas))
        nivel = ciclo + anomalia + ruido_medicion
        niveles[estacion] = np.clip(nivel, 0.1, None)
    return niveles


def ensuciar_datos(niveles: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Añade los problemas típicos de un registro real: saltos de referencia y huecos."""
    observados = niveles.copy()

    # Cambios de referencia: a partir de la fecha la regla marca `salto` metros distinto
    for fecha, salto in CAMBIOS_DE_REFERENCIA_SIMULADOS:
        despues = observados.index >= pd.Timestamp(fecha)
        observados.loc[despues, "Ciudad_Bolivar"] = observados.loc[despues, "Ciudad_Bolivar"] + salto

    # Días sin dato aislados
    for estacion, fraccion in FRACCION_FALTANTES.items():
        faltan = rng.random(len(observados)) < fraccion
        observados.loc[faltan, estacion] = np.nan

    # Tres huecos de 5 a 30 días por estación
    for estacion in observados.columns:
        for _ in range(3):
            inicio = rng.integers(0, len(observados) - 30)
            duracion = rng.integers(5, 31)
            observados.iloc[inicio:inicio + duracion, observados.columns.get_loc(estacion)] = np.nan

    # Hueco largo (todo un trimestre) en una estación
    estacion, inicio, fin = HUECO_LARGO
    observados.loc[inicio:fin, estacion] = np.nan

    # Fechas que no aparecen en el archivo (filas completas ausentes)
    posiciones = rng.choice(len(observados), size=FECHAS_AUSENTES, replace=False)
    observados = observados.drop(observados.index[posiciones])

    observados = observados.round(2)
    return observados


def generar() -> pd.DataFrame:
    """Construye el DataFrame final con la columna Fecha y una columna por estación."""
    rng = np.random.default_rng(SEMILLA)
    niveles = simular_niveles(rng)
    observados = ensuciar_datos(niveles, rng)
    tabla = observados.reset_index().rename(columns={"index": "Fecha"})
    return tabla


def main() -> None:
    """Genera y guarda el Excel simulado."""
    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)
    tabla = generar()
    tabla.to_excel(RUTA_SALIDA, index=False, sheet_name="Niveles")
    print("Dataset generado:", RUTA_SALIDA, "->", len(tabla), "filas, columnas", list(tabla.columns))
    print("Valores faltantes por estación:")
    print(tabla.isna().sum().to_string())


if __name__ == "__main__":
    main()
