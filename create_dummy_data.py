from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42
CARPETA_DATOS = Path(__file__).resolve().parent / "data"
RUTA_SALIDA = CARPETA_DATOS / "datos_rios.xlsx"
FECHA_INICIO = "1974-01-01"
FECHA_FIN = "2025-12-31"

CICLO_ANUAL = {
    "Ayacucho": (7.5, 10.9, 203, 58),
    "Caicara": (9.0, 11.8, 221, 76),
    "Ciudad_Bolivar": (9.6, 13.1, 225, 80),
    "Palua": (7.4, 9.6, 225, 80),
}

CAMBIOS_DE_REFERENCIA_SIMULADOS = [("1987-05-15", -2.20), ("2019-01-01", 1.00)]

FRACCION_FALTANTES = {"Ayacucho": 0.003, "Caicara": 0.002, "Ciudad_Bolivar": 0.001, "Palua": 0.001}
HUECO_LARGO = ("Ayacucho", "1993-06-01", "1993-08-31")
FECHAS_AUSENTES = 310   # días que ni siquiera aparecen como fila en el Excel


def forma_anual(dia_del_anio: np.ndarray, dia_maximo: int, dia_minimo: int) -> np.ndarray:
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
    phi = np.exp(-1.0 / tiempo_residencia)
    salida = np.zeros(len(entrada))
    salida[0] = entrada[0]
    for i in range(1, len(entrada)):
        salida[i] = phi * salida[i - 1] + (1.0 - phi) * entrada[i]
    return salida


def quitar_ciclo_y_estandarizar(serie: np.ndarray, dia_del_anio: np.ndarray, desviacion: float) -> np.ndarray:
    tabla = pd.DataFrame({"valor": serie, "dia": dia_del_anio})
    media_por_dia = tabla.groupby("dia")["valor"].transform("mean")
    anomalia = serie - media_por_dia.to_numpy()
    anomalia = anomalia / anomalia.std() * desviacion
    return anomalia


def transito(serie: np.ndarray, desfase: float, dispersion: float) -> np.ndarray:
    retardos = np.arange(0, int(desfase + 4 * dispersion) + 1)
    pesos = np.exp(-0.5 * ((retardos - desfase) / dispersion) ** 2)
    pesos = pesos / pesos.sum()
    salida = np.convolve(serie, pesos)[:len(serie)]
    salida[:len(retardos)] = salida[len(retardos)]
    return salida


def lluvia_simulada(fechas: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    dia_del_anio = fechas.dayofyear.to_numpy()
    temporada_humeda = 0.5 + 0.5 * np.cos(2 * np.pi * (dia_del_anio - 170) / 365.25)
    probabilidad = 0.03 + 0.25 * temporada_humeda
    hay_evento = rng.random(len(fechas)) < probabilidad
    intensidad = rng.exponential(1.0, len(fechas))
    return hay_evento * intensidad


def simular_niveles(rng: np.random.Generator) -> pd.DataFrame:
    fechas = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")
    dia_del_anio = fechas.dayofyear.to_numpy()

    ruido_temporada = rng.normal(0, 1, len(fechas))
    desfase_temporada = embalse_lineal(embalse_lineal(ruido_temporada, 90), 90)
    desfase_temporada = desfase_temporada / desfase_temporada.std() * 7.0

    lluvia_alta = lluvia_simulada(fechas, rng)
    cuenca_alta = embalse_lineal(embalse_lineal(lluvia_alta, 10), 60)
    cuenca_alta = quitar_ciclo_y_estandarizar(cuenca_alta, dia_del_anio, 0.7)

    ruido_lento = rng.normal(0, 1, len(fechas))
    enos = embalse_lineal(embalse_lineal(ruido_lento, 180), 180)
    enos = quitar_ciclo_y_estandarizar(enos, dia_del_anio, 0.5)

    lluvia_apure = 0.5 * lluvia_alta + 0.5 * lluvia_simulada(fechas, rng)
    apure = quitar_ciclo_y_estandarizar(embalse_lineal(embalse_lineal(lluvia_apure, 6), 35),
                                        dia_del_anio, 0.7)
    caura = quitar_ciclo_y_estandarizar(embalse_lineal(embalse_lineal(lluvia_simulada(fechas, rng), 15), 40),
                                        dia_del_anio, 1.0)
    caroni = quitar_ciclo_y_estandarizar(embalse_lineal(embalse_lineal(rng.normal(0, 1, len(fechas)), 45), 45),
                                         dia_del_anio, 0.5)

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
    observados = niveles.copy()

    for fecha, salto in CAMBIOS_DE_REFERENCIA_SIMULADOS:
        despues = observados.index >= pd.Timestamp(fecha)
        observados.loc[despues, "Ciudad_Bolivar"] = observados.loc[despues, "Ciudad_Bolivar"] + salto

    for estacion, fraccion in FRACCION_FALTANTES.items():
        faltan = rng.random(len(observados)) < fraccion
        observados.loc[faltan, estacion] = np.nan

    for estacion in observados.columns:
        for _ in range(3):
            inicio = rng.integers(0, len(observados) - 30)
            duracion = rng.integers(5, 31)
            observados.iloc[inicio:inicio + duracion, observados.columns.get_loc(estacion)] = np.nan

    estacion, inicio, fin = HUECO_LARGO
    observados.loc[inicio:fin, estacion] = np.nan

    posiciones = rng.choice(len(observados), size=FECHAS_AUSENTES, replace=False)
    observados = observados.drop(observados.index[posiciones])

    observados = observados.round(2)
    return observados


def generar() -> pd.DataFrame:
    rng = np.random.default_rng(SEMILLA)
    niveles = simular_niveles(rng)
    observados = ensuciar_datos(niveles, rng)
    tabla = observados.reset_index().rename(columns={"index": "Fecha"})
    return tabla


def main() -> None:
    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)
    tabla = generar()
    tabla.to_excel(RUTA_SALIDA, index=False, sheet_name="Niveles")
    print("Dataset generado:", RUTA_SALIDA, "->", len(tabla), "filas, columnas", list(tabla.columns))
    print("Valores faltantes por estación:")
    print(tabla.isna().sum().to_string())


if __name__ == "__main__":
    main()
