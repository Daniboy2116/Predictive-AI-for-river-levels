"""Preprocesamiento de las series de niveles de ríos (PASO 1 común a todos los modelos).

Flujo, en este orden (para evitar *data leakage*):
1. Cargar el Excel y usar ``Fecha`` como índice diario.
2. Rellenar los valores faltantes con interpolación temporal.
3. Dividir la serie cronológicamente: 70% train, 15% val, 15% test.
4. Ajustar el escalador Min-Max SOLO con train y transformar toda la serie.
5. Crear las columnas de retardos (lags) t-1 ... t-7 y separar X / y.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------------------------------
# Configuración general del proyecto
# --------------------------------------------------------------------------
CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = CARPETA_PROYECTO / "data" / "datos_rios.xlsx"
CARPETA_ESCALADORES = CARPETA_PROYECTO / "outputs" / "scalers"

ESTACIONES = ["Palua", "Orinoco", "Caicara", "Ciudad_Bolivar", "Ayacucho"]
SEMILLA = 42            # semilla fija para que los resultados sean reproducibles
NUMERO_DE_LAGS = 7      # usamos los niveles de los 7 días anteriores
PORCENTAJE_TRAIN = 0.70
PORCENTAJE_VAL = 0.15   # el 15% restante es test


def cargar_datos(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Lee el Excel y devuelve un DataFrame diario, indexado por fecha y sin NaN.

    Args:
        ruta: ruta al archivo Excel con la columna ``Fecha`` y una columna por estación.

    Returns:
        DataFrame con índice de fechas (frecuencia diaria) y niveles en metros.
    """
    df = pd.read_excel(ruta)

    # La columna Fecha pasa a ser el índice (tipo datetime)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.set_index("Fecha")
    df = df.sort_index()

    # Eliminamos fechas repetidas y forzamos frecuencia diaria
    # (si falta algún día, aparece como una fila con NaN)
    fechas_repetidas = df.index.duplicated(keep="first")
    df = df[~fechas_repetidas]
    df = df.asfreq("D")

    # Rellenamos los huecos interpolando según la distancia en el tiempo.
    # limit_direction="both" también rellena posibles NaN al principio o al final.
    df = df.interpolate(method="time", limit_direction="both")
    return df


def crear_lags(serie: pd.Series, numero_de_lags: int = NUMERO_DE_LAGS) -> pd.DataFrame:
    """Convierte una serie en una tabla supervisada: ``objetivo`` + ``lag_1`` ... ``lag_n``.

    Ejemplo: para el día t, ``lag_1`` es el nivel de t-1, ``lag_2`` el de t-2, etc.
    El ``objetivo`` es el nivel del propio día t (lo que queremos predecir).

    Args:
        serie: serie temporal (puede estar escalada).
        numero_de_lags: cuántos días anteriores usar como entradas.

    Returns:
        DataFrame sin filas incompletas (se pierden los primeros ``numero_de_lags`` días).
    """
    tabla = pd.DataFrame(index=serie.index)
    tabla["objetivo"] = serie

    for k in range(1, numero_de_lags + 1):
        tabla["lag_" + str(k)] = serie.shift(k)

    tabla = tabla.dropna()
    return tabla


def preparar_estacion(df: pd.DataFrame, estacion: str) -> dict:
    """Prepara los datos de una estación para entrenar cualquier modelo.

    Args:
        df: DataFrame devuelto por :func:`cargar_datos`.
        estacion: nombre de la columna de la estación (ej. ``"Palua"``).

    Returns:
        Diccionario con:
            - ``serie``: serie completa en metros (sin escalar).
            - ``escalador``: MinMaxScaler ajustado solo con train.
            - ``x_train``, ``y_train``, ``x_val``, ``y_val``, ``x_test``, ``y_test``:
              lags y objetivo ESCALADOS (valores entre ~0 y 1).
            - ``fechas_test``: fechas del conjunto de prueba.
            - ``y_test_metros``: valores reales de test en metros (para evaluar).
            - ``fecha_fin_train``, ``fecha_fin_val``: fechas de corte del split.
    """
    serie = df[estacion]

    # --- 1) Puntos de corte cronológicos 70 / 15 / 15 (NUNCA aleatorio) ---
    total_dias = len(serie)
    indice_fin_train = int(total_dias * PORCENTAJE_TRAIN)
    indice_fin_val = int(total_dias * (PORCENTAJE_TRAIN + PORCENTAJE_VAL))
    fecha_fin_train = serie.index[indice_fin_train - 1]
    fecha_fin_val = serie.index[indice_fin_val - 1]

    # --- 2) Escalado Min-Max ajustado SOLO con train (evita data leakage) ---
    valores_train = serie.iloc[:indice_fin_train].to_numpy().reshape(-1, 1)
    escalador = MinMaxScaler()
    escalador.fit(valores_train)

    valores_escalados = escalador.transform(serie.to_numpy().reshape(-1, 1))
    serie_escalada = pd.Series(valores_escalados.ravel(), index=serie.index)

    # Guardamos el escalador para poder desnormalizar más tarde (por ejemplo en la app)
    CARPETA_ESCALADORES.mkdir(parents=True, exist_ok=True)
    joblib.dump(escalador, CARPETA_ESCALADORES / (estacion + "_minmax.joblib"))

    # --- 3) Tabla supervisada con lags ---
    tabla = crear_lags(serie_escalada)
    columnas_lags = []
    for k in range(1, NUMERO_DE_LAGS + 1):
        columnas_lags.append("lag_" + str(k))

    # --- 4) Separar train / val / test según las fechas de corte ---
    tabla_train = tabla[tabla.index <= fecha_fin_train]
    tabla_val = tabla[(tabla.index > fecha_fin_train) & (tabla.index <= fecha_fin_val)]
    tabla_test = tabla[tabla.index > fecha_fin_val]

    datos = {
        "estacion": estacion,
        "serie": serie,
        "escalador": escalador,
        "x_train": tabla_train[columnas_lags],
        "y_train": tabla_train["objetivo"],
        "x_val": tabla_val[columnas_lags],
        "y_val": tabla_val["objetivo"],
        "x_test": tabla_test[columnas_lags],
        "y_test": tabla_test["objetivo"],
        "fechas_test": tabla_test.index,
        "y_test_metros": serie.loc[tabla_test.index].to_numpy(),
        "fecha_fin_train": fecha_fin_train,
        "fecha_fin_val": fecha_fin_val,
    }
    return datos


def invertir_escala(escalador: MinMaxScaler, valores_escalados) -> np.ndarray:
    """Convierte valores escalados (0-1) de vuelta a metros.

    Args:
        escalador: el MinMaxScaler usado en :func:`preparar_estacion`.
        valores_escalados: array o lista con predicciones escaladas.

    Returns:
        Array 1D con los valores en metros.
    """
    valores_2d = np.asarray(valores_escalados, dtype=float).reshape(-1, 1)
    valores_metros = escalador.inverse_transform(valores_2d)
    return valores_metros.ravel()


if __name__ == "__main__":
    # Prueba rápida del módulo: python -m src.preprocessing
    df = cargar_datos()
    print("Datos cargados:", df.shape, "| NaN restantes:", int(df.isna().sum().sum()))

    for estacion in ESTACIONES:
        datos = preparar_estacion(df, estacion)

        # Comprobaciones: el orden temporal se respeta y la escala se invierte bien
        assert datos["y_train"].index.max() < datos["y_val"].index.min()
        assert datos["y_val"].index.max() < datos["y_test"].index.min()
        recuperado = invertir_escala(datos["escalador"], datos["y_test"].to_numpy())
        assert np.allclose(recuperado, datos["y_test_metros"])

        print(
            estacion.ljust(15),
            "train =", len(datos["y_train"]),
            "val =", len(datos["y_val"]),
            "test =", len(datos["y_test"]),
        )
    print("Preprocesamiento validado correctamente.")
