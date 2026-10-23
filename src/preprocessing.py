"""Preprocesamiento de los niveles del Orinoco (PASO 1 común a todos los modelos).

Flujo, en este orden (para evitar *data leakage*):
1. ``leer_excel``: la columna ``Fecha`` pasa a ser un índice diario continuo.
2. Control de calidad: ``detectar_cambios_de_referencia`` busca saltos bruscos
   comparando cada estación con una vecina; ``corregir_cambios_de_referencia``
   aplica solo los saltos confirmados en la configuración.
3. ``imputar_huecos``: interpolación temporal, guardando qué días eran observados
   (las métricas se calculan solo con días observados).
4. ``ajustar_climatologia``: ciclo anual medio con armónicos, ajustado SOLO con train.
5. ``construir_tabla``: 79 variables conocidas en el día t y el objetivo
   ``cambio = y(t + h) - y(t)``.
6. ``preparar_datos``: split por fecha objetivo (train / validación / prueba) y
   escalado Z-Score ajustado SOLO con train.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.configuracion import (
    ARMONICOS_CALENDARIO, ARMONICOS_CLIMATOLOGIA, CAMBIOS_DE_REFERENCIA, CARPETA_ESCALADORES,
    ESTACIONES, ESTACIONES_OBJETIVO, FECHA_FIN_TRAIN, FECHA_FIN_VAL, HORIZONTES, REZAGOS, RUTA_DATOS,
    VECINAS_CONTROL, VENTANAS_MEDIA_MOVIL, VENTANAS_TENDENCIA,
)

# Columnas de la tabla que NO son variables de entrada (información para evaluar)
COLUMNAS_META = ["fecha_objetivo", "nivel_actual", "nivel_objetivo", "cambio", "observado"]


# ==========================================================================
# 1) CARGA
# ==========================================================================
def leer_excel(ruta=RUTA_DATOS) -> pd.DataFrame:
    """Lee el Excel y devuelve un DataFrame con índice diario continuo.

    Las fechas repetidas se eliminan (se conserva la primera) y las fechas que no
    aparecen en el archivo se añaden como filas con NaN.

    Args:
        ruta: ruta al Excel con la columna ``Fecha`` y una columna por estación.

    Returns:
        DataFrame con índice ``Fecha`` (frecuencia diaria) y una columna por estación.
    """
    tabla = pd.read_excel(ruta)
    tabla["Fecha"] = pd.to_datetime(tabla["Fecha"])
    tabla = tabla.set_index("Fecha")
    tabla = tabla.sort_index()

    fechas_repetidas = tabla.index.duplicated(keep="first")
    tabla = tabla[~fechas_repetidas]
    tabla = tabla.asfreq("D")
    return tabla[ESTACIONES]


# ==========================================================================
# 2) CONTROL DE CALIDAD
# ==========================================================================
def detectar_cambios_de_referencia(niveles: pd.DataFrame, estacion: str, vecina: str, desfase: int,
                                   ventana: int = 15, umbral: float = 0.5,
                                   maximo_hueco_dias: int = 7, veces_lo_normal: float = 5.0) -> pd.DataFrame:
    """Busca saltos bruscos (posibles cambios del cero de la regla) en una estación.

    Idea: una estación y su vecina suben y bajan juntas. Si explicamos la estación
    con una recta a partir de la vecina, el residuo cambia muy poco de un día a
    otro. Un cambio de referencia es un escalón: el residuo salta de golpe y se
    queda en el nuevo valor.

    1. Ajustar ``estacion(t) ≈ a + b * vecina(t + desfase)`` y calcular el residuo
       en los días con dato en ambas.
    2. Salto diario: diferencia entre el residuo de un día observado y el anterior.
    3. Salto sostenido: mediana de los ``ventana`` residuos siguientes menos la de
       los ``ventana`` anteriores.
    4. Salto aislado: el salto diario debe ser al menos ``veces_lo_normal`` veces la
       mediana de los saltos diarios de los días cercanos. Un cambio de regla es un
       escalón entre días tranquilos; en una crecida rápida todos los días cambian mucho.
    Se reporta el día si se cumplen las tres condiciones (un dato aislado erróneo
    no cumple la 3; una deriva lenta no cumple la 2; una crecida no cumple la 4).
    5. Además, no se evalúan saltos tras huecos de más de ``maximo_hueco_dias``: en un
       hueco largo el río cambia mucho y eso se confundiría con un escalón.

    Args:
        niveles: niveles crudos (pueden tener NaN).
        estacion: estación a revisar.
        vecina: estación de comparación.
        desfase: días que tarda la onda entre ``estacion`` y ``vecina``.
        ventana: número de días observados a cada lado para el salto sostenido.
        umbral: salto mínimo (m) para reportar.
        maximo_hueco_dias: hueco máximo entre dos días observados para comparar.
        veces_lo_normal: cuántas veces mayor que lo habitual debe ser el salto diario.

    Returns:
        DataFrame con columnas ``Estacion``, ``Fecha`` y ``Salto`` (m). Si el salto
        cae dentro de un hueco, la fecha es el primer día con dato después del hueco.
    """
    serie = niveles[estacion]
    referencia = niveles[vecina].shift(-desfase)

    ambos_observados = serie.notna() & referencia.notna()
    pendiente, intercepto = np.polyfit(referencia[ambos_observados], serie[ambos_observados], 1)
    residuo = serie - (intercepto + pendiente * referencia)
    residuo = residuo[ambos_observados]

    salto_diario = residuo.diff()
    dias_desde_anterior = residuo.index.to_series().diff().dt.days
    mediana_antes = residuo.rolling(ventana).median().shift(1)
    mediana_despues = residuo.iloc[::-1].rolling(ventana).median().iloc[::-1]
    salto_sostenido = mediana_despues - mediana_antes

    sin_hueco_largo = dias_desde_anterior <= maximo_hueco_dias

    salto_diario_habitual = salto_diario.abs().rolling(2 * ventana + 1, center=True, min_periods=ventana).median()
    # Si entre dos datos hay días sin medir, lo "habitual" se multiplica por esos días
    es_aislado = salto_diario.abs() > veces_lo_normal * salto_diario_habitual * dias_desde_anterior

    es_salto = (salto_diario.abs() > umbral) & (salto_sostenido.abs() > umbral) & es_aislado & sin_hueco_largo
    filas = []
    for fecha in residuo.index[es_salto.to_numpy()]:
        filas.append({"Estacion": estacion, "Fecha": fecha, "Salto": round(float(salto_sostenido[fecha]), 2)})
    detectados = pd.DataFrame(filas, columns=["Estacion", "Fecha", "Salto"])
    detectados["Fecha"] = pd.to_datetime(detectados["Fecha"])
    return detectados


def revisar_calidad(niveles: pd.DataFrame, tolerancia_dias: int = 7) -> pd.DataFrame:
    """Detecta saltos en todas las estaciones comparando con sus dos vecinas (``VECINAS_CONTROL``).

    Un salto queda ``Confirmado`` si se detecta frente a TODAS las vecinas de la
    estación con una diferencia de fechas de como mucho ``tolerancia_dias``.
    Los no confirmados suelen ser "reflejos" de un salto de la vecina o efectos
    de un hueco largo; se listan para revisarlos a mano.

    Returns:
        DataFrame con columnas ``Estacion``, ``Vecina``, ``Fecha``, ``Salto`` y ``Confirmado``.
    """
    filas = []
    for estacion in ESTACIONES:
        detecciones_por_vecina = []
        for vecina, desfase in VECINAS_CONTROL[estacion]:
            detectados = detectar_cambios_de_referencia(niveles, estacion, vecina, desfase)
            detectados["Vecina"] = vecina
            detecciones_por_vecina.append(detectados)

        for detectados in detecciones_por_vecina:
            for _, fila in detectados.iterrows():
                confirmado = True
                for otras in detecciones_por_vecina:
                    distancia_dias = (otras["Fecha"] - fila["Fecha"]).abs().dt.days
                    if not (distancia_dias <= tolerancia_dias).any():
                        confirmado = False
                filas.append({
                    "Estacion": estacion,
                    "Vecina": fila["Vecina"],
                    "Fecha": fila["Fecha"],
                    "Salto": fila["Salto"],
                    "Confirmado": confirmado,
                })
    return pd.DataFrame(filas, columns=["Estacion", "Vecina", "Fecha", "Salto", "Confirmado"])


def corregir_cambios_de_referencia(niveles: pd.DataFrame, cambios: list) -> pd.DataFrame:
    """Lleva cada serie a su referencia más reciente.

    Si a partir de ``fecha`` la regla marca ``salto`` metros distinto, se suma
    ``salto`` a todos los valores ANTERIORES a esa fecha. Así la serie completa
    queda en la referencia actual (la de los umbrales de alerta vigentes).

    Args:
        niveles: niveles crudos.
        cambios: lista de tuplas ``(estacion, fecha, salto)`` confirmadas.

    Returns:
        Copia corregida de ``niveles``.
    """
    corregidos = niveles.copy()
    for estacion, fecha, salto in cambios:
        antes = corregidos.index < pd.Timestamp(fecha)
        corregidos.loc[antes, estacion] = corregidos.loc[antes, estacion] + salto
    return corregidos


def imputar_huecos(niveles: pd.DataFrame) -> tuple:
    """Rellena los NaN por interpolación temporal y devuelve qué días eran observados.

    Returns:
        Tupla ``(niveles_imputados, observados)``; ``observados`` es un DataFrame
        booleano con ``True`` donde había un dato real.
    """
    observados = niveles.notna()
    imputados = niveles.interpolate(method="time", limit_direction="both")
    return imputados, observados


def cargar_datos(ruta=RUTA_DATOS) -> tuple:
    """Lee, corrige los cambios de referencia confirmados e imputa los huecos.

    Returns:
        Tupla ``(niveles, observados)`` (ver :func:`imputar_huecos`).
    """
    crudos = leer_excel(ruta)
    corregidos = corregir_cambios_de_referencia(crudos, CAMBIOS_DE_REFERENCIA)
    niveles, observados = imputar_huecos(corregidos)
    return niveles, observados


# ==========================================================================
# 3) CALENDARIO Y CLIMATOLOGÍA ARMÓNICA
# ==========================================================================
def variables_calendario(fechas: pd.DatetimeIndex, armonicos: int) -> pd.DataFrame:
    """Seno y coseno del día del año: sen(2πk·d/N), cos(2πk·d/N) para k = 1..armonicos.

    Con estas variables el 31 de diciembre y el 1 de enero quedan "cerca", algo que
    el número de día (365 frente a 1) no consigue.
    """
    dia = fechas.dayofyear.to_numpy()
    dias_del_anio = np.where(fechas.is_leap_year, 366, 365)
    calendario = pd.DataFrame(index=fechas)
    for k in range(1, armonicos + 1):
        angulo = 2 * np.pi * k * dia / dias_del_anio
        calendario["sen_" + str(k)] = np.sin(angulo)
        calendario["cos_" + str(k)] = np.cos(angulo)
    return calendario


def ajustar_climatologia(serie: pd.Series, observados: pd.Series, armonicos: int = ARMONICOS_CLIMATOLOGIA) -> np.ndarray:
    """Ajusta el ciclo anual medio (regresión armónica) usando SOLO días observados de train.

    Modelo: nivel ≈ c0 + Σ_k [a_k·sen(2πk·d/N) + b_k·cos(2πk·d/N)].

    Returns:
        Coeficientes ``[c0, a_1, b_1, ..., a_K, b_K]``.
    """
    en_train = serie.index <= pd.Timestamp(FECHA_FIN_TRAIN)
    usar = en_train & observados.to_numpy()
    matriz = variables_calendario(serie.index[usar], armonicos).to_numpy()
    matriz = np.column_stack([np.ones(len(matriz)), matriz])
    coeficientes, _, _, _ = np.linalg.lstsq(matriz, serie.to_numpy()[usar], rcond=None)
    return coeficientes


def evaluar_climatologia(coeficientes: np.ndarray, fechas: pd.DatetimeIndex) -> np.ndarray:
    """Valor de la climatología armónica en las fechas pedidas."""
    armonicos = (len(coeficientes) - 1) // 2
    matriz = variables_calendario(fechas, armonicos).to_numpy()
    matriz = np.column_stack([np.ones(len(matriz)), matriz])
    return np.dot(matriz, coeficientes)


def ajustar_climatologias(niveles: pd.DataFrame, observados: pd.DataFrame) -> dict:
    """Climatología de cada estación: ``{estacion: coeficientes}``."""
    climatologias = {}
    for estacion in ESTACIONES:
        climatologias[estacion] = ajustar_climatologia(niveles[estacion], observados[estacion])
    return climatologias


# ==========================================================================
# 4) TABLA DE VARIABLES (79) Y OBJETIVO
# ==========================================================================
def construir_tabla(niveles: pd.DataFrame, observados: pd.DataFrame, climatologias: dict,
                    estacion_objetivo: str, horizonte: int) -> pd.DataFrame:
    """Construye la tabla supervisada para pronosticar ``estacion_objetivo`` a ``horizonte`` días.

    Cada fila es un día de emisión t. Las variables solo usan datos hasta t:

    - Cíclicas (7): seno/coseno del día del año en t + h (k = 1..3) y el cambio
      que espera la climatología entre t y t + h.
    - Rezagos (40): nivel de cada estación en t - L.
    - Tendencias (20): y(t) - y(t - w) de cada estación.
    - Suavizado (12): medias móviles de 7 y 30 días y anomalía (nivel - climatología).

    Objetivo: ``cambio = y(t + h) - y(t)`` de la estación objetivo.

    Args:
        niveles: niveles imputados (sin NaN, frecuencia diaria).
        observados: máscara de días observados.
        climatologias: salida de :func:`ajustar_climatologias`.
        estacion_objetivo: estación a pronosticar.
        horizonte: días de anticipación h.

    Returns:
        DataFrame indexado por la fecha de emisión t con las variables y ``COLUMNAS_META``.
    """
    fechas = niveles.index
    fechas_objetivo = fechas + pd.Timedelta(days=horizonte)
    tabla = pd.DataFrame(index=fechas)

    # --- Cíclicas: se evalúan en la fecha objetivo (se conocen de antemano) ---
    calendario = variables_calendario(fechas_objetivo, ARMONICOS_CALENDARIO)
    for columna in calendario.columns:
        tabla["calendario_" + columna] = calendario[columna].to_numpy()
    coeficientes_objetivo = climatologias[estacion_objetivo]
    climatologia_hoy = evaluar_climatologia(coeficientes_objetivo, fechas)
    climatologia_futura = evaluar_climatologia(coeficientes_objetivo, fechas_objetivo)
    tabla["climatologia_cambio_esperado"] = climatologia_futura - climatologia_hoy

    # --- Rezagos, tendencias y suavizado de las cuatro estaciones ---
    for estacion in ESTACIONES:
        serie = niveles[estacion]
        for rezago in REZAGOS:
            tabla[estacion + "_rezago_" + str(rezago)] = serie.shift(rezago)
        for ventana in VENTANAS_TENDENCIA:
            tabla[estacion + "_tendencia_" + str(ventana)] = serie - serie.shift(ventana)
        for ventana in VENTANAS_MEDIA_MOVIL:
            tabla[estacion + "_media_" + str(ventana)] = serie.rolling(ventana).mean()
        climatologia_estacion = evaluar_climatologia(climatologias[estacion], fechas)
        tabla[estacion + "_anomalia"] = serie - climatologia_estacion

    # --- Objetivo e información para evaluar ---
    serie_objetivo = niveles[estacion_objetivo]
    observado_objetivo = observados[estacion_objetivo]
    tabla["fecha_objetivo"] = fechas_objetivo
    tabla["nivel_actual"] = serie_objetivo
    tabla["nivel_objetivo"] = serie_objetivo.shift(-horizonte)
    tabla["cambio"] = tabla["nivel_objetivo"] - tabla["nivel_actual"]
    # [JUSTIFICACIÓN]: un día imputado se rellenó mirando datos posteriores. Para que la
    # evaluación sea honesta, solo cuentan las filas con y(t) e y(t + h) observados.
    tabla["observado"] = observado_objetivo & observado_objetivo.shift(-horizonte, fill_value=False)

    # Se pierden los primeros 30 días (sin historia) y los últimos h (sin objetivo)
    tabla = tabla.dropna()
    return tabla


def nombres_variables(tabla: pd.DataFrame) -> list:
    """Nombres de las columnas que son variables de entrada (todo menos ``COLUMNAS_META``)."""
    variables = []
    for columna in tabla.columns:
        if columna not in COLUMNAS_META:
            variables.append(columna)
    return variables


def grupos_de_variables(variables: list, estacion_objetivo: str) -> dict:
    """Clasifica las variables en los grupos del análisis de contribución (Figura 8 del PDF).

    Returns:
        ``{"ciclicas": [...], "propias": [...], "otras_estaciones": [...]}``.
    """
    grupos = {"ciclicas": [], "propias": [], "otras_estaciones": []}
    for variable in variables:
        if variable.startswith("calendario_") or variable.startswith("climatologia_"):
            grupos["ciclicas"].append(variable)
        elif variable.startswith(estacion_objetivo + "_"):
            grupos["propias"].append(variable)
        else:
            grupos["otras_estaciones"].append(variable)
    return grupos


# ==========================================================================
# 5) SPLIT CRONOLÓGICO Y ESCALADO
# ==========================================================================
def dividir_por_fecha(tabla: pd.DataFrame) -> tuple:
    """Divide según la fecha objetivo: train ≤ 2014, validación 2015-2019, prueba ≥ 2020."""
    fin_train = pd.Timestamp(FECHA_FIN_TRAIN)
    fin_val = pd.Timestamp(FECHA_FIN_VAL)
    fecha_objetivo = tabla["fecha_objetivo"]

    tabla_train = tabla[fecha_objetivo <= fin_train]
    tabla_val = tabla[(fecha_objetivo > fin_train) & (fecha_objetivo <= fin_val)]
    tabla_test = tabla[fecha_objetivo > fin_val]
    return tabla_train, tabla_val, tabla_test


def preparar_datos(niveles: pd.DataFrame, observados: pd.DataFrame, estacion_objetivo: str,
                   horizonte: int) -> dict:
    """Prepara los conjuntos de una estación y un horizonte para cualquier modelo tabular.

    Returns:
        Diccionario con:
            - ``x_train``, ``x_val``, ``x_test``: variables ESCALADAS (Z-Score ajustado con train).
            - ``y_train``, ``y_val``, ``y_test``: cambio y(t + h) - y(t) en metros.
            - ``tabla_train``, ``tabla_val``, ``tabla_test``: tablas completas sin escalar
              (incluyen ``nivel_actual``, ``nivel_objetivo``, ``observado`` y ``fecha_objetivo``).
            - ``escalador``, ``variables``, ``climatologias``, ``estacion``, ``horizonte``.
    """
    climatologias = ajustar_climatologias(niveles, observados)
    tabla = construir_tabla(niveles, observados, climatologias, estacion_objetivo, horizonte)
    variables = nombres_variables(tabla)
    tabla_train, tabla_val, tabla_test = dividir_por_fecha(tabla)

    # Escalado Z-Score ajustado SOLO con train (evita data leakage)
    escalador = StandardScaler()
    escalador.fit(tabla_train[variables])
    CARPETA_ESCALADORES.mkdir(parents=True, exist_ok=True)
    nombre_archivo = estacion_objetivo + "_h" + str(horizonte) + "_zscore.joblib"
    joblib.dump(escalador, CARPETA_ESCALADORES / nombre_archivo)

    datos = {
        "estacion": estacion_objetivo,
        "horizonte": horizonte,
        "variables": variables,
        "climatologias": climatologias,
        "escalador": escalador,
        "tabla_train": tabla_train,
        "tabla_val": tabla_val,
        "tabla_test": tabla_test,
    }
    for nombre, tabla_conjunto in [("train", tabla_train), ("val", tabla_val), ("test", tabla_test)]:
        valores_escalados = escalador.transform(tabla_conjunto[variables])
        datos["x_" + nombre] = pd.DataFrame(valores_escalados, index=tabla_conjunto.index, columns=variables)
        datos["y_" + nombre] = tabla_conjunto["cambio"]
    return datos


def preparar_ventanas(niveles: pd.DataFrame, observados: pd.DataFrame, estacion_objetivo: str,
                      longitud_ventana: int, maximo_horizonte: int) -> dict:
    """Prepara ventanas de días consecutivos para las redes neuronales secuenciales.

    Cada muestra es el día de emisión t con:

    - Entrada (ventana): los ``longitud_ventana`` días que terminan en t, con 10
      canales por día: nivel y anomalía de las 4 estaciones y seno/coseno del día del año.
    - Salida: los cambios y(t + k) - y(t) de la estación objetivo para k = 1..``maximo_horizonte``
      (una sola red pronostica todos los horizontes a la vez).

    Escalado (ajustado SOLO con días de train): Z-Score por canal y por horizonte.

    Muestras para entrenar: días de emisión cuyo objetivo más lejano (t + maximo_horizonte)
    cae en train. Muestras para early stopping: las que caen completas en validación.

    Returns:
        Diccionario con ``ventanas`` (array float32 de forma (muestras, longitud_ventana, 10)),
        ``origenes`` (fechas t de cada muestra), ``cambios_escalados`` (muestras, maximo_horizonte),
        ``escalador_objetivo``, ``escalador_canales``, ``canales`` (nombres) y ``en_train`` /
        ``en_val`` (máscaras booleanas por muestra).
    """
    fin_train = pd.Timestamp(FECHA_FIN_TRAIN)
    fin_val = pd.Timestamp(FECHA_FIN_VAL)
    climatologias = ajustar_climatologias(niveles, observados)
    dia_en_train = niveles.index <= fin_train

    # --- 1) Canales diarios (todos conocidos en su propio día) ---
    canales = pd.DataFrame(index=niveles.index)
    for estacion in ESTACIONES:
        canales[estacion + "_nivel"] = niveles[estacion]
        climatologia = evaluar_climatologia(climatologias[estacion], niveles.index)
        canales[estacion + "_anomalia"] = niveles[estacion] - climatologia
    calendario = variables_calendario(niveles.index, 1)
    canales["sen_1"] = calendario["sen_1"]
    canales["cos_1"] = calendario["cos_1"]

    escalador_canales = StandardScaler()
    escalador_canales.fit(canales[dia_en_train])
    canales_escalados = escalador_canales.transform(canales).astype(np.float32)

    # --- 2) Ventanas: la muestra i contiene los días i .. i + longitud_ventana - 1 ---
    # sliding_window_view devuelve (muestras, canales, días); lo pasamos a (muestras, días, canales)
    ventanas = np.lib.stride_tricks.sliding_window_view(canales_escalados, longitud_ventana, axis=0)
    ventanas = np.ascontiguousarray(ventanas.transpose(0, 2, 1))
    origenes = niveles.index[longitud_ventana - 1:]

    # --- 3) Objetivos: cambios a 1..maximo_horizonte días ---
    serie_objetivo = niveles[estacion_objetivo]
    cambios = pd.DataFrame(index=niveles.index)
    for paso in range(1, maximo_horizonte + 1):
        cambios[paso] = serie_objetivo.shift(-paso) - serie_objetivo
    cambios = cambios.loc[origenes].to_numpy()

    fecha_mas_lejana = origenes + pd.Timedelta(days=maximo_horizonte)
    fecha_mas_cercana = origenes + pd.Timedelta(days=1)
    en_train = np.asarray(fecha_mas_lejana <= fin_train)
    en_val = np.asarray((fecha_mas_cercana > fin_train) & (fecha_mas_lejana <= fin_val))

    escalador_objetivo = StandardScaler()
    escalador_objetivo.fit(cambios[en_train])
    # Al final de la serie faltan objetivos (NaN); esas muestras solo se usan para predecir
    cambios_escalados = escalador_objetivo.transform(cambios).astype(np.float32)

    datos = {
        "ventanas": ventanas,
        "origenes": origenes,
        "cambios_escalados": cambios_escalados,
        "escalador_objetivo": escalador_objetivo,
        "escalador_canales": escalador_canales,
        "en_train": en_train,
        "en_val": en_val,
        "canales": list(canales.columns),
    }
    return datos


def cambio_a_nivel(nivel_actual, cambio_predicho) -> np.ndarray:
    """Reconstruye el nivel pronosticado: ŷ(t + h) = y(t) + cambio predicho."""
    nivel = np.asarray(nivel_actual, dtype=float) + np.asarray(cambio_predicho, dtype=float)
    return nivel


if __name__ == "__main__":
    # Prueba rápida del módulo: python -m src.preprocessing
    crudos = leer_excel()
    print("Datos crudos:", crudos.shape, "| NaN por estación:", crudos.isna().sum().to_dict())
    print("\nSaltos detectados (control de calidad):")
    print(revisar_calidad(crudos).to_string(index=False))

    niveles, observados = cargar_datos()
    print("\nNaN restantes tras imputar:", int(niveles.isna().sum().sum()))

    for estacion in ESTACIONES_OBJETIVO:
        for horizonte in HORIZONTES:
            datos = preparar_datos(niveles, observados, estacion, horizonte)
            assert datos["tabla_train"]["fecha_objetivo"].max() < datos["tabla_val"]["fecha_objetivo"].min()
            assert datos["tabla_val"]["fecha_objetivo"].max() < datos["tabla_test"]["fecha_objetivo"].min()
            print(
                estacion.ljust(15), "h =", str(horizonte).rjust(2),
                "| variables =", len(datos["variables"]),
                "| train =", len(datos["y_train"]),
                "val =", len(datos["y_val"]),
                "test =", len(datos["y_test"]),
                "(observados:", str(int(datos["tabla_test"]["observado"].sum())) + ")",
            )
    print("Preprocesamiento validado correctamente.")
