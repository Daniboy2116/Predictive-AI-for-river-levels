import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.configuracion import (
    ARMONICOS_CALENDARIO, ARMONICOS_CLIMATOLOGIA, CAMBIOS_DE_REFERENCIA, CARPETA_ESCALADORES, COLUMNAS_EXCEL,
    ESTACIONES, ESTACIONES_OBJETIVO, FECHA_FIN_TRAIN, FECHA_FIN_VAL, HOJA_DATOS, HORIZONTES, REZAGOS, RUTA_DATOS,
    VECINAS_CONTROL, VENTANAS_MEDIA_MOVIL, VENTANAS_TENDENCIA,
)

COLUMNAS_META = ["fecha_objetivo", "nivel_actual", "nivel_objetivo", "cambio", "observado"]


def leer_excel(ruta=RUTA_DATOS, hoja=HOJA_DATOS, columnas=COLUMNAS_EXCEL) -> pd.DataFrame:
    tabla = pd.read_excel(ruta, sheet_name=hoja, usecols=["Fecha"] + list(columnas))
    tabla = tabla.rename(columns=columnas)
    tabla["Fecha"] = pd.to_datetime(tabla["Fecha"])
    tabla = tabla.set_index("Fecha")
    tabla = tabla.sort_index()

    fechas_repetidas = tabla.index.duplicated(keep="first")
    tabla = tabla[~fechas_repetidas]
    tabla = tabla.asfreq("D")
    tabla = tabla.loc[:tabla.last_valid_index()]
    return tabla[ESTACIONES]


def detectar_cambios_de_referencia(niveles: pd.DataFrame, estacion: str, vecina: str, desfase: int,
                                   ventana: int = 15, umbral: float = 0.5,
                                   maximo_hueco_dias: int = 7, veces_lo_normal: float = 5.0) -> pd.DataFrame:
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
    es_aislado = salto_diario.abs() > veces_lo_normal * salto_diario_habitual * dias_desde_anterior

    es_salto = (salto_diario.abs() > umbral) & (salto_sostenido.abs() > umbral) & es_aislado & sin_hueco_largo
    filas = []
    for fecha in residuo.index[es_salto.to_numpy()]:
        filas.append({"Estacion": estacion, "Fecha": fecha, "Salto": round(float(salto_sostenido[fecha]), 2)})
    detectados = pd.DataFrame(filas, columns=["Estacion", "Fecha", "Salto"])
    detectados["Fecha"] = pd.to_datetime(detectados["Fecha"])
    return detectados


def revisar_calidad(niveles: pd.DataFrame, tolerancia_dias: int = 7) -> pd.DataFrame:
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
    corregidos = niveles.copy()
    for estacion, fecha, salto in cambios:
        antes = corregidos.index < pd.Timestamp(fecha)
        corregidos.loc[antes, estacion] = corregidos.loc[antes, estacion] + salto
    return corregidos


def imputar_huecos(niveles: pd.DataFrame) -> tuple:
    observados = niveles.notna()
    imputados = niveles.interpolate(method="time", limit_direction="both")
    return imputados, observados


def cargar_datos(ruta=RUTA_DATOS, hoja=HOJA_DATOS, columnas=COLUMNAS_EXCEL) -> tuple:
    crudos = leer_excel(ruta, hoja, columnas)
    corregidos = corregir_cambios_de_referencia(crudos, CAMBIOS_DE_REFERENCIA)
    niveles, observados = imputar_huecos(corregidos)
    return niveles, observados


def variables_calendario(fechas: pd.DatetimeIndex, armonicos: int) -> pd.DataFrame:
    dia = fechas.dayofyear.to_numpy()
    dias_del_anio = np.where(fechas.is_leap_year, 366, 365)
    calendario = pd.DataFrame(index=fechas)
    for k in range(1, armonicos + 1):
        angulo = 2 * np.pi * k * dia / dias_del_anio
        calendario["sen_" + str(k)] = np.sin(angulo)
        calendario["cos_" + str(k)] = np.cos(angulo)
    return calendario


def ajustar_climatologia(serie: pd.Series, observados: pd.Series, armonicos: int = ARMONICOS_CLIMATOLOGIA) -> np.ndarray:
    en_train = serie.index <= pd.Timestamp(FECHA_FIN_TRAIN)
    usar = en_train & observados.to_numpy()
    matriz = variables_calendario(serie.index[usar], armonicos).to_numpy()
    matriz = np.column_stack([np.ones(len(matriz)), matriz])
    coeficientes, _, _, _ = np.linalg.lstsq(matriz, serie.to_numpy()[usar], rcond=None)
    return coeficientes


def evaluar_climatologia(coeficientes: np.ndarray, fechas: pd.DatetimeIndex) -> np.ndarray:
    armonicos = (len(coeficientes) - 1) // 2
    matriz = variables_calendario(fechas, armonicos).to_numpy()
    matriz = np.column_stack([np.ones(len(matriz)), matriz])
    return np.dot(matriz, coeficientes)


def ajustar_climatologias(niveles: pd.DataFrame, observados: pd.DataFrame) -> dict:
    climatologias = {}
    for estacion in ESTACIONES:
        climatologias[estacion] = ajustar_climatologia(niveles[estacion], observados[estacion])
    return climatologias


def construir_tabla(niveles: pd.DataFrame, observados: pd.DataFrame, climatologias: dict,
                    estacion_objetivo: str, horizonte: int) -> pd.DataFrame:
    fechas = niveles.index
    fechas_objetivo = fechas + pd.Timedelta(days=horizonte)
    tabla = pd.DataFrame(index=fechas)

    calendario = variables_calendario(fechas_objetivo, ARMONICOS_CALENDARIO)
    for columna in calendario.columns:
        tabla["calendario_" + columna] = calendario[columna].to_numpy()
    coeficientes_objetivo = climatologias[estacion_objetivo]
    climatologia_hoy = evaluar_climatologia(coeficientes_objetivo, fechas)
    climatologia_futura = evaluar_climatologia(coeficientes_objetivo, fechas_objetivo)
    tabla["climatologia_cambio_esperado"] = climatologia_futura - climatologia_hoy

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

    serie_objetivo = niveles[estacion_objetivo]
    observado_objetivo = observados[estacion_objetivo]
    tabla["fecha_objetivo"] = fechas_objetivo
    tabla["nivel_actual"] = serie_objetivo
    tabla["nivel_objetivo"] = serie_objetivo.shift(-horizonte)
    tabla["cambio"] = tabla["nivel_objetivo"] - tabla["nivel_actual"]
    tabla["observado"] = observado_objetivo & observado_objetivo.shift(-horizonte, fill_value=False)

    tabla = tabla.dropna(subset=nombres_variables(tabla))
    return tabla


def nombres_variables(tabla: pd.DataFrame) -> list:
    variables = []
    for columna in tabla.columns:
        if columna not in COLUMNAS_META:
            variables.append(columna)
    return variables


def grupos_de_variables(variables: list, estacion_objetivo: str) -> dict:
    grupos = {"ciclicas": [], "propias": [], "otras_estaciones": []}
    for variable in variables:
        if variable.startswith("calendario_") or variable.startswith("climatologia_"):
            grupos["ciclicas"].append(variable)
        elif variable.startswith(estacion_objetivo + "_"):
            grupos["propias"].append(variable)
        else:
            grupos["otras_estaciones"].append(variable)
    return grupos


def dividir_por_fecha(tabla: pd.DataFrame) -> tuple:
    fin_train = pd.Timestamp(FECHA_FIN_TRAIN)
    fin_val = pd.Timestamp(FECHA_FIN_VAL)
    fecha_objetivo = tabla["fecha_objetivo"]

    tabla_train = tabla[fecha_objetivo <= fin_train]
    tabla_val = tabla[(fecha_objetivo > fin_train) & (fecha_objetivo <= fin_val)]
    tabla_test = tabla[fecha_objetivo > fin_val]
    return tabla_train, tabla_val, tabla_test


def preparar_datos(niveles: pd.DataFrame, observados: pd.DataFrame, estacion_objetivo: str,
                   horizonte: int) -> dict:
    climatologias = ajustar_climatologias(niveles, observados)
    tabla = construir_tabla(niveles, observados, climatologias, estacion_objetivo, horizonte)
    variables = nombres_variables(tabla)
    sin_objetivo = tabla["nivel_objetivo"].isna()
    tabla_futuro = tabla[sin_objetivo]
    tabla_train, tabla_val, tabla_test = dividir_por_fecha(tabla[~sin_objetivo])

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
        "tabla_futuro": tabla_futuro,
    }
    conjuntos = [("train", tabla_train), ("val", tabla_val), ("test", tabla_test), ("futuro", tabla_futuro)]
    for nombre, tabla_conjunto in conjuntos:
        valores_escalados = escalador.transform(tabla_conjunto[variables])
        datos["x_" + nombre] = pd.DataFrame(valores_escalados, index=tabla_conjunto.index, columns=variables)
        datos["y_" + nombre] = tabla_conjunto["cambio"]
    return datos


def preparar_ventanas(niveles: pd.DataFrame, observados: pd.DataFrame, estacion_objetivo: str,
                      longitud_ventana: int, maximo_horizonte: int) -> dict:
    fin_train = pd.Timestamp(FECHA_FIN_TRAIN)
    fin_val = pd.Timestamp(FECHA_FIN_VAL)
    climatologias = ajustar_climatologias(niveles, observados)
    dia_en_train = niveles.index <= fin_train

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

    ventanas = np.lib.stride_tricks.sliding_window_view(canales_escalados, longitud_ventana, axis=0)
    ventanas = np.ascontiguousarray(ventanas.transpose(0, 2, 1))
    origenes = niveles.index[longitud_ventana - 1:]

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
    return np.asarray(nivel_actual, dtype=float) + np.asarray(cambio_predicho, dtype=float)


def preparar_datos_backtest(
    niveles: pd.DataFrame,
    observados: pd.DataFrame,
    estacion_objetivo: str,
    inicio_holdout: pd.Timestamp,
    horizonte: int,
) -> dict:
    """Prepara variables tabulares para backtesting ajustando climatología y escalador solo con train."""
    corte_train = pd.Timestamp(inicio_holdout)
    dia_en_train = niveles.index < corte_train

    # Ajustar climatología SOLO con datos estrictamente anteriores a inicio_holdout
    climatologias = {}
    for est in ESTACIONES:
        usar = dia_en_train & observados[est].to_numpy()
        matriz = variables_calendario(niveles.index[usar], ARMONICOS_CLIMATOLOGIA).to_numpy()
        matriz = np.column_stack([np.ones(len(matriz)), matriz])
        coefs, _, _, _ = np.linalg.lstsq(matriz, niveles[est].to_numpy()[usar], rcond=None)
        climatologias[est] = coefs

    tabla = construir_tabla(niveles, observados, climatologias, estacion_objetivo, horizonte)
    variables = nombres_variables(tabla)

    sin_objetivo = tabla["nivel_objetivo"].isna()
    tabla_con_obj = tabla[~sin_objetivo]
    tabla_train = tabla_con_obj[tabla_con_obj["fecha_objetivo"] < corte_train]
    tabla_holdout = tabla_con_obj[tabla_con_obj["fecha_objetivo"] >= corte_train]

    escalador = StandardScaler()
    escalador.fit(tabla_train[variables])

    x_train = pd.DataFrame(escalador.transform(tabla_train[variables]), index=tabla_train.index, columns=variables)
    x_holdout = pd.DataFrame(escalador.transform(tabla_holdout[variables]), index=tabla_holdout.index, columns=variables)

    return {
        "estacion": estacion_objetivo,
        "horizonte": horizonte,
        "variables": variables,
        "climatologias": climatologias,
        "escalador": escalador,
        "tabla_train": tabla_train,
        "tabla_holdout": tabla_holdout,
        "tabla_completa": tabla,
        "x_train": x_train,
        "y_train": tabla_train["cambio"],
        "x_holdout": x_holdout,
        "y_holdout": tabla_holdout["cambio"],
    }


def preparar_ventanas_backtest(
    niveles: pd.DataFrame,
    observados: pd.DataFrame,
    estacion_objetivo: str,
    longitud_ventana: int,
    inicio_holdout: pd.Timestamp,
    maximo_horizonte: int,
) -> dict:
    """Prepara tensores de ventanas para redes neuronales garantizando cero fuga del holdout."""
    corte_train = pd.Timestamp(inicio_holdout)
    dia_en_train = niveles.index < corte_train

    climatologias = {}
    for est in ESTACIONES:
        usar = dia_en_train & observados[est].to_numpy()
        matriz = variables_calendario(niveles.index[usar], ARMONICOS_CLIMATOLOGIA).to_numpy()
        matriz = np.column_stack([np.ones(len(matriz)), matriz])
        coefs, _, _, _ = np.linalg.lstsq(matriz, niveles[est].to_numpy()[usar], rcond=None)
        climatologias[est] = coefs

    canales = pd.DataFrame(index=niveles.index)
    for estacion in ESTACIONES:
        canales[estacion + "_nivel"] = niveles[estacion]
        climatologia = evaluar_climatologia(climatologias[estacion], niveles.index)
        canales[estacion + "_anomalia"] = niveles[estacion] - climatologia
    calendario = variables_calendario(niveles.index, 1)
    canales["sen_1"] = calendario["sen_1"]
    canales["cos_1"] = calendario["cos_1"]

    escalador_canales = StandardScaler()
    escalador_canales.fit(canales.loc[dia_en_train])
    canales_escalados = escalador_canales.transform(canales).astype(np.float32)

    ventanas = np.lib.stride_tricks.sliding_window_view(canales_escalados, longitud_ventana, axis=0)
    ventanas = np.ascontiguousarray(ventanas.transpose(0, 2, 1))
    origenes = niveles.index[longitud_ventana - 1:]

    serie_objetivo = niveles[estacion_objetivo]
    cambios = pd.DataFrame(index=niveles.index)
    for paso in range(1, maximo_horizonte + 1):
        cambios[paso] = serie_objetivo.shift(-paso) - serie_objetivo
    cambios = cambios.loc[origenes].to_numpy()

    fecha_mas_lejana = origenes + pd.Timedelta(days=maximo_horizonte)
    fecha_mas_cercana = origenes + pd.Timedelta(days=1)
    en_train = np.asarray(fecha_mas_lejana < corte_train)
    en_holdout = np.asarray(fecha_mas_cercana >= corte_train)

    escalador_objetivo = StandardScaler()
    escalador_objetivo.fit(cambios[en_train])
    cambios_escalados = escalador_objetivo.transform(cambios).astype(np.float32)

    return {
        "ventanas": ventanas,
        "origenes": origenes,
        "cambios_escalados": cambios_escalados,
        "escalador_objetivo": escalador_objetivo,
        "escalador_canales": escalador_canales,
        "en_train": en_train,
        "en_holdout": en_holdout,
        "canales": list(canales.columns),
        "climatologias": climatologias,
    }


if __name__ == "__main__":
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
