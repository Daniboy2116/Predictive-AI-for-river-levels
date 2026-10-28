import warnings
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from src.configuracion import (
    ESTACIONES,
    ESTACIONES_OBJETIVO,
    FECHA_FIN_TRAIN,
    HORIZONTE_PRINCIPAL,
    HORIZONTES,
)
from src.evaluate import armar_predicciones, evaluar_y_guardar
from src.preprocessing import (
    ajustar_climatologias,
    cargar_datos,
    evaluar_climatologia,
    preparar_datos,
)

REZAGO_EXOGENAS = HORIZONTE_PRINCIPAL


def calcular_anomalias(niveles: pd.DataFrame, climatologias: dict) -> pd.DataFrame:
    anomalias = pd.DataFrame(index=niveles.index)
    for nombre in ESTACIONES:
        anomalias[nombre] = niveles[nombre] - evaluar_climatologia(climatologias[nombre], niveles.index)
    return anomalias


def preparar_exogenas(anomalias: pd.DataFrame, estacion: str, rezago: int = REZAGO_EXOGENAS) -> tuple[pd.DataFrame, pd.DataFrame]:
    otras_estaciones = [e for e in ESTACIONES if e != estacion]
    nombres_cols = [f"{e}_t-{rezago}" for e in otras_estaciones]
    
    ultimo_dia = anomalias.index[-1]
    fechas_extendidas = pd.date_range(anomalias.index[0], ultimo_dia + pd.Timedelta(days=rezago), freq="D")
    
    exogenas_extendidas = anomalias[otras_estaciones].reindex(fechas_extendidas).shift(rezago)
    exogenas_extendidas.columns = nombres_cols
    
    exogenas_extendidas = exogenas_extendidas.iloc[rezago:]
    exogenas = exogenas_extendidas.loc[:ultimo_dia]
    
    return exogenas, exogenas_extendidas


def seleccionar_orden_arima(anomalia_train: pd.Series, exog_train: pd.DataFrame) -> tuple:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        modelo_auto = auto_arima(
            anomalia_train,
            X=exog_train,
            start_p=1,
            max_p=3,
            d=0,
            start_q=0,
            max_q=2,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
    return modelo_auto.order


def generar_pronosticos_dinamicos(
    modelo_entrenado,
    anomalia_serie: pd.Series,
    exogenas: pd.DataFrame,
    exogenas_extendidas: pd.DataFrame,
    origenes: set,
    maximo_horizonte: int = max(HORIZONTES),
) -> pd.DataFrame:
    modelo_aplicado = modelo_entrenado.apply(anomalia_serie, exog=exogenas)
    ultima_posicion = len(anomalia_serie) - 1
    pronosticos = {}

    for origen in sorted(origenes):
        posicion = anomalia_serie.index.get_loc(origen)
        fin = posicion + maximo_horizonte

        if fin <= ultima_posicion:
            pred = modelo_aplicado.get_prediction(start=posicion + 1, end=fin, dynamic=0)
        else:
            exog_futuras = exogenas_extendidas.iloc[ultima_posicion + 1:fin + 1]
            pred = modelo_aplicado.get_prediction(
                start=posicion + 1,
                end=fin,
                dynamic=0,
                exog=exog_futuras,
            )
        pronosticos[origen] = pred.predicted_mean.to_numpy()

    return pd.DataFrame.from_dict(pronosticos, orient="index", columns=range(1, maximo_horizonte + 1))


def ejecutar_sarimax(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    # === Preparación de datos ===
    climatologias = ajustar_climatologias(niveles, observados)
    anomalias = calcular_anomalias(niveles, climatologias)
    exogenas, exogenas_extendidas = preparar_exogenas(anomalias, estacion, REZAGO_EXOGENAS)
    
    anomalia_objetivo = anomalias[estacion].iloc[REZAGO_EXOGENAS:]
    en_train = anomalia_objetivo.index <= pd.Timestamp(FECHA_FIN_TRAIN)

    valor_p_adf = adfuller(anomalia_objetivo[en_train].to_numpy(), maxlag=30, result_object=True).pvalue
    print(f"{estacion.ljust(15)} ADF p-valor = {valor_p_adf:.2e} -> d = 0")

    # === Definición del modelo ===
    mejor_orden = seleccionar_orden_arima(anomalia_objetivo[en_train], exogenas[en_train])
    print(f"{estacion.ljust(15)} orden AutoARIMA elegido: {mejor_orden}")
    modelo = SARIMAX(anomalia_objetivo[en_train], exog=exogenas[en_train], order=mejor_orden, trend="n")

    # === Entrenamiento ===
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        modelo_entrenado = modelo.fit(disp=False)

    # === Evaluación ===
    tablas_por_horizonte = {}
    origenes = set()
    for horizonte in HORIZONTES:
        datos = preparar_datos(niveles, observados, estacion, horizonte)
        tablas_por_horizonte[horizonte] = datos
        origenes.update(datos["tabla_val"].index)
        origenes.update(datos["tabla_test"].index)
        origenes.update(datos["tabla_futuro"].index)

    pronosticos_anomalia = generar_pronosticos_dinamicos(
        modelo_entrenado,
        anomalia_objetivo,
        exogenas,
        exogenas_extendidas,
        origenes,
    )

    predicciones_val = []
    predicciones_test = []

    for horizonte in HORIZONTES:
        datos = tablas_por_horizonte[horizonte]
        for conjunto in ["val", "test", "futuro"]:
            tabla = datos["tabla_" + conjunto]
            anomalia_pred = pronosticos_anomalia.loc[tabla.index, horizonte].to_numpy()
            fechas_obj = pd.DatetimeIndex(tabla["fecha_objetivo"])
            nivel_pred = evaluar_climatologia(climatologias[estacion], fechas_obj) + anomalia_pred
            
            res = armar_predicciones(tabla, nivel_pred, estacion, "AutoARIMA", horizonte)

            if conjunto == "val":
                predicciones_val.append(res)
            else:
                predicciones_test.append(res)

    evaluar_y_guardar(estacion, "AutoARIMA", pd.concat(predicciones_val), pd.concat(predicciones_test))
    
    val_sari = [r.assign(Modelo="SARIMAX") for r in predicciones_val]
    test_sari = [r.assign(Modelo="SARIMAX") for r in predicciones_test]
    return evaluar_y_guardar(estacion, "SARIMAX", pd.concat(val_sari), pd.concat(test_sari))


def ejecutar() -> None:
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_sarimax(niveles, observados, estacion)


if __name__ == "__main__":
    ejecutar()
