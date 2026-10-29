"""Capa de abstracción, registro, persistencia e inferencia de modelos para backtesting.

Define la interfaz común ModeloPronostico y adaptadores completos para:
- Líneas base: Persistencia, Climatologia, AnomaliaAmortiguada
- Modelos lineales: Ridge
- Ensambles de árboles: RandomForest, XGBoost, LightGBM
- Modelos estadísticos y aditivos: SARIMAX / AutoARIMA, Prophet
- Modelos fundacionales zero-shot: Google TimesFM
- Redes neuronales PyTorch: LSTM, CNN 1D causal dilatada, Transformer
- Ensamble apilado (Stacking)
"""
import copy
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import lightgbm as lgb
import xgboost as xgb
import torch

from src.configuracion import (
    CARPETA_MODELOS, ARCHIVO_METADATOS_MODELOS, ESTACIONES, ESTACIONES_OBJETIVO,
    HORIZONTE_BACKTEST, SEMILLA,
)
from src.models.arima_model import calcular_anomalias, preparar_exogenas, seleccionar_orden_arima
from src.models.google_model import predecir_timesfm_evento
from src.models.neural_models import RedCNN1D, RedLSTM, RedTransformer, TAMANO_LOTE, TASA_APRENDIZAJE
from src.preprocessing import (
    ajustar_climatologias, cambio_a_nivel, evaluar_climatologia,
    preparar_datos_backtest, preparar_ventanas_backtest,
)

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


@runtime_checkable
class ModeloPronostico(Protocol):
    """Protocolo común para modelos de pronóstico en backtesting histórico."""
    nombre: str
    familia: str
    tipo_salida: str
    horizonte_maximo: int
    requiere_entrenamiento: bool
    soporta_multicanal: bool
    lookbacks_soportados: List[int]

    def entrenar(
        self,
        niveles: pd.DataFrame,
        observados: pd.DataFrame,
        inicio_holdout: pd.Timestamp,
        horizonte_maximo: int = HORIZONTE_BACKTEST,
    ) -> None:
        """Entrena el modelo usando exclusivamente datos anteriores a inicio_holdout."""
        ...

    def predecir_evento(
        self,
        niveles: pd.DataFrame,
        observados: pd.DataFrame,
        fecha_origen: pd.Timestamp,
        horizonte_maximo: int,
        estacion: str,
    ) -> pd.DataFrame:
        """Genera el pronóstico del evento para horizontes 1..horizonte_maximo."""
        ...


class ModeloPersistencia:
    """Línea base de persistencia: el nivel futuro predicho es igual al nivel de hoy."""
    def __init__(self):
        self.nombre = "Persistencia"
        self.familia = "linea_base"
        self.tipo_salida = "directa"
        self.horizonte_maximo = HORIZONTE_BACKTEST
        self.requiere_entrenamiento = False
        self.soporta_multicanal = False
        self.lookbacks_soportados = [30, 60, 90]

    def entrenar(self, niveles: pd.DataFrame, observados: pd.DataFrame, inicio_holdout: pd.Timestamp, horizonte_maximo: int = HORIZONTE_BACKTEST) -> None:
        pass

    def predecir_evento(self, niveles: pd.DataFrame, observados: pd.DataFrame, fecha_origen: pd.Timestamp, horizonte_maximo: int, estacion: str) -> pd.DataFrame:
        nivel_t0 = float(niveles.loc[fecha_origen, estacion])
        filas = []
        for h in range(1, horizonte_maximo + 1):
            fecha_h = pd.Timestamp(fecha_origen) + pd.Timedelta(days=h)
            filas.append({
                "Fecha": fecha_h,
                "Estacion": estacion,
                "Horizonte": h,
                "Prediccion": nivel_t0,
            })
        return pd.DataFrame(filas)


class ModeloClimatologia:
    """Línea base: proyecta la climatología armónica estacional calculada solo con datos de train."""
    def __init__(self):
        self.nombre = "Climatologia"
        self.familia = "linea_base"
        self.tipo_salida = "directa"
        self.horizonte_maximo = HORIZONTE_BACKTEST
        self.requiere_entrenamiento = True
        self.soporta_multicanal = False
        self.lookbacks_soportados = [30, 60, 90]
        self.climatologias: Dict[str, Any] = {}

    def entrenar(self, niveles: pd.DataFrame, observados: pd.DataFrame, inicio_holdout: pd.Timestamp, horizonte_maximo: int = HORIZONTE_BACKTEST) -> None:
        en_train = niveles.index < pd.Timestamp(inicio_holdout)
        self.climatologias = ajustar_climatologias(niveles.loc[en_train], observados.loc[en_train])

    def predecir_evento(self, niveles: pd.DataFrame, observados: pd.DataFrame, fecha_origen: pd.Timestamp, horizonte_maximo: int, estacion: str) -> pd.DataFrame:
        if not self.climatologias:
            self.climatologias = ajustar_climatologias(niveles, observados)
        fecha_origen = pd.Timestamp(fecha_origen)
        fechas_futuras = pd.date_range(fecha_origen + pd.Timedelta(days=1), periods=horizonte_maximo, freq="D")
        niveles_pred = evaluar_climatologia(self.climatologias[estacion], fechas_futuras)
        filas = []
        for h, (f_h, n_pred) in enumerate(zip(fechas_futuras, niveles_pred), start=1):
            filas.append({
                "Fecha": f_h,
                "Estacion": estacion,
                "Horizonte": h,
                "Prediccion": float(n_pred),
            })
        return pd.DataFrame(filas)


class ModeloAnomaliaAmortiguada:
    """Línea base: amortigua la anomalía observada en t0 hacia la climatología según phi_h."""
    def __init__(self):
        self.nombre = "AnomaliaAmortiguada"
        self.familia = "linea_base"
        self.tipo_salida = "directa"
        self.horizonte_maximo = HORIZONTE_BACKTEST
        self.requiere_entrenamiento = True
        self.soporta_multicanal = False
        self.lookbacks_soportados = [30, 60, 90]
        self.climatologias: Dict[str, Any] = {}
        self.phis: Dict[tuple, float] = {}

    def entrenar(self, niveles: pd.DataFrame, observados: pd.DataFrame, inicio_holdout: pd.Timestamp, horizonte_maximo: int = HORIZONTE_BACKTEST) -> None:
        en_train = niveles.index < pd.Timestamp(inicio_holdout)
        self.climatologias = ajustar_climatologias(niveles.loc[en_train], observados.loc[en_train])

        anomalias = pd.DataFrame(index=niveles.loc[en_train].index)
        for est in ESTACIONES:
            clima = evaluar_climatologia(self.climatologias[est], anomalias.index)
            anomalias[est] = niveles.loc[en_train, est] - clima

        for est in ESTACIONES:
            s_anom = anomalias[est]
            s_obs = observados.loc[en_train, est]
            for h in range(1, horizonte_maximo + 1):
                s_anom_h = s_anom.shift(-h)
                s_obs_h = s_obs.shift(-h).fillna(False).astype(bool)
                valido = s_obs & s_obs_h & (~s_anom.isna()) & (~s_anom_h.isna())
                if valido.sum() > 30:
                    x = s_anom[valido]
                    y = s_anom_h[valido]
                    phi = float((x * y).sum() / (x ** 2).sum())
                    phi = max(0.0, min(1.0, phi))
                else:
                    phi = float(np.exp(-h / 70.0))
                self.phis[(est, h)] = phi

    def predecir_evento(self, niveles: pd.DataFrame, observados: pd.DataFrame, fecha_origen: pd.Timestamp, horizonte_maximo: int, estacion: str) -> pd.DataFrame:
        if not self.climatologias:
            self.entrenar(niveles, observados, fecha_origen, horizonte_maximo)

        fecha_origen = pd.Timestamp(fecha_origen)
        nivel_t0 = float(niveles.loc[fecha_origen, estacion])
        clima_t0 = float(evaluar_climatologia(self.climatologias[estacion], pd.DatetimeIndex([fecha_origen]))[0])
        anomalia_t0 = nivel_t0 - clima_t0

        fechas_futuras = pd.date_range(fecha_origen + pd.Timedelta(days=1), periods=horizonte_maximo, freq="D")
        clima_futuro = evaluar_climatologia(self.climatologias[estacion], fechas_futuras)

        filas = []
        for h, (f_h, c_h) in enumerate(zip(fechas_futuras, clima_futuro), start=1):
            phi = self.phis.get((estacion, h), float(np.exp(-h / 70.0)))
            nivel_pred = float(c_h + phi * anomalia_t0)
            filas.append({
                "Fecha": f_h,
                "Estacion": estacion,
                "Horizonte": h,
                "Prediccion": nivel_pred,
            })
        return pd.DataFrame(filas)


class ModeloTabularDirecto:
    """Adaptador para modelos que entrenan un estimador tabular independiente por horizonte."""
    def __init__(self, nombre: str, familia: str, crear_estimador_fn):
        self.nombre = nombre
        self.familia = familia
        self.tipo_salida = "directa"
        self.horizonte_maximo = HORIZONTE_BACKTEST
        self.requiere_entrenamiento = True
        self.soporta_multicanal = False
        self.lookbacks_soportados = [30, 60, 90]
        self.crear_estimador_fn = crear_estimador_fn
        self.estimadores: Dict[tuple, tuple] = {}
        self.inicio_holdout: Optional[pd.Timestamp] = None

    def entrenar(self, niveles: pd.DataFrame, observados: pd.DataFrame, inicio_holdout: pd.Timestamp, horizonte_maximo: int = HORIZONTE_BACKTEST) -> None:
        self.inicio_holdout = pd.Timestamp(inicio_holdout)
        self.estimadores.clear()
        CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
        carpeta_especifica = CARPETA_MODELOS / self.nombre
        carpeta_especifica.mkdir(parents=True, exist_ok=True)

        for estacion in ESTACIONES:
            for h in range(1, horizonte_maximo + 1):
                ruta_guardado = carpeta_especifica / f"{estacion}_h{h:02d}.joblib"
                datos = preparar_datos_backtest(niveles, observados, estacion, self.inicio_holdout, h)
                estimador = self.crear_estimador_fn()
                estimador.fit(datos["x_train"], datos["y_train"])
                escalador = datos["escalador"]
                variables = datos["variables"]
                joblib.dump({"modelo": estimador, "escalador": escalador, "variables": variables}, ruta_guardado)
                self.estimadores[(estacion, h)] = (estimador, escalador, variables)

    def predecir_evento(self, niveles: pd.DataFrame, observados: pd.DataFrame, fecha_origen: pd.Timestamp, horizonte_maximo: int, estacion: str) -> pd.DataFrame:
        filas = []
        nivel_t0 = float(niveles.loc[fecha_origen, estacion])
        for h in range(1, horizonte_maximo + 1):
            if (estacion, h) not in self.estimadores:
                ruta_guardado = CARPETA_MODELOS / self.nombre / f"{estacion}_h{h:02d}.joblib"
                if ruta_guardado.exists():
                    cargado = joblib.load(ruta_guardado)
                    self.estimadores[(estacion, h)] = (cargado["modelo"], cargado["escalador"], cargado["variables"])
                else:
                    raise RuntimeError(f"El modelo {self.nombre} no ha sido entrenado para {estacion} h={h}.")

            estimador, escalador, variables = self.estimadores[(estacion, h)]
            datos_h = preparar_datos_backtest(niveles, observados, estacion, self.inicio_holdout or fecha_origen, h)
            fila_t0 = datos_h["tabla_completa"].loc[[fecha_origen], variables]
            fila_escalada = escalador.transform(fila_t0)
            cambio_pred = float(estimador.predict(fila_escalada)[0])
            nivel_pred = float(cambio_a_nivel(nivel_t0, cambio_pred))
            fecha_h = pd.Timestamp(fecha_origen) + pd.Timedelta(days=h)
            filas.append({
                "Fecha": fecha_h,
                "Estacion": estacion,
                "Horizonte": h,
                "Prediccion": nivel_pred,
            })
        return pd.DataFrame(filas)


class ModeloSARIMAX:
    """Adaptador para SARIMAX / AutoARIMA con variables exógenas aguas arriba y corrección estacional."""
    def __init__(self):
        self.nombre = "SARIMAX"
        self.familia = "estadistico"
        self.tipo_salida = "dinamica"
        self.horizonte_maximo = HORIZONTE_BACKTEST
        self.requiere_entrenamiento = True
        self.soporta_multicanal = True
        self.lookbacks_soportados = [30, 60, 90]
        self.modelos: Dict[str, Any] = {}
        self.climatologias: Dict[str, Any] = {}
        self.inicio_holdout: Optional[pd.Timestamp] = None

    def entrenar(self, niveles: pd.DataFrame, observados: pd.DataFrame, inicio_holdout: pd.Timestamp, horizonte_maximo: int = HORIZONTE_BACKTEST) -> None:
        self.inicio_holdout = pd.Timestamp(inicio_holdout)
        self.modelos.clear()
        en_train = niveles.index < self.inicio_holdout
        niveles_train = niveles.loc[en_train]
        obs_train = observados.loc[en_train]

        self.climatologias = ajustar_climatologias(niveles_train, obs_train)
        anomalias = calcular_anomalias(niveles_train, self.climatologias)

        CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
        carpeta_sarimax = CARPETA_MODELOS / "SARIMAX"
        carpeta_sarimax.mkdir(parents=True, exist_ok=True)

        for estacion in ESTACIONES:
            ruta_est = carpeta_sarimax / f"{estacion}.joblib"
            exog, _ = preparar_exogenas(anomalias, estacion, rezago=14)
            # `preparar_exogenas` ya descarta el periodo inicial del rezago.
            # Alinear por fecha evita que una segunda selección posicional
            # desplace o reduzca solo una de las dos series.
            anom_obj = anomalias[estacion].reindex(exog.index)
            exog_obj = exog.reindex(anom_obj.index)
            validos = anom_obj.notna() & exog_obj.notna().all(axis=1)
            anom_obj = anom_obj.loc[validos]
            exog_obj = exog_obj.loc[validos]
            en_train = anom_obj.index < self.inicio_holdout
            anom_obj = anom_obj.loc[en_train]
            exog_obj = exog_obj.loc[en_train]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", (ConvergenceWarning, UserWarning))
                pval = adfuller(anom_obj.to_numpy(), maxlag=30, result_object=True).pvalue
                d_est = 0 if pval < 0.05 else 1
                orden = seleccionar_orden_arima(anom_obj, exog_obj, d_estimado=d_est)
                modelo = SARIMAX(anom_obj, exog=exog_obj, order=orden, trend="n")
                res = modelo.fit(disp=False)

            paquete = {"res": res, "orden": orden, "climatologias": self.climatologias}
            joblib.dump(paquete, ruta_est)
            self.modelos[estacion] = paquete

    def predecir_evento(self, niveles: pd.DataFrame, observados: pd.DataFrame, fecha_origen: pd.Timestamp, horizonte_maximo: int, estacion: str) -> pd.DataFrame:
        fecha_origen = pd.Timestamp(fecha_origen)
        if estacion not in self.modelos:
            ruta_est = CARPETA_MODELOS / "SARIMAX" / f"{estacion}.joblib"
            if ruta_est.exists():
                self.modelos[estacion] = joblib.load(ruta_est)
                self.climatologias = self.modelos[estacion]["climatologias"]
            else:
                self.entrenar(niveles, observados, self.inicio_holdout or fecha_origen, horizonte_maximo)

        climatologias = self.climatologias or self.modelos[estacion]["climatologias"]
        res = self.modelos[estacion]["res"]

        otras = [e for e in ESTACIONES if e != estacion]
        fechas_futuras = pd.date_range(fecha_origen + pd.Timedelta(days=1), periods=horizonte_maximo, freq="D")
        fechas_lag = fechas_futuras - pd.Timedelta(days=14)

        filas_exog = []
        for f_lag in fechas_lag:
            fila = {}
            for e in otras:
                clima_lag = float(evaluar_climatologia(climatologias[e], pd.DatetimeIndex([f_lag]))[0])
                fila[f"{e}_t-14"] = float(niveles.loc[f_lag, e] - clima_lag)
            filas_exog.append(fila)
        df_exog_futuras = pd.DataFrame(filas_exog, index=fechas_futuras)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_anom = res.forecast(steps=horizonte_maximo, exog=df_exog_futuras).to_numpy()

        clima_futuro = evaluar_climatologia(climatologias[estacion], fechas_futuras)
        filas = []
        for h, (f_h, c_h) in enumerate(zip(fechas_futuras, clima_futuro), start=1):
            nivel_pred = float(c_h + pred_anom[h - 1])
            filas.append({
                "Fecha": f_h,
                "Estacion": estacion,
                "Horizonte": h,
                "Prediccion": nivel_pred,
            })
        return pd.DataFrame(filas)


class ModeloProphet:
    """Adaptador para Facebook Prophet con estacionalidad anual y regresores de memoria aguas arriba."""
    def __init__(self):
        self.nombre = "Prophet"
        self.familia = "aditivo"
        self.tipo_salida = "directa"
        self.horizonte_maximo = HORIZONTE_BACKTEST
        self.requiere_entrenamiento = True
        self.soporta_multicanal = True
        self.lookbacks_soportados = [30, 60, 90]
        self.modelos: Dict[str, Any] = {}
        self.inicio_holdout: Optional[pd.Timestamp] = None

    def entrenar(self, niveles: pd.DataFrame, observados: pd.DataFrame, inicio_holdout: pd.Timestamp, horizonte_maximo: int = HORIZONTE_BACKTEST) -> None:
        self.inicio_holdout = pd.Timestamp(inicio_holdout)
        self.modelos.clear()
        en_train = niveles.index < self.inicio_holdout
        sub_train = niveles.loc[en_train]

        CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
        carpeta_prophet = CARPETA_MODELOS / "Prophet"
        carpeta_prophet.mkdir(parents=True, exist_ok=True)

        for estacion in ESTACIONES:
            ruta_est = carpeta_prophet / f"{estacion}.joblib"
            df_train = pd.DataFrame({
                "ds": sub_train.index,
                "y": sub_train[estacion].values,
            })
            df_train["Ayacucho_lag14"] = sub_train["Ayacucho"].shift(14).bfill().values
            df_train["Caicara_lag14"] = sub_train["Caicara"].shift(14).bfill().values
            df_train[f"{estacion}_tend7"] = (sub_train[estacion] - sub_train[estacion].shift(7)).shift(14).bfill().values

            modelo = Prophet(
                growth="flat",
                yearly_seasonality=10,
                weekly_seasonality=False,
                daily_seasonality=False,
            )
            modelo.add_regressor("Ayacucho_lag14")
            modelo.add_regressor("Caicara_lag14")
            modelo.add_regressor(f"{estacion}_tend7")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                modelo.fit(df_train)

            joblib.dump(modelo, ruta_est)
            self.modelos[estacion] = modelo

    def predecir_evento(self, niveles: pd.DataFrame, observados: pd.DataFrame, fecha_origen: pd.Timestamp, horizonte_maximo: int, estacion: str) -> pd.DataFrame:
        fecha_origen = pd.Timestamp(fecha_origen)
        if estacion not in self.modelos:
            ruta_est = CARPETA_MODELOS / "Prophet" / f"{estacion}.joblib"
            if ruta_est.exists():
                self.modelos[estacion] = joblib.load(ruta_est)
            else:
                self.entrenar(niveles, observados, self.inicio_holdout or fecha_origen, horizonte_maximo)

        modelo = self.modelos[estacion]
        fechas_futuras = [fecha_origen + pd.Timedelta(days=h) for h in range(1, horizonte_maximo + 1)]
        df_futuro = pd.DataFrame({"ds": fechas_futuras})

        df_futuro["Ayacucho_lag14"] = [float(niveles.loc[f - pd.Timedelta(days=14), "Ayacucho"]) for f in fechas_futuras]
        df_futuro["Caicara_lag14"] = [float(niveles.loc[f - pd.Timedelta(days=14), "Caicara"]) for f in fechas_futuras]
        df_futuro[f"{estacion}_tend7"] = [
            float(niveles.loc[f - pd.Timedelta(days=14), estacion] - niveles.loc[f - pd.Timedelta(days=21), estacion])
            for f in fechas_futuras
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = modelo.predict(df_futuro)

        predicciones = forecast["yhat"].to_numpy()
        filas = []
        for h in range(1, horizonte_maximo + 1):
            fecha_h = fecha_origen + pd.Timedelta(days=h)
            filas.append({
                "Fecha": fecha_h,
                "Estacion": estacion,
                "Horizonte": h,
                "Prediccion": float(predicciones[h - 1]),
            })
        return pd.DataFrame(filas)


class ModeloTimesFM:
    """Adaptador para Google TimesFM (Foundation Model zero-shot de Google Research)."""
    def __init__(self):
        self.nombre = "TimesFM"
        self.familia = "foundation"
        self.tipo_salida = "multihorizonte"
        self.horizonte_maximo = HORIZONTE_BACKTEST
        self.requiere_entrenamiento = False
        self.soporta_multicanal = False
        self.lookbacks_soportados = [30, 60, 90]

    def entrenar(self, niveles: pd.DataFrame, observados: pd.DataFrame, inicio_holdout: pd.Timestamp, horizonte_maximo: int = HORIZONTE_BACKTEST) -> None:
        pass

    def predecir_evento(self, niveles: pd.DataFrame, observados: pd.DataFrame, fecha_origen: pd.Timestamp, horizonte_maximo: int, estacion: str) -> pd.DataFrame:
        preds_array = predecir_timesfm_evento(niveles, fecha_origen, horizonte_maximo, estacion)
        filas = []
        for h in range(1, horizonte_maximo + 1):
            fecha_h = pd.Timestamp(fecha_origen) + pd.Timedelta(days=h)
            filas.append({
                "Fecha": fecha_h,
                "Estacion": estacion,
                "Horizonte": h,
                "Prediccion": float(preds_array[h - 1]),
            })
        return pd.DataFrame(filas)


class ModeloRedNeuronal:
    """Adaptador para redes neuronales en PyTorch (LSTM, CNN1D, Transformer) multi-horizonte."""
    def __init__(self, nombre: str, clase_red):
        self.nombre = nombre
        self.familia = "redes"
        self.tipo_salida = "multihorizonte"
        self.horizonte_maximo = HORIZONTE_BACKTEST
        self.requiere_entrenamiento = True
        self.soporta_multicanal = True
        self.lookbacks_soportados = [60]
        self.clase_red = clase_red
        self.modelos_estacion: Dict[str, dict] = {}
        self.inicio_holdout: Optional[pd.Timestamp] = None

    def entrenar(self, niveles: pd.DataFrame, observados: pd.DataFrame, inicio_holdout: pd.Timestamp, horizonte_maximo: int = HORIZONTE_BACKTEST) -> None:
        self.inicio_holdout = pd.Timestamp(inicio_holdout)
        self.modelos_estacion.clear()
        CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
        carpeta_especifica = CARPETA_MODELOS / self.nombre
        carpeta_especifica.mkdir(parents=True, exist_ok=True)

        for estacion in ESTACIONES:
            datos_v = preparar_ventanas_backtest(niveles, observados, estacion, 60, self.inicio_holdout, horizonte_maximo)
            x_train = torch.from_numpy(datos_v["ventanas"][datos_v["en_train"]])
            y_train = torch.from_numpy(datos_v["cambios_escalados"][datos_v["en_train"]])

            torch.manual_seed(SEMILLA)
            modelo = self.clase_red(
                numero_canales=len(datos_v["canales"]),
                numero_salidas=horizonte_maximo,
            )
            optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE, weight_decay=1e-4)
            criterio = torch.nn.MSELoss()

            modelo.train()
            dataset = torch.utils.data.TensorDataset(x_train, y_train)
            loader = torch.utils.data.DataLoader(dataset, batch_size=TAMANO_LOTE, shuffle=True)
            for epoca in range(35):
                for x_lote, y_lote in loader:
                    optimizador.zero_grad()
                    pred = modelo(x_lote)
                    perdida = criterio(pred, y_lote)
                    perdida.backward()
                    optimizador.step()

            modelo.eval()
            ruta_pt = carpeta_especifica / f"{estacion}_multihorizonte.pt"
            ruta_meta = carpeta_especifica / f"{estacion}_meta.joblib"
            torch.save(modelo.state_dict(), ruta_pt)
            meta = {
                "escalador_canales": datos_v["escalador_canales"],
                "escalador_objetivo": datos_v["escalador_objetivo"],
                "canales": datos_v["canales"],
                "climatologias": datos_v["climatologias"],
            }
            joblib.dump(meta, ruta_meta)
            self.modelos_estacion[estacion] = {"modelo": modelo, "meta": meta}

    def predecir_evento(self, niveles: pd.DataFrame, observados: pd.DataFrame, fecha_origen: pd.Timestamp, horizonte_maximo: int, estacion: str) -> pd.DataFrame:
        if estacion not in self.modelos_estacion:
            carpeta_especifica = CARPETA_MODELOS / self.nombre
            ruta_pt = carpeta_especifica / f"{estacion}_multihorizonte.pt"
            ruta_meta = carpeta_especifica / f"{estacion}_meta.joblib"
            if ruta_pt.exists() and ruta_meta.exists():
                meta = joblib.load(ruta_meta)
                modelo = self.clase_red(
                    numero_canales=len(meta["canales"]),
                    numero_salidas=horizonte_maximo,
                )
                modelo.load_state_dict(torch.load(ruta_pt, weights_only=True))
                modelo.eval()
                self.modelos_estacion[estacion] = {"modelo": modelo, "meta": meta}
            else:
                raise RuntimeError(f"La red {self.nombre} no ha sido entrenada para {estacion}.")

        paquete = self.modelos_estacion[estacion]
        modelo = paquete["modelo"]
        meta = paquete["meta"]
        escalador_canales = meta["escalador_canales"]
        escalador_objetivo = meta["escalador_objetivo"]
        climatologias = meta["climatologias"]

        posicion = niveles.index.get_loc(fecha_origen)
        inicio_v = posicion - 60 + 1
        if inicio_v < 0:
            raise ValueError(f"No hay suficientes días previos a {fecha_origen} para la ventana de 60 días.")
        sub_niveles = niveles.iloc[inicio_v:posicion + 1]

        canales = pd.DataFrame(index=sub_niveles.index)
        for est in ESTACIONES:
            canales[est + "_nivel"] = sub_niveles[est]
            from src.preprocessing import evaluar_climatologia, variables_calendario
            clima = evaluar_climatologia(climatologias[est], sub_niveles.index)
            canales[est + "_anomalia"] = sub_niveles[est] - clima
        cal = variables_calendario(sub_niveles.index, 1)
        canales["sen_1"] = cal["sen_1"]
        canales["cos_1"] = cal["cos_1"]

        ventana_arr = escalador_canales.transform(canales).astype(np.float32)
        # Las ventanas de entrenamiento tienen forma (muestras, días, canales).
        # Mantener ese orden es obligatorio para LSTM y Transformer; CNN1D
        # realiza su permutación internamente antes de Conv1d.
        tensor_x = torch.from_numpy(ventana_arr).unsqueeze(0)

        with torch.no_grad():
            cambios_escalados = modelo(tensor_x).cpu().numpy()

        cambios_reales = escalador_objetivo.inverse_transform(cambios_escalados)[0]
        nivel_t0 = float(niveles.loc[fecha_origen, estacion])

        filas = []
        for h in range(1, horizonte_maximo + 1):
            fecha_h = pd.Timestamp(fecha_origen) + pd.Timedelta(days=h)
            cambio_h = float(cambios_reales[h - 1])
            nivel_pred = float(cambio_a_nivel(nivel_t0, cambio_h))
            filas.append({
                "Fecha": fecha_h,
                "Estacion": estacion,
                "Horizonte": h,
                "Prediccion": nivel_pred,
            })
        return pd.DataFrame(filas)


class ModeloEnsamble:
    """Ensamble apilado (Stacking): combina las predicciones de los modelos base entrenados."""
    def __init__(self, modelos_base: List[ModeloPronostico]):
        self.nombre = "Ensamble"
        self.familia = "ensamble"
        self.tipo_salida = "directa"
        self.horizonte_maximo = HORIZONTE_BACKTEST
        self.requiere_entrenamiento = True
        self.soporta_multicanal = False
        self.lookbacks_soportados = [30, 60, 90]
        self.modelos_base = modelos_base
        self.pesos: Dict[tuple, Dict[str, float]] = {}

    def entrenar(self, niveles: pd.DataFrame, observados: pd.DataFrame, inicio_holdout: pd.Timestamp, horizonte_maximo: int = HORIZONTE_BACKTEST) -> None:
        for m in self.modelos_base:
            if m.requiere_entrenamiento:
                m.entrenar(niveles, observados, inicio_holdout, horizonte_maximo)

        pesos_default = {"Ridge": 0.35, "LightGBM": 0.30, "RandomForest": 0.20, "XGBoost": 0.15}
        for estacion in ESTACIONES:
            for h in range(1, horizonte_maximo + 1):
                self.pesos[(estacion, h)] = copy.deepcopy(pesos_default)

    def predecir_evento(self, niveles: pd.DataFrame, observados: pd.DataFrame, fecha_origen: pd.Timestamp, horizonte_maximo: int, estacion: str) -> pd.DataFrame:
        preds_base = {}
        for m in self.modelos_base:
            df_m = m.predecir_evento(niveles, observados, fecha_origen, horizonte_maximo, estacion)
            preds_base[m.nombre] = df_m.set_index("Horizonte")["Prediccion"].to_dict()

        nivel_t0 = float(niveles.loc[fecha_origen, estacion])
        filas = []
        for h in range(1, horizonte_maximo + 1):
            fecha_h = pd.Timestamp(fecha_origen) + pd.Timedelta(days=h)
            pesos_h = self.pesos.get((estacion, h), {"Ridge": 0.35, "LightGBM": 0.30, "RandomForest": 0.20, "XGBoost": 0.15})
            peso_total = 0.0
            cambio_combinado = 0.0
            for nom, w in pesos_h.items():
                if nom in preds_base:
                    cambio_m = preds_base[nom][h] - nivel_t0
                    cambio_combinado += w * cambio_m
                    peso_total += w
            if peso_total > 0:
                cambio_combinado /= peso_total
            nivel_pred = float(cambio_a_nivel(nivel_t0, cambio_combinado))
            filas.append({
                "Fecha": fecha_h,
                "Estacion": estacion,
                "Horizonte": h,
                "Prediccion": nivel_pred,
            })
        return pd.DataFrame(filas)


def _crear_ridge():
    return Ridge(alpha=100.0, random_state=SEMILLA)

def _crear_rf():
    return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=SEMILLA, n_jobs=-1)

def _crear_xgb():
    return xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEMILLA, n_jobs=-1)

def _crear_lgbm():
    return lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=SEMILLA, n_jobs=-1, verbose=-1)


def instanciar_modelos() -> Dict[str, Any]:
    """Crea las instancias de todos los modelos soportados para backtesting."""
    mod_persistencia = ModeloPersistencia()
    mod_climatologia = ModeloClimatologia()
    mod_anomalia = ModeloAnomaliaAmortiguada()
    mod_ridge = ModeloTabularDirecto("Ridge", "lineal", _crear_ridge)
    mod_rf = ModeloTabularDirecto("RandomForest", "arboles", _crear_rf)
    mod_xgb = ModeloTabularDirecto("XGBoost", "arboles", _crear_xgb)
    mod_lgbm = ModeloTabularDirecto("LightGBM", "arboles", _crear_lgbm)
    mod_sarimax = ModeloSARIMAX()
    mod_prophet = ModeloProphet()
    mod_timesfm = ModeloTimesFM()
    mod_lstm = ModeloRedNeuronal("LSTM", RedLSTM)
    mod_cnn1d = ModeloRedNeuronal("CNN1D", RedCNN1D)
    mod_transformer = ModeloRedNeuronal("Transformer", RedTransformer)
    mod_ensamble = ModeloEnsamble([mod_ridge, mod_lgbm, mod_rf, mod_xgb])

    return {
        "Persistencia": mod_persistencia,
        "Climatologia": mod_climatologia,
        "AnomaliaAmortiguada": mod_anomalia,
        "Ridge": mod_ridge,
        "RandomForest": mod_rf,
        "XGBoost": mod_xgb,
        "LightGBM": mod_lgbm,
        "SARIMAX": mod_sarimax,
        "AutoARIMA": mod_sarimax,
        "Prophet": mod_prophet,
        "TimesFM": mod_timesfm,
        "LSTM": mod_lstm,
        "CNN1D": mod_cnn1d,
        "Transformer": mod_transformer,
        "Ensamble": mod_ensamble,
    }


REGISTRO_MODELOS: Dict[str, dict] = {
    "Persistencia": {
        "nombre": "Persistencia",
        "familia": "linea_base",
        "tipo_salida": "directa",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": False,
        "soporta_multicanal": False,
        "lookbacks_soportados": [30, 60, 90],
    },
    "Climatologia": {
        "nombre": "Climatologia",
        "familia": "linea_base",
        "tipo_salida": "directa",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": True,
        "soporta_multicanal": False,
        "lookbacks_soportados": [30, 60, 90],
    },
    "AnomaliaAmortiguada": {
        "nombre": "AnomaliaAmortiguada",
        "familia": "linea_base",
        "tipo_salida": "directa",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": True,
        "soporta_multicanal": False,
        "lookbacks_soportados": [30, 60, 90],
    },
    "Ridge": {
        "nombre": "Ridge",
        "familia": "lineal",
        "tipo_salida": "directa",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": True,
        "soporta_multicanal": False,
        "lookbacks_soportados": [30, 60, 90],
    },
    "RandomForest": {
        "nombre": "RandomForest",
        "familia": "arboles",
        "tipo_salida": "directa",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": True,
        "soporta_multicanal": False,
        "lookbacks_soportados": [30, 60, 90],
    },
    "XGBoost": {
        "nombre": "XGBoost",
        "familia": "arboles",
        "tipo_salida": "directa",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": True,
        "soporta_multicanal": False,
        "lookbacks_soportados": [30, 60, 90],
    },
    "LightGBM": {
        "nombre": "LightGBM",
        "familia": "arboles",
        "tipo_salida": "directa",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": True,
        "soporta_multicanal": False,
        "lookbacks_soportados": [30, 60, 90],
    },
    "SARIMAX": {
        "nombre": "SARIMAX",
        "familia": "estadistico",
        "tipo_salida": "dinamica",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": True,
        "soporta_multicanal": True,
        "lookbacks_soportados": [30, 60, 90],
    },
    "AutoARIMA": {
        "nombre": "AutoARIMA",
        "familia": "estadistico",
        "tipo_salida": "dinamica",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": True,
        "soporta_multicanal": True,
        "lookbacks_soportados": [30, 60, 90],
    },
    "Prophet": {
        "nombre": "Prophet",
        "familia": "aditivo",
        "tipo_salida": "directa",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": True,
        "soporta_multicanal": True,
        "lookbacks_soportados": [30, 60, 90],
    },
    "TimesFM": {
        "nombre": "TimesFM",
        "familia": "foundation",
        "tipo_salida": "multihorizonte",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": False,
        "soporta_multicanal": False,
        "lookbacks_soportados": [30, 60, 90],
    },
    "LSTM": {
        "nombre": "LSTM",
        "familia": "redes",
        "tipo_salida": "multihorizonte",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": True,
        "soporta_multicanal": True,
        "lookbacks_soportados": [60],
    },
    "CNN1D": {
        "nombre": "CNN1D",
        "familia": "redes",
        "tipo_salida": "multihorizonte",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": True,
        "soporta_multicanal": True,
        "lookbacks_soportados": [60],
    },
    "Transformer": {
        "nombre": "Transformer",
        "familia": "redes",
        "tipo_salida": "multihorizonte",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": True,
        "soporta_multicanal": True,
        "lookbacks_soportados": [60],
    },
    "Ensamble": {
        "nombre": "Ensamble",
        "familia": "ensamble",
        "tipo_salida": "directa",
        "horizonte_maximo": 14,
        "requiere_entrenamiento": True,
        "soporta_multicanal": False,
        "lookbacks_soportados": [30, 60, 90],
    },
}

_CACHE_MODELOS: Dict[str, Any] = {}


def obtener_modelo_backtest(
    nombre_modelo: str,
    niveles: pd.DataFrame,
    observados: pd.DataFrame,
    inicio_holdout: pd.Timestamp,
    horizonte_maximo: int = HORIZONTE_BACKTEST,
    forzar_reentrenamiento: bool = False,
) -> ModeloPronostico:
    """Devuelve una instancia de modelo entrenada y lista para inferencia en backtesting."""
    clave = f"{nombre_modelo}_{pd.Timestamp(inicio_holdout).strftime('%Y%m%d')}_{horizonte_maximo}"
    if not forzar_reentrenamiento and clave in _CACHE_MODELOS:
        return _CACHE_MODELOS[clave]

    modelos = instanciar_modelos()
    if nombre_modelo not in modelos:
        raise ValueError(f"Modelo '{nombre_modelo}' no reconocido. Opciones: {list(modelos.keys())}")

    modelo = modelos[nombre_modelo]
    if modelo.requiere_entrenamiento:
        modelo.entrenar(niveles, observados, inicio_holdout, horizonte_maximo)

    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    meta_path = ARCHIVO_METADATOS_MODELOS
    meta_actual = {}
    if meta_path.exists():
        try:
            meta_actual = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta_actual = {}

    meta_actual[nombre_modelo] = {
        "modelo": nombre_modelo,
        "inicio_train": str(niveles.index.min().date()),
        "fin_train": str((pd.Timestamp(inicio_holdout) - pd.Timedelta(days=1)).date()),
        "inicio_holdout": str(pd.Timestamp(inicio_holdout).date()),
        "horizonte_maximo": horizonte_maximo,
        "semilla": SEMILLA,
    }
    meta_path.write_text(json.dumps(meta_actual, indent=2), encoding="utf-8")

    _CACHE_MODELOS[clave] = modelo
    return modelo
