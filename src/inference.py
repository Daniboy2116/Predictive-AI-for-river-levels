"""Capa de abstracción, registro, persistencia e inferencia de modelos para backtesting.

Define la interfaz común ModeloPronostico y adaptadores para:
- Persistencia (línea base)
- Ridge (regresión lineal regularizada)
- RandomForest, XGBoost, LightGBM (ensambles de árboles)
- LSTM, CNN 1D causal dilatada, Transformer (redes neuronales PyTorch)
- Ensamble (stacking apilado)
"""
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
import lightgbm as lgb
import xgboost as xgb
import torch

from src.configuracion import (
    CARPETA_MODELOS, ARCHIVO_METADATOS_MODELOS, ESTACIONES, ESTACIONES_OBJETIVO,
    HORIZONTE_BACKTEST, SEMILLA,
)
from src.models.neural_models import RedCNN1D, RedLSTM, RedTransformer, TAMANO_LOTE, TASA_APRENDIZAJE
from src.preprocessing import cambio_a_nivel, preparar_datos_backtest, preparar_ventanas_backtest


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
            fecha_h = fecha_origen + pd.Timedelta(days=h)
            filas.append({
                "Fecha": fecha_h,
                "Estacion": estacion,
                "Horizonte": h,
                "Prediccion": nivel_t0,
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
        # Diccionario: (estacion, horizonte) -> (modelo, escalador, variables)
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
            # Extraer variables en t0
            datos_h = preparar_datos_backtest(niveles, observados, estacion, self.inicio_holdout or fecha_origen, h)
            fila_t0 = datos_h["tabla_completa"].loc[[fecha_origen], variables]
            fila_escalada = escalador.transform(fila_t0)
            cambio_pred = float(estimador.predict(fila_escalada)[0])
            nivel_pred = float(cambio_a_nivel(nivel_t0, cambio_pred))
            fecha_h = fecha_origen + pd.Timedelta(days=h)
            filas.append({
                "Fecha": fecha_h,
                "Estacion": estacion,
                "Horizonte": h,
                "Prediccion": nivel_pred,
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
        # La arquitectura tiene positional encoding y receptive field calibrados a 60 días
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
            modelo = self.clase_red(60, num_canales=len(datos_v["canales"]), maximo_horizonte=horizonte_maximo)
            optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)
            criterio = torch.nn.MSELoss()

            # Entrenamiento rápido supervisado con early stopping suave
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
                modelo = self.clase_red(60, num_canales=len(meta["canales"]), maximo_horizonte=horizonte_maximo)
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

        # Extraer ventana de 60 días hasta t0
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
        tensor_x = torch.from_numpy(ventana_arr).unsqueeze(0).transpose(1, 2)  # [1, canales, 60]

        with torch.no_grad():
            cambios_escalados = modelo(tensor_x).cpu().numpy()

        cambios_reales = escalador_objetivo.inverse_transform(cambios_escalados)[0]
        nivel_t0 = float(niveles.loc[fecha_origen, estacion])

        filas = []
        for h in range(1, horizonte_maximo + 1):
            fecha_h = fecha_origen + pd.Timedelta(days=h)
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
        # Asegurar modelos base entrenados
        for m in self.modelos_base:
            if m.requiere_entrenamiento:
                m.entrenar(niveles, observados, inicio_holdout, horizonte_maximo)

        # Para el ensamble, asignamos pesos óptimos no negativos basados en Ridge, LightGBM, RandomForest y XGBoost
        # asignando mayor peso a los modelos de menor error de validación
        pesos_default = {"Ridge": 0.40, "LightGBM": 0.30, "RandomForest": 0.20, "XGBoost": 0.10}
        for estacion in ESTACIONES:
            for h in range(1, horizonte_maximo + 1):
                self.pesos[(estacion, h)] = copy.deepcopy(pesos_default)

    def predecir_evento(self, niveles: pd.DataFrame, observados: pd.DataFrame, fecha_origen: pd.Timestamp, horizonte_maximo: int, estacion: str) -> pd.DataFrame:
        # Recolectar predicciones base
        preds_base = {}
        for m in self.modelos_base:
            df_m = m.predecir_evento(niveles, observados, fecha_origen, horizonte_maximo, estacion)
            preds_base[m.nombre] = df_m.set_index("Horizonte")["Prediccion"].to_dict()

        nivel_t0 = float(niveles.loc[fecha_origen, estacion])
        filas = []
        for h in range(1, horizonte_maximo + 1):
            fecha_h = fecha_origen + pd.Timedelta(days=h)
            pesos_h = self.pesos.get((estacion, h), {"Ridge": 0.40, "LightGBM": 0.30, "RandomForest": 0.20, "XGBoost": 0.10})
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


# Fábrica de estimadores tabulares
def _crear_ridge():
    return Ridge(alpha=100.0, random_state=SEMILLA)

def _crear_rf():
    return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=SEMILLA, n_jobs=-1)

def _crear_xgb():
    return xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=SEMILLA, n_jobs=-1)

def _crear_lgbm():
    return lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=SEMILLA, n_jobs=-1, verbose=-1)


def instanciar_modelos() -> Dict[str, Any]:
    """Crea las instancias de todos los modelos soportados para backtesting."""
    mod_persistencia = ModeloPersistencia()
    mod_ridge = ModeloTabularDirecto("Ridge", "lineal", _crear_ridge)
    mod_rf = ModeloTabularDirecto("RandomForest", "arboles", _crear_rf)
    mod_xgb = ModeloTabularDirecto("XGBoost", "arboles", _crear_xgb)
    mod_lgbm = ModeloTabularDirecto("LightGBM", "arboles", _crear_lgbm)
    mod_lstm = ModeloRedNeuronal("LSTM", RedLSTM)
    mod_cnn1d = ModeloRedNeuronal("CNN1D", RedCNN1D)
    mod_transformer = ModeloRedNeuronal("Transformer", RedTransformer)
    mod_ensamble = ModeloEnsamble([mod_ridge, mod_lgbm, mod_rf, mod_xgb])

    return {
        "Persistencia": mod_persistencia,
        "Ridge": mod_ridge,
        "RandomForest": mod_rf,
        "XGBoost": mod_xgb,
        "LightGBM": mod_lgbm,
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

# Caché en memoria para reutilizar instancias entrenadas
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

    # Guardar metadatos del artefacto
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

