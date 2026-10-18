"""Preprocesamiento de series temporales de niveles de ríos.

Flujo (sin *data leakage*):
1. Cargar Excel, fijar ``Fecha`` como índice ``DatetimeIndex`` diario.
2. Imputar NaN con interpolación temporal lineal.
3. Dividir cronológicamente la serie cruda en 70/15/15.
4. Ajustar el escalador SOLO con train y transformar toda la serie.
5. Construir lags t-1..t-n y separar X/y por partición.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "datos_rios.xlsx"
SCALER_DIR = ROOT / "outputs" / "scalers"
STATIONS = ["Palua", "Orinoco", "Caicara", "Ciudad_Bolivar", "Ayacucho"]
SEED = 42

ScalerType = Literal["minmax", "zscore"]


def load_data(path: Path | str = DATA_PATH, date_col: str = "Fecha") -> pd.DataFrame:
    """Carga el Excel y devuelve un DataFrame indexado por fecha (frecuencia diaria)."""
    df = pd.read_excel(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df[~df.index.duplicated(keep="first")].asfreq("D")
    return df.apply(pd.to_numeric, errors="coerce")


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolación lineal temporal; rellena extremos con el valor válido más cercano."""
    return df.interpolate(method="time", limit_direction="both").ffill().bfill()


def split_indices(n: int, train: float = 0.70, val: float = 0.15) -> tuple[int, int]:
    """Devuelve los índices de corte (fin_train, fin_val) para un split cronológico."""
    i_train = int(n * train)
    i_val = int(n * (train + val))
    return i_train, i_val


def make_lag_features(series: pd.Series, n_lags: int = 7) -> pd.DataFrame:
    """Transforma una serie en problema supervisado con columnas lag_1..lag_n y target."""
    frame = pd.DataFrame({"target": series})
    for k in range(1, n_lags + 1):
        frame[f"lag_{k}"] = series.shift(k)
    return frame.dropna()


def _build_scaler(kind: ScalerType) -> MinMaxScaler | StandardScaler:
    """Crea el escalador solicitado."""
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "zscore":
        return StandardScaler()
    raise ValueError(f"Escalador desconocido: {kind!r} (use 'minmax' o 'zscore')")


@dataclass
class StationData:
    """Contenedor de datos preparados para una estación."""

    station: str
    raw: pd.Series                      # serie imputada en escala original
    scaler: MinMaxScaler | StandardScaler
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    lag_cols: list[str] = field(default_factory=list)

    def inverse(self, values: np.ndarray | pd.Series) -> np.ndarray:
        """Desnormaliza valores a la escala original (metros)."""
        arr = np.asarray(values, dtype=float).reshape(-1, 1)
        return self.scaler.inverse_transform(arr).ravel()

    @property
    def raw_train(self) -> pd.Series:
        """Serie original de entrenamiento (para ARIMA/Prophet)."""
        return self.raw.loc[: self.y_train.index[-1]]

    @property
    def raw_val(self) -> pd.Series:
        return self.raw.loc[self.y_val.index]

    @property
    def raw_test(self) -> pd.Series:
        return self.raw.loc[self.y_test.index]


def prepare_station(
    df: pd.DataFrame,
    station: str,
    n_lags: int = 7,
    scaler_type: ScalerType = "minmax",
    save_scaler: bool = True,
) -> StationData:
    """Prepara una estación: imputación, split 70/15/15, escalado (fit en train) y lags."""
    if station not in df.columns:
        raise KeyError(f"Estación {station!r} no encontrada. Disponibles: {list(df.columns)}")
    series = impute(df[[station]])[station]

    i_train, i_val = split_indices(len(series))
    scaler = _build_scaler(scaler_type)
    scaler.fit(series.iloc[:i_train].to_frame())  # SOLO train
    scaled = pd.Series(
        scaler.transform(series.to_frame()).ravel(), index=series.index, name=station
    )

    sup = make_lag_features(scaled, n_lags)
    lag_cols = [f"lag_{k}" for k in range(1, n_lags + 1)]
    train_end, val_end = series.index[i_train - 1], series.index[i_val - 1]
    tr = sup.loc[:train_end]
    va = sup.loc[train_end + pd.Timedelta(days=1) : val_end]
    te = sup.loc[val_end + pd.Timedelta(days=1) :]

    if save_scaler:
        SCALER_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_DIR / f"{station}_{scaler_type}.joblib")

    return StationData(
        station=station, raw=series, scaler=scaler,
        X_train=tr[lag_cols], y_train=tr["target"],
        X_val=va[lag_cols], y_val=va["target"],
        X_test=te[lag_cols], y_test=te["target"],
        lag_cols=lag_cols,
    )


def load_scaler(station: str, scaler_type: ScalerType = "minmax"):
    """Carga un escalador previamente guardado."""
    return joblib.load(SCALER_DIR / f"{station}_{scaler_type}.joblib")


def make_sequences(X: pd.DataFrame) -> np.ndarray:
    """Convierte lags (lag_1..lag_n) en tensores (muestras, pasos, 1) en orden cronológico."""
    return X.iloc[:, ::-1].to_numpy(dtype=np.float32)[..., np.newaxis]


if __name__ == "__main__":
    data = load_data()
    print(f"Datos: {data.shape}, NaN totales: {int(data.isna().sum().sum())}")
    for st in STATIONS:
        d = prepare_station(data, st)
        assert d.y_train.index.max() < d.y_val.index.min() < d.y_test.index.min()
        assert not d.X_train.isna().any().any()
        rec = d.inverse(d.y_test.values[:3])
        assert np.allclose(rec, d.raw_test.values[:3])
        print(
            f"{st:15s} train={len(d.y_train)} val={len(d.y_val)} test={len(d.y_test)} "
            f"| test: {d.y_test.index.min().date()} → {d.y_test.index.max().date()}"
        )
    print("Preprocesamiento validado ✔")
