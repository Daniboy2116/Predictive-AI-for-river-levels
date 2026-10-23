"""Genera un dataset simulado de niveles de ríos en data/datos_rios.xlsx.

Simula la dinámica típica del Orinoco: crecida en temporada de lluvias
(pico ~agosto) y estiaje (mínimo ~marzo), con tendencia leve, ruido
autocorrelacionado AR(1) y ~2% de valores faltantes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = Path(__file__).parent / "data"
OUTPUT = DATA_DIR / "datos_rios.xlsx"

# estación: (nivel medio m, amplitud m, desfase días, sigma ruido)
STATIONS = {
    "Palua": (8.0, 5.5, 0, 0.25),
    "Orinoco": (10.0, 6.5, 5, 0.30),
    "Caicara": (9.5, 6.2, 4, 0.28),
    "Ciudad_Bolivar": (9.0, 6.0, 3, 0.28),
    "Ayacucho": (5.5, 4.0, -10, 0.20),
}


def ar1_noise(n: int, sigma: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Ruido autorregresivo de orden 1."""
    e = rng.normal(0, sigma, n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + e[i]
    return out


def generate(missing_frac: float = 0.02) -> pd.DataFrame:
    """Construye el DataFrame con columna Fecha y una columna por estación."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear.to_numpy()
    t = np.arange(n)
    df = pd.DataFrame({"Fecha": dates})
    for name, (mean, amp, shift, sigma) in STATIONS.items():
        # pico cerca del día 225 (mediados de agosto)
        seasonal = amp * np.cos(2 * np.pi * (doy - 225 - shift) / 365.25)
        trend = 0.0002 * t
        interannual = 0.6 * np.sin(2 * np.pi * t / (365.25 * 3.5))  # ciclo tipo ENSO
        level = mean + seasonal + trend + interannual + ar1_noise(n, sigma, 0.9, rng)
        level = np.clip(level, 0.1, None).round(3)
        mask = rng.random(n) < missing_frac
        level = level.astype(float)
        level[mask] = np.nan
        df[name] = level
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_excel(OUTPUT, index=False, sheet_name="Niveles")
    print(f"Dataset generado: {OUTPUT} -> {df.shape[0]} filas, columnas {list(df.columns)}")
    print(f"Valores faltantes por estación:\n{df.isna().sum()}")


if __name__ == "__main__":
    main()
