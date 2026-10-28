"""Paso 2: comparación por estación entre TODAS las fuentes, en fechas comunes."""
import os
import sys
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

src = common.load_all()
STATIONS = common.STATIONS
rows = []


def compare(sa, sb):
    idx = sa.index.intersection(sb.index)
    if len(idx) < 30:
        return None
    a, b = sa[idx], sb[idx]
    d = a - b
    ident = float((d.abs() < 0.005).mean() * 100)
    # mejor desfase: desplazar b para que b(t+lag) se compare con a(t)
    best = (0, ident, float(d.abs().median()))
    for lag in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
        bb = b.copy()
        bb.index = bb.index - pd.Timedelta(days=lag)
        ii = a.index.intersection(bb.index)
        if len(ii) < 30:
            continue
        dd = a[ii] - bb[ii]
        score = float((dd.abs() < 0.005).mean() * 100)
        mad = float(dd.abs().median())
        if score > best[1] + 0.5 or (abs(score - best[1]) <= 0.5 and mad < best[2]):
            best = (lag, score, mad)
    ratio = float(a.median() / b.median()) if b.median() else np.nan
    return {
        "n_comun": len(idx),
        "fecha_min_comun": idx.min().date().isoformat(),
        "fecha_max_comun": idx.max().date().isoformat(),
        "pct_identicos": round(ident, 4),
        "dif_media": round(float(d.mean()), 4),
        "dif_mediana": round(float(d.median()), 4),
        "dif_max_abs": round(float(d.abs().max()), 4),
        "mejor_lag_dias": best[0],
        "pct_identicos_mejor_lag": round(best[1], 4),
        "mad_mejor_lag": round(best[2], 4),
        "ratio_medianas": round(ratio, 4),
        "posible_x100": bool(abs(ratio - 100) < 5 or abs(ratio - 0.01) < 0.001),
    }


for st in STATIONS:
    series = common.series(src, st)
    names = sorted(series.keys())
    for a, b in itertools.combinations(names, 2):
        r = compare(series[a], series[b])
        if r is None:
            continue
        r.update({"estacion": st, "fuente_a": a, "fuente_b": b})
        rows.append(r)

df = pd.DataFrame(rows)[[
    "estacion", "fuente_a", "fuente_b", "n_comun", "fecha_min_comun", "fecha_max_comun",
    "pct_identicos", "dif_media", "dif_mediana", "dif_max_abs",
    "mejor_lag_dias", "pct_identicos_mejor_lag", "mad_mejor_lag",
    "ratio_medianas", "posible_x100"]]
df.to_csv(os.path.join(common.SALIDA, "comparacion_pares.csv"), index=False)

# Resumen: cada fuente contra el raw del libro
RAW = {"Ayacucho": "DATOS:Puerto Ayacucho", "Caicara": "DATOS:Caicara",
       "Ciudad_Bolivar": "DATOS:Ciudad Bolivar", "Palua": "DATOS:Palua"}
summ = df[df.apply(lambda r: r["fuente_a"] == RAW[r["estacion"]] or
                   r["fuente_b"] == RAW[r["estacion"]], axis=1)].copy()
summ["fuente"] = summ.apply(
    lambda r: r["fuente_b"] if r["fuente_a"] == RAW[r["estacion"]] else r["fuente_a"], axis=1)
summ = summ[["estacion", "fuente", "n_comun", "pct_identicos", "dif_media",
             "dif_max_abs", "mejor_lag_dias", "pct_identicos_mejor_lag", "ratio_medianas"]]
summ = summ.sort_values(["estacion", "fuente"])
summ.to_csv(os.path.join(common.SALIDA, "comparacion_vs_raw.csv"), index=False)

print("Pares comparados:", len(df))
print("\n=== Cada fuente vs raw del libro ===")
print(summ.to_string(index=False))
print("\nPares con mejor lag != 0 (posible desfase de fechas):")
print(df[(df["mejor_lag_dias"] != 0) & (df["pct_identicos_mejor_lag"] > 50)].to_string(index=False))
print("\nPares con posible factor x100:", int(df["posible_x100"].sum()))
print("\nPares con diferencia media |>0.05|:")
print(df[df["dif_media"].abs() > 0.05].to_string(index=False))
