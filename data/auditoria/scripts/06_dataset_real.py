"""Paso 6: dataset_real.csv con solo valores reales y limpios."""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

src = common.load_all()
RAW = {"Ayacucho": "DATOS:Puerto Ayacucho", "Caicara": "DATOS:Caicara",
       "Ciudad_Bolivar": "DATOS:Ciudad Bolivar", "Palua": "DATOS:Palua"}
S = {st: common.series(src, st)[RAW[st]] for st in common.STATIONS}

# días (b) detectados en el paso 4
b = pd.read_csv(os.path.join(common.SALIDA, "dias_b_eliminar.csv"))
print("Días (b) a eliminar:", len(b))

start = min(s.index.min() for s in S.values())
end = max(s.index.max() for s in S.values())
idx = pd.date_range(start, end, freq="D")
print("Rango:", start.date(), "->", end.date(), "días:", len(idx))

out = pd.DataFrame({"Fecha": idx})
for st in common.STATIONS:
    out[st] = S[st].reindex(idx).values
    bad = b[b.estacion == st]["fecha"]
    mask = out["Fecha"].isin(pd.to_datetime(bad))
    n_bad = int((mask & out[st].notna()).sum())
    out.loc[mask, st] = np.nan
    print(f"{st}: {n_bad} valores (b) puestos a NaN")

out["Fecha"] = out["Fecha"].dt.strftime("%Y-%m-%d")
path = os.path.join(common.SALIDA, "dataset_real.csv")
out.to_csv(path, index=False)

# ---------------- Verificaciones ----------------
print("\n=== Verificaciones ===")
print("filas:", len(out), "columnas:", list(out.columns))
print("fechas duplicadas:", int(out["Fecha"].duplicated().sum()))
print("orden diario:", bool((pd.to_datetime(out["Fecha"]).diff().dropna() == pd.Timedelta(days=1)).all()))
print("\nNaN por columna:")
print(out[common.STATIONS].isna().sum().to_string())
print("\nPrimer/último dato real y min/max:")
for st in common.STATIONS:
    s = out[st].dropna()
    d = pd.to_datetime(out.loc[s.index, "Fecha"])
    print(f"{st}: {d.min().date()}..{d.max().date()}, n={len(s)}, min={s.min()}, max={s.max()}")

print("\nPuntos de control:")
checks = [
    ("Ciudad_Bolivar", "1987-05-14", 7.52), ("Ciudad_Bolivar", "1987-05-15", 5.32),
    ("Ciudad_Bolivar", "2018-12-31", 5.60), ("Ciudad_Bolivar", "2019-01-01", 6.60),
    ("Palua", "1974-04-08", 2.12), ("Palua", "1974-04-09", 1.88),
    ("Ayacucho", "2007-06-01", np.nan), ("Ayacucho", "2008-12-01", 46.79),
    ("Ayacucho", "2024-02-29", 40.70), ("Ciudad_Bolivar", "2024-02-29", 3.54),
    ("Palua", "2024-02-29", 1.31),
    ("Caicara", "1969-04-01", 21.93), ("Caicara", "1969-04-03", np.nan),
    ("Caicara", "1969-04-07", 22.02), ("Ayacucho", "1983-01-02", np.nan),
    ("Ayacucho", "1983-01-03", 42.74),
]
for st, d, exp in checks:
    v = out.loc[out["Fecha"] == d, st].iloc[0]
    ok = (pd.isna(v) and pd.isna(exp)) or (not pd.isna(v) and not pd.isna(exp) and abs(float(v) - exp) < 1e-9)
    print(f"{st:15s} {d} -> {v} (esperado {exp}) {'OK' if ok else 'REVISAR'}")

# ---------------- Resumen ----------------
summary = {
    "archivo": "salida/dataset_real.csv",
    "filas": int(len(out)),
    "rango": [out["Fecha"].iloc[0], out["Fecha"].iloc[-1]],
    "no_nulos": {st: int(out[st].notna().sum()) for st in common.STATIONS},
    "nan": {st: int(out[st].isna().sum()) for st in common.STATIONS},
    "b_eliminados": int(len(b)),
    "nota": "No se corrigen (a); los cambios de referencia (c) se dejan tal cual y solo se reportan.",
}
with open(os.path.join(common.SALIDA, "dataset_real_resumen.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\nGuardado:", path)
