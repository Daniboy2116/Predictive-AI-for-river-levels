"""Paso 5: cobertura real por año/estación y propuesta de rango + partición."""
import os
import sys
import json
import calendar
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

src = common.load_all()
RAW = {"Ayacucho": "DATOS:Puerto Ayacucho", "Caicara": "DATOS:Caicara",
       "Ciudad_Bolivar": "DATOS:Ciudad Bolivar", "Palua": "DATOS:Palua"}
S = {st: common.series(src, st)[RAW[st]] for st in common.STATIONS}

start_all = min(s.index.min() for s in S.values()).date()
end_all = max(s.index.max() for s in S.values()).date()
print("Rango total:", start_all, "->", end_all)

# ---------------- Cobertura anual ----------------
rows = []
for year in range(start_all.year, end_all.year + 1):
    days = 366 if calendar.isleap(year) else 365
    row = {"anio": year, "dias_anio": days}
    for st, s in S.items():
        first = s.index.min().date()
        last = s.index.max().date()
        if year < first.year or year > last.year:
            row[st] = np.nan
            row[st + "_dias"] = np.nan
            row[st + "_faltan"] = np.nan
            continue
        n = int(s.loc[f"{year}-01-01":f"{year}-12-31"].notna().sum())
        row[st] = round(100.0 * n / days, 2)
        row[st + "_dias"] = n
        row[st + "_faltan"] = days - n
    rows.append(row)
cov = pd.DataFrame(rows)
cov.to_csv(os.path.join(common.SALIDA, "cobertura_anual.csv"), index=False)

print("\n=== Cobertura por estación ===")
for st, s in S.items():
    n = int(s.notna().sum())
    first, last = s.index.min().date(), s.index.max().date()
    span = (last - first).days + 1
    print(f"{st}: {first}..{last}, reales={n}, rango={span}, cobertura={100*n/span:.2f}%")

# ---------------- Matriz 1974-2025 ----------------
print("\n=== Matriz de cobertura 1974-2025 (%) ===")
mat = cov[(cov.anio >= 1974) & (cov.anio <= 2025)][["anio"] + common.STATIONS].copy()
mat = mat.set_index("anio")
print(mat.to_string(float_format=lambda x: f"{x:6.2f}"))

# ---------------- Evaluación de rangos candidatos ----------------
def eval_range(y0, y1):
    out = {"desde": y0, "hasta": y1, "anios": y1 - y0 + 1}
    complete_all = 0
    for year in range(y0, y1 + 1):
        all_ok = True
        for st, s in S.items():
            if year < s.index.min().year or year > s.index.max().year:
                all_ok = False
                continue
            days = 366 if calendar.isleap(year) else 365
            n = int(s.loc[f"{year}-01-01":f"{year}-12-31"].notna().sum())
            if days - n > 1:
                all_ok = False
        if all_ok:
            complete_all += 1
    out["anios_completos_4"] = complete_all
    for st, s in S.items():
        sub = s.loc[f"{y0}-01-01":f"{y1}-12-31"]
        days = (pd.Timestamp(f"{y1}-12-31") - pd.Timestamp(f"{y0}-01-01")).days + 1
        out[st] = round(100.0 * sub.notna().sum() / days, 3)
    out["min_cobertura"] = min(out[st] for st in common.STATIONS)
    return out

cands = [(1963, 2006), (1968, 2006), (1974, 2006), (1974, 2024),
         (1988, 2006), (1988, 2024), (2000, 2024), (2009, 2024),
         (2010, 2024), (1974, 2018)]
rng = pd.DataFrame([eval_range(a, b) for a, b in cands])
rng.to_csv(os.path.join(common.SALIDA, "cobertura_rangos.csv"), index=False)
print("\n=== Rangos candidatos ===")
print(rng.to_string(index=False))

# ---------------- Rango recomendado y partición ----------------
REC = (2009, 2024)
part = [(2009, 2018, "train"), (2019, 2021, "validacion"), (2022, 2024, "prueba")]
prows = []
for y0, y1, name in part:
    row = {"particion": name, "desde": y0, "hasta": y1, "anios": y1 - y0 + 1}
    for st, s in S.items():
        sub = s.loc[f"{y0}-01-01":f"{y1}-12-31"]
        days = (pd.Timestamp(f"{y1}-12-31") - pd.Timestamp(f"{y0}-01-01")).days + 1
        row[st] = round(100.0 * sub.notna().sum() / days, 3)
    row["dias_totales"] = days
    prows.append(row)
pdf = pd.DataFrame(prows)
pdf.to_csv(os.path.join(common.SALIDA, "particion_recomendada.csv"), index=False)
print("\n=== Partición recomendada", REC, "===")
print(pdf.to_string(index=False))

# comprobar que el cambio de referencia 2019-01-01 cae en validación
print("\n2019-01-01 dentro de validación:", part[1][0] <= 2019 <= part[1][1])

# ---------------- Huecos largos (>1 día) por estación ----------------
print("\n=== Huecos internos >1 día ===")
gaps = []
for st, s in S.items():
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    miss = full.difference(s.index)
    if len(miss) == 0:
        continue
    start = prev = miss[0]
    for t in miss[1:]:
        if (t - prev).days == 1:
            prev = t
            continue
        gaps.append({"estacion": st, "desde": start.date().isoformat(),
                     "hasta": prev.date().isoformat(), "dias": (prev - start).days + 1})
        start = prev = t
    gaps.append({"estacion": st, "desde": start.date().isoformat(),
                 "hasta": prev.date().isoformat(), "dias": (prev - start).days + 1})
gapdf = pd.DataFrame(gaps).sort_values(["estacion", "dias"], ascending=[True, False])
gapdf.to_csv(os.path.join(common.SALIDA, "huecos.csv"), index=False)
print(gapdf[gapdf.dias >= 2].to_string(index=False))

summary = {
    "rango_total": [str(start_all), str(end_all)],
    "recomendado": {"desde": REC[0], "hasta": REC[1],
                    "particion": [{"nombre": n, "desde": a, "hasta": b} for a, b, n in part]},
    "cobertura_total": {st: round(100.0 * s.notna().sum() / ((s.index.max() - s.index.min()).days + 1), 3)
                        for st, s in S.items()},
}
with open(os.path.join(common.SALIDA, "cobertura_resumen.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
