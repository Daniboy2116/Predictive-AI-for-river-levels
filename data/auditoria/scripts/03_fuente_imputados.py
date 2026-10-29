"""Paso 3: fuente original por estación e identificación de valores imputados."""
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
raw_series = {st: common.series(src, st)[RAW[st]] for st in common.STATIONS}

DATASETS = ["dataset_combinado.csv", "dataset_hasta_2006.csv",
            "dataset_imputado_simpleml.csv", "dataset_imputado_simpleml.xlsx",
            "dataset-orinoco-imputado (1).xlsx", "DATOS:Entrenar"]

rows = []
blocks = []
for st in common.STATIONS:
    rs = raw_series[st]
    for name in DATASETS:
        s = common.series(src, st).get(name)
        if s is None:
            continue
        common_dates = s.index.intersection(rs.index)
        d = s[common_dates] - rs[common_dates]
        no_raw = s.index.difference(rs.index)
        missing_raw = rs.index.difference(s.index)
        rows.append({
            "estacion": st, "fuente": name,
            "fechas": len(s),
            "fechas_con_raw": len(common_dates),
            "identicas": int((d.abs() < 0.005).sum()),
            "distintas": int((d.abs() >= 0.005).sum()),
            "fechas_sin_raw": len(no_raw),
            "fechas_raw_ausentes": len(missing_raw),
        })
        # bloques contiguos de fechas sin raw (imputadas)
        if len(no_raw):
            no_raw = no_raw.sort_values()
            start = prev = no_raw[0]
            for t in no_raw[1:]:
                if (t - prev).days > 1:
                    blocks.append((st, name, start.date().isoformat(), prev.date().isoformat(),
                                   (prev - start).days + 1))
                    start = t
                prev = t
            blocks.append((st, name, start.date().isoformat(), prev.date().isoformat(),
                           (prev - start).days + 1))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(common.SALIDA, "imputados_resumen.csv"), index=False)

bdf = pd.DataFrame(blocks, columns=["estacion", "fuente", "desde", "hasta", "dias"])
bdf.to_csv(os.path.join(common.SALIDA, "imputados_bloques.csv"), index=False)

print("=== Cobertura de cada dataset respecto del raw ===")
print(df.to_string(index=False))
print("\n=== Bloques de fechas sin dato raw (valores imputados o artefactos) ===")
print(bdf.to_string(index=False))

# Describir el bloque imputado de Ayacucho 2007-2008
imp = common.series(src, "Ayacucho")["dataset_imputado_simpleml.csv"]
raw = raw_series["Ayacucho"]
gap = imp.index.difference(raw.index)
gap = gap[(gap >= "2007-01-01") & (gap <= "2008-12-31")]
print("\nBloque imputado Ayacucho 2007-2008:", len(gap), "días",
      gap.min().date(), gap.max().date())
# continuidad: diffs diarios del bloque
gser = imp[imp.index.isin(gap)]
print("  min/max/media/mediana del bloque:", round(gser.min(), 2), round(gser.max(), 2),
      round(gser.mean(), 2), round(gser.median(), 2))
print("  dif diaria |>0.5| dentro del bloque:", int((gser.diff().abs() > 0.5).sum()))
print("  valores repetidos consecutivos >=5 días:", int((gser.diff() == 0).astype(int).groupby((gser.diff() != 0).cumsum()).sum().max()))

# Comparación con los años vecinos (2006 y 2009): el bloque imputado debe ser sintético
for y in [2006, 2007, 2008, 2009]:
    yy = imp[imp.index.year == y]
    print(f"  {y}: n={len(yy)} media={yy.mean():.2f} min={yy.min():.2f} max={yy.max():.2f}")

# Feb 29 interpolados en los CSV raw (Ayacucho/Palua)
print("\n=== 29-feb presentes en CSV raw pero NO en la hoja del libro ===")
for st, fname in [("Ayacucho", "ayacucho_raw.csv"), ("Palua", "palua_raw.csv")]:
    s = common.series(src, st)[fname]
    rs = raw_series[st]
    extra = s.index.difference(rs.index)
    print(st, len(extra), [x.date().isoformat() for x in extra])
    for t in extra:
        prev = s.get(t - pd.Timedelta(days=1)); nxt = s.get(t + pd.Timedelta(days=1))
        print("   ", t.date(), "valor=", s.get(t), "interp=", None if pd.isna(prev) or pd.isna(nxt)
              else round((prev + nxt) / 2, 3), "dif=", None if pd.isna(prev) or pd.isna(nxt)
              else round(s.get(t) - (prev + nxt) / 2, 3))

# Entrenar: desfase local de Ciudad Bolívar
print("\n=== Entrenar: desfase de fechas en Ciudad Bolívar ===")
e = common.series(src, "Ciudad_Bolivar")["DATOS:Entrenar"]
r = raw_series["Ciudad_Bolivar"]
d0 = (e - r).dropna()
bad = d0[d0.abs() >= 0.005]
cb_bad = bad
print("fechas distintas:", len(bad), bad.index.min().date(), bad.index.max().date())
ee = e.copy(); ee.index = ee.index + pd.Timedelta(days=1)  # e(t) = r(t+1)
ii = ee.index.intersection(r.index)
sub = (ee[ii] - r[ii])
mask = (sub.index >= "2020-01-01") & (sub.index <= "2021-12-30")
print("e(t) vs r(t+1) en 2020-01-01..2021-12-30: ident% =",
      round(100 * (sub[mask].abs() < 0.005).mean(), 2), "n =", int(mask.sum()))
print("e(t) vs r(t) en el mismo tramo: ident% =",
      round(100 * (d0[(d0.index >= "2020-01-01") & (d0.index <= "2021-12-30")].abs() < 0.005).mean(), 2))
for t in ["2019-12-31", "2020-01-01", "2020-01-02", "2021-12-30", "2021-12-31", "2022-01-01"]:
    tt = pd.Timestamp(t)
    print("  ", t, "entrenar=", e.get(tt), "raw(t)=", r.get(tt), "raw(t+1)=", r.get(tt + pd.Timedelta(days=1)))

# Otras diferencias de Entrenar frente al raw
for st in common.STATIONS:
    es = common.series(src, st)["DATOS:Entrenar"]
    rr = raw_series[st]
    ii = es.index.intersection(rr.index)
    dd = es[ii] - rr[ii]
    bad = dd[dd.abs() >= 0.005]
    print(f"Entrenar {st}: fechas distintas={len(bad)} | max={bad.abs().max() if len(bad) else 0}",
          [x.date().isoformat() for x in bad.index[:5]])

print("\n=== Duplicados exactos ===")
print("imputado csv == xlsx:",
      common.series(src, "Ayacucho")["dataset_imputado_simpleml.csv"].equals(
          common.series(src, "Ayacucho")["dataset_imputado_simpleml.xlsx"]))
print("imputado csv == (1).xlsx:",
      common.series(src, "Ayacucho")["dataset_imputado_simpleml.csv"].equals(
          common.series(src, "Ayacucho")["dataset-orinoco-imputado (1).xlsx"]))
c = common.series(src, "Ayacucho")["dataset_combinado.csv"]
c06 = c[c.index <= "2006-12-31"]
print("combinado<=2006 == hasta_2006:",
      c06.equals(common.series(src, "Ayacucho")["dataset_hasta_2006.csv"]))

summary = {
    "fuente_raw": RAW,
    "imputados_por_estacion": df.groupby("estacion")["fechas_sin_raw"].max().to_dict(),
    "entrenar_cb_fechas_desplazadas": int(len(cb_bad)),
    "entrenar_cb_rango_desfase": [cb_bad.index.min().date().isoformat(), cb_bad.index.max().date().isoformat()],
}
with open(os.path.join(common.SALIDA, "fuente_imputados.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
