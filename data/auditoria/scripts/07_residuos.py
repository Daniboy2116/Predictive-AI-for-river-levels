"""Residuos anuales estación-vecina para verificar permanencia de los escalones (c)."""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

src = common.load_all()
RAW = {"Ayacucho": "DATOS:Puerto Ayacucho", "Caicara": "DATOS:Caicara",
       "Ciudad_Bolivar": "DATOS:Ciudad Bolivar", "Palua": "DATOS:Palua"}
S = {st: common.series(src, st)[RAW[st]] for st in common.STATIONS}
PAIRS = [("Ciudad_Bolivar", "Palua", 0), ("Ayacucho", "Ciudad_Bolivar", 18),
         ("Caicara", "Ciudad_Bolivar", 4)]

rows = []
for st, nb, lag in PAIRS:
    nbal = S[nb].copy(); nbal.index = nbal.index - pd.Timedelta(days=lag)
    res = (S[st] - nbal).dropna()
    annual = res.groupby(res.index.year).mean()
    for y, v in annual.items():
        rows.append({"par": f"{st}-{nb}", "anio": int(y), "residuo_medio": round(float(v), 3)})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(common.SALIDA, "residuo_anual.csv"), index=False)

print("=== Residuo anual (m) ===")
for st, nb, lag in PAIRS:
    a = df[df.par == f"{st}-{nb}"].set_index("anio")["residuo_medio"]
    print(f"\n{st} - {nb} (lag {lag} d)")
    years = sorted(a.index)
    # mostrar todos los años con |cambio| > 0.3 o los años clave
    key = {"Ciudad_Bolivar-Palua": [1986, 1987, 1988, 2018, 2019, 2020],
           "Ayacucho-Ciudad_Bolivar": [1985, 1986, 1987, 2019, 2020, 2021],
           "Caicara-Ciudad_Bolivar": [1986, 1987, 1988]}[f"{st}-{nb}"]
    for y in key:
        if y in a.index:
            print(f"  {y}: {a.loc[y]:+.3f}")

print("\nMayores cambios interanuales del residuo (|dif| > 0.5):")
for st, nb, lag in PAIRS:
    a = df[df.par == f"{st}-{nb}"].set_index("anio")["residuo_medio"]
    d = a.diff()
    for y, v in d[d.abs() > 0.5].items():
        prev = f"{a.loc[y-1]:+.3f}" if (y - 1) in a.index else "sin dato"
        print(f"  {st}-{nb} {int(y)}: {v:+.3f} (de {prev} a {a.loc[y]:+.3f})")
