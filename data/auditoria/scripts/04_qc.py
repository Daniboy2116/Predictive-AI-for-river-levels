"""Paso 4: control de calidad de la serie real (saltos, picos, escalones, constantes)."""
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
CB = S["Ciudad_Bolivar"]


def aligned_neighbor(st):
    lag = {"Ayacucho": 18, "Caicara": 4, "Ciudad_Bolivar": 0, "Palua": 0}[st]
    nb = CB if st != "Ciudad_Bolivar" else S["Palua"]
    return nb, lag


def mean_between(s, a, b):
    x = s.loc[a:b].dropna()
    return float(x.mean()) if len(x) else np.nan


# ---------------- 1. Todos los saltos > 0.5 m ----------------
jumps = []
for st, s in S.items():
    d = s.diff()
    nb, lag = aligned_neighbor(st)
    # residuo estación - vecina con el retardo indicado
    nbal = nb.copy(); nbal.index = nbal.index - pd.Timedelta(days=lag)
    res = (s - nbal).dropna()
    for t in d[d.abs() > 0.5].index:
        prev = s.get(t - pd.Timedelta(days=1), np.nan)
        nxt = s.get(t + pd.Timedelta(days=1), np.nan)
        tn = t + pd.Timedelta(days=lag)
        nd = nb.get(tn, np.nan) - nb.get(tn - pd.Timedelta(days=1), np.nan)
        pre5 = mean_between(res, t - pd.Timedelta(days=6), t - pd.Timedelta(days=1))
        post5 = mean_between(res, t + pd.Timedelta(days=1), t + pd.Timedelta(days=5))
        post30 = mean_between(res, t + pd.Timedelta(days=15), t + pd.Timedelta(days=30))
        post90 = mean_between(res, t + pd.Timedelta(days=60), t + pd.Timedelta(days=90))
        jumps.append({
            "estacion": st, "fecha": t.date().isoformat(),
            "valor": round(float(s[t]), 3),
            "anterior": round(float(prev), 3) if not np.isnan(prev) else np.nan,
            "siguiente": round(float(nxt), 3) if not np.isnan(nxt) else np.nan,
            "salto": round(float(d[t]), 3),
            "salto_vecina": round(float(nd), 3) if not np.isnan(nd) else np.nan,
            "resid_5d": round(post5 - pre5, 3) if not np.isnan(post5) else np.nan,
            "resid_30d": round(post30 - pre5, 3) if not np.isnan(post30) else np.nan,
            "resid_90d": round(post90 - pre5, 3) if not np.isnan(post90) else np.nan,
        })
jdf = pd.DataFrame(jumps).sort_values(["estacion", "fecha"]).reset_index(drop=True)

# ---------------- 2. Clasificación curada ----------------
# (b) días aberrantes que vuelven al nivel previo (se eliminan a NaN)
B_DAYS = {
    "Ayacucho": ["1982-06-07", "1983-01-02", "1989-09-17", "1992-12-27",
                 "1994-04-08", "1994-12-29", "1997-01-30", "2006-09-24"],
    "Caicara": ["1969-01-27", "1969-01-28", "1969-04-02", "1969-04-03",
                "1969-04-04", "1969-04-05", "1969-04-06", "1969-11-29"],
    "Ciudad_Bolivar": ["1979-11-21", "1997-07-13", "2000-01-20"],
    "Palua": ["1949-05-11", "1970-07-10", "1971-04-27", "1972-03-03",
              "1972-03-20", "1974-02-04", "1975-07-25", "1988-12-02",
              "2021-10-29"],
}
# (a) pulsos positivos en Palúa no presentes en Ciudad Bolívar
A_EVENTS = {
    "1971-04-23": "pulso +0.58 m; vecina +0.12; residuo +0.62 a 5 d y -0.86 a 30 d",
    "1971-05-18": "pulso +0.58 m; vecina +0.26; residuo +0.43 a 5 d y -0.30 a 30 d",
    "1972-05-01": "pulso +1.16 m; vecina +0.30; residuo +0.27 a 5 d y -0.43 a 30 d",
    "1974-04-08": "pulso +0.73 m; vecina 0.00; residuo +0.65 a 5 d y -1.02 a 30 d",
    "1983-05-27": "pulso +0.51 m; vecina +0.09; residuo +0.54 a 5 d y -0.38 a 90 d",
    "1994-07-17": "pulso +0.58 m; vecina +0.11; residuo +0.46 a 5 d y -0.07 a 30 d",
    "1997-06-28": "pulso +0.58 m; vecina +0.10; residuo +0.63 a 5 d y +0.12 a 30 d",
    "2019-12-19": "pulso +0.58 m; vecina -0.12; residuo +0.60 a 5 d y +0.35 a 30 d",
}
# (c) escalones no replicados por la vecina
C_EVENTS = {
    "1987-05-15": "Ciudad_Bolivar: -2.20 m exactos; Palúa +0.12; cambio de referencia conocido",
    "2019-01-01": "Ciudad_Bolivar: +1.00 m exactos; Palúa -0.18; cambio de referencia conocido",
    "2019-01-11": "Ciudad_Bolivar: -0.54 m; forma parte de la recuperación del escalón de 2019-01-01",
    "1986-10-01": "Ayacucho: -1.10 m; vecina (CB+18) 0.00; escalón no replicado; no persiste al año",
    "1987-01-01": "Caicara: -1.02 m; vecina (CB+4) -0.19; escalón no replicado; no persiste al año",
    "1972-04-04": "Palúa: -0.94 m; CB -0.01; escalón no replicado; se recupera en meses",
    "2000-11-17": "Ciudad_Bolivar: -0.56 m; Palúa +0.20; escalón no replicado; no persiste al año",
    "2020-06-20": "Ayacucho: +0.84 m; vecina (CB+18) +0.14; salto aislado no replicado; no persiste al año",
}


def classify(st, date):
    if any(date == d for d in B_DAYS.get(st, [])):
        return "b"
    if st == "Palua" and date in A_EVENTS:
        return "a"
    if date in C_EVENTS and (
            (st == "Ciudad_Bolivar" and date.startswith(("1987-05-15", "2019-01-01", "2019-01-11", "2000-11-17")))
            or (st == "Ayacucho" and date in ("1986-10-01", "2020-06-20"))
            or (st == "Caicara" and date == "1987-01-01")
            or (st == "Palua" and date == "1972-04-04")):
        return "c"
    return "d"


jdf["clase"] = [classify(st, d) for st, d in zip(jdf["estacion"], jdf["fecha"])]
jdf["evidencia"] = [
    C_EVENTS.get(d, "") if c == "c" else (A_EVENTS.get(d, "") if c == "a" else
    ("pico aislado que vuelve al nivel previo" if c == "b" else "ascenso/descenso rápido de temporada, también presente en la vecina"))
    for d, c in zip(jdf["fecha"], jdf["clase"])]

jdf.to_csv(os.path.join(common.SALIDA, "saltos_mayores_05.csv"), index=False)

# Tabla curada de eventos sospechosos (un evento por día o bloque)
sosp = []
for st, days in B_DAYS.items():
    for d in days:
        s = S[st]
        t = pd.Timestamp(d)
        prev = s.get(t - pd.Timedelta(days=1), np.nan)
        nxt = s.get(t + pd.Timedelta(days=1), np.nan)
        spike = s.get(t) - (prev + nxt) / 2 if not (np.isnan(prev) or np.isnan(nxt)) else np.nan
        sosp.append({"estacion": st, "fecha": d, "clase": "b",
                     "tamano_m": round(float(spike), 3),
                     "evidencia": f"valor={s.get(t):.2f}; anterior={prev:.2f}; siguiente={nxt:.2f}; retorna al nivel previo"})
for d, ev in A_EVENTS.items():
    sosp.append({"estacion": "Palua", "fecha": d, "clase": "a",
                 "tamano_m": round(float(S["Palua"].diff().get(pd.Timestamp(d))), 3), "evidencia": ev})
for d, ev in C_EVENTS.items():
    st = "Ciudad_Bolivar" if ev.startswith("Ciudad_Bolivar") else (
        "Ayacucho" if ev.startswith("Ayacucho") else ("Caicara" if ev.startswith("Caicara") else "Palua"))
    t = pd.Timestamp(d)
    sosp.append({"estacion": st, "fecha": d, "clase": "c",
                 "tamano_m": round(float(S[st].diff().get(t)), 3), "evidencia": ev})
sdf = pd.DataFrame(sosp).sort_values(["clase", "estacion", "fecha"])
sdf.to_csv(os.path.join(common.SALIDA, "eventos_sospechosos.csv"), index=False)

# ---------------- 3. Series constantes ----------------
runs = []
for st, s in S.items():
    idx = list(s.index)
    start = idx[0]; prev = s.iloc[0]; cnt = 1
    for i in range(1, len(s)):
        t, v = idx[i], s.iloc[i]
        if v == prev and (t - idx[i - 1]).days == 1:
            cnt += 1
        else:
            if cnt >= 5:
                runs.append({"estacion": st, "desde": start.date().isoformat(),
                             "hasta": idx[i - 1].date().isoformat(), "dias": cnt,
                             "valor": round(float(prev), 3)})
            start = t; prev = v; cnt = 1
    if cnt >= 5:
        runs.append({"estacion": st, "desde": start.date().isoformat(),
                     "hasta": idx[-1].date().isoformat(), "dias": cnt,
                     "valor": round(float(prev), 3)})
rdf = pd.DataFrame(runs)
# percentil del valor dentro de la estación y bandera de posible relleno
rdf["pctl"] = [round(float((S[r.estacion] <= r.valor).mean() * 100), 1) for r in rdf.itertuples()]
rdf["posible_relleno"] = (rdf["dias"] >= 7) & (rdf["pctl"] >= 75)
rdf = rdf.sort_values(["estacion", "dias"], ascending=[True, False])
rdf.to_csv(os.path.join(common.SALIDA, "constantes.csv"), index=False)

print("=== Saltos > 0.5 m por estación y clase ===")
print(jdf.groupby(["estacion", "clase"]).size().to_string())
print("\nTotal saltos >0.5:", len(jdf))
print("\n=== Eventos (b) a eliminar: días ===")
print(sdf[sdf.clase == "b"][["estacion", "fecha", "tamano_m"]].to_string(index=False))
print("\n=== Eventos (a) y (c) ===")
print(sdf[sdf.clase != "b"].to_string(index=False))
print("\n=== Series constantes >=5 días ===")
print(rdf.groupby("estacion").agg(n_runs=("dias", "size"), max_dias=("dias", "max"),
                                  dias_totales=("dias", "sum")).to_string())
print("\nTop 8 por estación:")
for st in common.STATIONS:
    print(st)
    print(rdf[rdf.estacion == st].head(8).to_string(index=False))
print("\nPosibles rellenos (>=7 días y valor >= p75):", int(rdf["posible_relleno"].sum()))
print(rdf[rdf["posible_relleno"]].to_string(index=False))

# ---------------- 4. Rango físico ----------------
print("\n=== Rangos físicos ===")
for st, s in S.items():
    print(st, "min", s.min(), "max", s.max(), "negativos", int((s <= 0).sum()),
          "p99.9", round(s.quantile(0.999), 2))

# ---------------- 5. Verificación de los cambios de referencia conocidos ---
print("\n=== Cambios de referencia conocidos en Ciudad Bolívar ===")
for d, expected in [("1987-05-15", -2.20), ("2019-01-01", 1.00)]:
    t = pd.Timestamp(d)
    delta = S["Ciudad_Bolivar"].get(t) - S["Ciudad_Bolivar"].get(t - pd.Timedelta(days=1))
    palua_delta = S["Palua"].get(t) - S["Palua"].get(t - pd.Timedelta(days=1))
    print(d, "delta CB=", round(float(delta), 3), "esperado=", expected,
          "delta Palua=", round(float(palua_delta), 3))

# ---------------- 6. Guardar días (b) para limpieza ----------------
b_rows = [(st, d) for st, ds in B_DAYS.items() for d in ds]
pd.DataFrame(b_rows, columns=["estacion", "fecha"]).to_csv(
    os.path.join(common.SALIDA, "dias_b_eliminar.csv"), index=False)

# Resumen JSON
summary = {
    "saltos_mayores_05": int(len(jdf)),
    "por_estacion": jdf.groupby("estacion").size().to_dict(),
    "por_clase": jdf.groupby("clase").size().to_dict(),
    "dias_b": len(b_rows),
    "eventos_a": len(A_EVENTS),
    "eventos_c": len(C_EVENTS),
    "runs_constantes_ge5": {st: int((rdf.estacion == st).sum()) for st in common.STATIONS},
    "posibles_rellenos": int(rdf["posible_relleno"].sum()),
    "rangos_fisicos": {st: [float(S[st].min()), float(S[st].max())] for st in common.STATIONS},
}
with open(os.path.join(common.SALIDA, "qc_resumen.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
