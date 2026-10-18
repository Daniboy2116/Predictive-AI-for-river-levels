"""Dashboard Streamlit: comparación de modelos de predicción de niveles de ríos.

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evaluate import load_metrics, load_predictions

STATION_LABELS = {
    "Palua": "Palúa", "Orinoco": "Orinoco", "Caicara": "Caicara",
    "Ciudad_Bolivar": "Ciudad Bolívar", "Ayacucho": "Ayacucho",
}
MODEL_COLORS = {
    "ARIMA": "#f59e0b", "RandomForest": "#22c55e", "XGBoost": "#10b981",
    "LSTM": "#a78bfa", "CNN1D": "#f472b6", "Transformer": "#60a5fa",
    "TimesFM": "#facc15", "Prophet": "#fb923c",
}

st.set_page_config(page_title="Predicción de Niveles de Ríos", page_icon="🌊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #0e3a5c 0%, transparent 60%),
         linear-gradient(160deg, #050b18 0%, #0a1628 60%, #06101f 100%); color: #e2e8f0; }
section[data-testid="stSidebar"] { background: rgba(10, 22, 40, .85); backdrop-filter: blur(12px);
         border-right: 1px solid rgba(56, 189, 248, .15); }
.hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0;
         background: linear-gradient(90deg, #38bdf8, #818cf8, #22d3ee); -webkit-background-clip: text;
         -webkit-text-fill-color: transparent; }
.hero p { color: #94a3b8; margin-top: .2rem; }
.card { background: rgba(255,255,255,.04); border: 1px solid rgba(56,189,248,.18);
        border-radius: 16px; padding: 1rem 1.2rem; backdrop-filter: blur(10px);
        transition: transform .25s ease, box-shadow .25s ease; }
.card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(56,189,248,.15); }
.card .lbl { color: #94a3b8; font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }
.card .val { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; }
.card .sub { color: #38bdf8; font-size: .85rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga predicciones y métricas exportadas por main.py."""
    return load_predictions(), load_metrics()


def card(col, label: str, value: str, sub: str) -> None:
    """Renderiza una tarjeta KPI."""
    col.markdown(f'<div class="card"><div class="lbl">{label}</div>'
                 f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


preds, metrics = get_data()
st.markdown('<div class="hero"><h1>🌊 Predicción de Niveles de Ríos</h1>'
            '<p>Comparación de modelos estadísticos, ensamblados y redes profundas '
            '· predicción a 1 día · conjunto de prueba</p></div>', unsafe_allow_html=True)

if preds.empty:
    st.warning("No hay resultados. Ejecuta primero: `python main.py`")
    st.stop()

# ----------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ Configuración")
    stations = [s for s in STATION_LABELS if s in preds["Estacion"].unique()]
    station = st.selectbox("Río / estación", stations, format_func=STATION_LABELS.get,
                           key="station_select")
    st.subheader("Modelos")
    available = sorted(preds.loc[preds["Estacion"] == station, "Modelo"].unique())
    selected = [m for m in available if st.checkbox(m, value=True, key=f"chk_{m}")]
    if st.button("🔄 Recargar resultados", key="reload_btn"):
        st.cache_data.clear()
        st.rerun()

sp = preds[preds["Estacion"] == station]
sm = metrics[(metrics["Estacion"] == station) & (metrics["Modelo"].isin(selected))]
sm = sm.sort_values(["RMSE", "MAE"]).reset_index(drop=True)
real = sp.drop_duplicates("Fecha").set_index("Fecha")["Real"].sort_index()

# ----------------------------------------------------------------- KPIs
c1, c2, c3, c4 = st.columns(4)
card(c1, "Nivel actual", f"{real.iloc[-1]:.2f} m", f"{real.index[-1]:%d %b %Y}")
card(c2, "Rango en test", f"{real.min():.1f} – {real.max():.1f} m", "mín – máx")
if not sm.empty:
    best = sm.iloc[0]
    card(c3, "Mejor modelo", best["Modelo"], f"RMSE {best['RMSE']:.3f} m")
    card(c4, "R² del mejor", f"{best['R2']:.4f}", f"MAE {best['MAE']:.3f} m")
st.write("")

# ----------------------------------------------------------------- Gráfica
fig = go.Figure()
fig.add_trace(go.Scatter(x=real.index, y=real.values, name="Nivel real",
                         line=dict(color="#e2e8f0", width=3)))
for model in selected:
    s = sp[sp["Modelo"] == model].sort_values("Fecha")
    fig.add_trace(go.Scatter(x=s["Fecha"], y=s["Prediccion"], name=model, opacity=.9,
                             line=dict(color=MODEL_COLORS.get(model), width=1.6, dash="dot")))
fig.update_layout(
    template="plotly_dark", height=520, hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
    title=f"Nivel real vs. predicciones — {STATION_LABELS[station]}",
    yaxis_title="Nivel (m)", legend=dict(orientation="h", y=1.08, x=0),
    margin=dict(l=10, r=10, t=80, b=10),
    xaxis=dict(rangeslider=dict(visible=True, thickness=.06)),
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------- Tabla de métricas
st.subheader("📊 Métricas en test (ordenadas de mejor a peor por RMSE)")
if sm.empty:
    st.info("Selecciona al menos un modelo en la barra lateral.")
else:
    table = sm[["Modelo", "RMSE", "MAE", "R2"]].rename(columns={"R2": "R²"})
    table.index = range(1, len(table) + 1)
    st.dataframe(
        table.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}"})
        .highlight_min(subset=["RMSE", "MAE"], color="#14532d")
        .highlight_max(subset=["R²"], color="#14532d"),
        use_container_width=True,
    )
