import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.configuracion import (
    ESTACIONES,
    ESTACIONES_OBJETIVO,
    HORIZONTE_MAXIMO,
    HORIZONTE_PRINCIPAL,
    NOMBRES_ESTACIONES,
    UMBRALES_ALERTA,
)
from src.evaluate import RUTA_PREDICCIONES, cargar_predicciones

COLORES_MODELOS = {
    "Persistencia": "#94a3b8",
    "Climatologia": "#64748b",
    "AnomaliaAmortiguada": "#cbd5e1",
    "Ridge": "#38bdf8",
    "RandomForest": "#22c55e",
    "XGBoost": "#10b981",
    "LightGBM": "#a3e635",
    "SARIMAX": "#f59e0b",
    "AutoARIMA": "#d97706",
    "TimesFM": "#ef4444",
    "Prophet": "#fb923c",
    "LSTM": "#a78bfa",
    "CNN1D": "#f472b6",
    "Transformer": "#60a5fa",
    "Ensamble": "#facc15",
}

ESTACIONES_GEO = {
    "Ayacucho": {
        "nombre": "Puerto Ayacucho",
        "lat": 5.6606,
        "lon": -67.6256,
        "tipo": "Entrada (Aguas arriba)",
        "descripcion": "Estación cabecera de la cuenca media. Tiempo de viaje a Caicara: ~14 días.",
    },
    "Caicara": {
        "nombre": "Caicara del Orinoco",
        "lat": 7.6228,
        "lon": -66.1628,
        "tipo": "Intermedia (Aguas arriba)",
        "descripcion": "Punto de confluencia con el río Apure. Tiempo de viaje a Ciudad Bolívar: ~4 días.",
    },
    "Ciudad_Bolivar": {
        "nombre": "Ciudad Bolívar",
        "lat": 8.1406,
        "lon": -63.5481,
        "tipo": "Objetivo de Pronóstico",
        "descripcion": "Estación hidrométrica histórica (Angostura). Umbral de alerta por crecida: 16.50 m.",
    },
    "Palua": {
        "nombre": "Palúa (San Félix)",
        "lat": 8.3567,
        "lon": -62.6844,
        "tipo": "Objetivo de Pronóstico",
        "descripcion": "Confluencia del Río Caroní con el Orinoco en Ciudad Guayana.",
    },
}

MODELOS_DEFAULT = ["Persistencia", "Ridge", "LightGBM", "TimesFM", "Ensamble"]

ESTILO_CSS = """
<style>
.stApp {
    background: radial-gradient(1100px 550px at 10% -5%, #0c2d4a 0%, transparent 60%),
                linear-gradient(160deg, #060b17 0%, #0a1526 50%, #060f1d 100%);
    color: #f1f5f9;
}
section[data-testid="stSidebar"] {
    background: rgba(10, 21, 38, 0.95);
    border-right: 1px solid rgba(56, 189, 248, 0.18);
}
.header-container {
    padding: 1.3rem 1.6rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(99, 102, 241, 0.06) 100%);
    border: 1px solid rgba(56, 189, 248, 0.22);
    margin-bottom: 1.5rem;
}
.header-container h1 {
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-container p {
    color: #94a3b8;
    margin: 0.3rem 0 0 0;
    font-size: 0.95rem;
}
.geo-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.geo-card h4 {
    margin: 0 0 0.3rem 0;
    color: #38bdf8;
    font-size: 1.05rem;
}
.geo-card p {
    margin: 0;
    color: #cbd5e1;
    font-size: 0.88rem;
}
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    margin-bottom: 0.4rem;
}
</style>
"""


def estilo_grafica(fig: go.Figure, titulo: str, eje_y: str = "Nivel (m)", altura: int = 520) -> None:
    fig.update_layout(
        template="plotly_dark",
        height=altura,
        hovermode="x unified",
        title=dict(text=titulo, x=0.01, y=0.98, yanchor="top", font=dict(size=16, color="#f8fafc")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.015)",
        yaxis=dict(title=eje_y, gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"),
        margin=dict(l=10, r=10, t=75, b=10),
    )


@st.cache_data
def cargar_datos_predicciones():
    if not RUTA_PREDICCIONES.exists():
        return pd.DataFrame()
    return cargar_predicciones("prueba")


def construir_grafica_estacion(datos_estacion: pd.DataFrame, nombre_est: str, modelos_sel: list, horizonte_sel: int) -> go.Figure:
    fig = go.Figure()

    reales = datos_estacion.drop_duplicates("Fecha").set_index("Fecha").sort_index()
    reales_obs = reales[reales["Observado"]]["Real"]

    if not reales_obs.empty:
        fig.add_trace(go.Scatter(
            x=reales_obs.index,
            y=reales_obs.to_numpy(),
            name="Nivel Real Observado",
            mode="lines",
            line=dict(color="#f8fafc", width=2.5),
        ))

    for mod in modelos_sel:
        sub = datos_estacion[datos_estacion["Modelo"] == mod].sort_values("Fecha")
        if not sub.empty:
            color = COLORES_MODELOS.get(mod, "#38bdf8")
            fig.add_trace(go.Scatter(
                x=sub["Fecha"],
                y=sub["Prediccion"],
                name=mod,
                mode="lines",
                opacity=0.9,
                line=dict(color=color, width=1.6, dash="solid"),
            ))

    if nombre_est in UMBRALES_ALERTA:
        umbral = UMBRALES_ALERTA[nombre_est]
        fig.add_hline(
            y=umbral,
            line=dict(color="#ef4444", dash="dash", width=1.8),
            annotation_text=f"Cota de alerta: {umbral:.1f} m",
            annotation_position="top left",
            annotation_font=dict(color="#ef4444"),
        )

    titulo = f"Nivel Real vs Pronósticos a {horizonte_sel} días — {NOMBRES_ESTACIONES.get(nombre_est, nombre_est)}"
    estilo_grafica(fig, titulo, "Nivel (m)")
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)))
    return fig


def construir_mapa_cuenca() -> go.Figure:
    lats = [info["lat"] for info in ESTACIONES_GEO.values()]
    lons = [info["lon"] for info in ESTACIONES_GEO.values()]
    textos = [f"<b>{info['nombre']}</b><br>{info['tipo']}<br>{info['descripcion']}" for info in ESTACIONES_GEO.values()]

    fig = go.Figure()

    # Trazo de la línea fluvial aproximada del Río Orinoco conectando estaciones
    fig.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode="lines",
        line=dict(width=3.5, color="#38bdf8"),
        name="Cauce Principal del Río Orinoco",
        hoverinfo="none",
    ))

    # Puntos de las estaciones
    colores_marcadores = ["#a78bfa", "#38bdf8", "#f59e0b", "#22c55e"]
    simbolos = ["circle", "circle", "diamond", "diamond"]
    tamanos = [14, 14, 18, 18]

    fig.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode="markers+text",
        text=[info["nombre"] for info in ESTACIONES_GEO.values()],
        textposition="top right",
        textfont=dict(size=12, color="#f8fafc"),
        hovertext=textos,
        hoverinfo="text",
        marker=dict(
            size=tamanos,
            color=colores_marcadores,
            symbol=simbolos,
            line=dict(width=2, color="#ffffff"),
        ),
        name="Estaciones de Monitoreo",
    ))

    fig.update_layout(
        template="plotly_dark",
        height=540,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            scope="south america",
            center=dict(lat=7.0, lon=-65.0),
            projection_scale=6.5,
            showland=True,
            landcolor="#0b172a",
            showocean=True,
            oceancolor="#070e1b",
            showrivers=True,
            rivercolor="#0ea5e9",
            showlakes=True,
            lakecolor="#070e1b",
            showcountries=True,
            countrycolor="#334155",
            countrywidth=1,
            subunitcolor="#1e293b",
        ),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"),
    )
    return fig


st.set_page_config(page_title="Nivel del Orinoco - Pronósticos", page_icon="🌊", layout="wide")
st.markdown(ESTILO_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <h1>🌊 Sistema de Predicción de Niveles del Río Orinoco</h1>
    <p>Comparativa directa entre mediciones hidrométricas reales y modelos predictivos de Inteligencia Artificial</p>
</div>
""", unsafe_allow_html=True)

predicciones = cargar_datos_predicciones()

if predicciones.empty:
    st.warning("⚠️ No se encontraron predicciones guardadas en `outputs/predictions.csv`.")
    st.info("Para generar los pronósticos y habilitar la visualización, ejecuta el pipeline en tu terminal con: `python main.py`")
    st.stop()

estaciones_disponibles = sorted(predicciones["Estacion"].unique())
opciones_estacion = ["Todas las estaciones"] + estaciones_disponibles

with st.sidebar:
    st.markdown("### 📍 Selección de Estación")
    seleccion_estacion = st.selectbox(
        "Estación hidrológica",
        opciones_estacion,
        format_func=lambda e: "Todas las estaciones" if e == "Todas las estaciones" else NOMBRES_ESTACIONES.get(e, e),
    )

    st.markdown("---")
    st.markdown("### ⏱️ Horizonte de Pronóstico")
    horizontes_disponibles = sorted(predicciones["Horizonte"].unique())
    horizonte_default = HORIZONTE_PRINCIPAL if HORIZONTE_PRINCIPAL in horizontes_disponibles else horizontes_disponibles[-1]
    horizonte_seleccionado = st.select_slider(
        "Días de anticipación (h)",
        options=horizontes_disponibles,
        value=horizonte_default,
    )

    st.markdown("---")
    st.markdown("### 🤖 Modelos Predictivos")
    if seleccion_estacion == "Todas las estaciones":
        modelos_disponibles = sorted(predicciones["Modelo"].unique())
    else:
        modelos_disponibles = sorted(predicciones.loc[predicciones["Estacion"] == seleccion_estacion, "Modelo"].unique())

    col_btn1, col_btn2 = st.columns(2)
    marcar_todos = col_btn1.button("Todos", use_container_width=True)
    marcar_def = col_btn2.button("Por defecto", use_container_width=True)

    if marcar_todos:
        modelos_seleccionados = modelos_disponibles
    elif marcar_def:
        modelos_seleccionados = [m for m in modelos_disponibles if m in MODELOS_DEFAULT]
    else:
        modelos_seleccionados = []
        for m in modelos_disponibles:
            activo = st.checkbox(m, value=(m in MODELOS_DEFAULT), key=f"check_{m}")
            if activo:
                modelos_seleccionados.append(m)

    st.markdown("---")
    if st.button("🔄 Actualizar Interfaz", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

pestana_grafica, pestana_mapa = st.tabs([
    "📈 Nivel Real vs Predicción de Modelos",
    "🗺️ Mapa del Río y Estaciones",
])

with pestana_grafica:
    if not modelos_seleccionados:
        st.info("💡 Por favor, selecciona al menos un modelo en la barra lateral para comparar sus predicciones con el nivel real.")
    else:
        sub_predicciones = predicciones[predicciones["Horizonte"] == horizonte_seleccionado]

        if seleccion_estacion == "Todas las estaciones":
            for est in estaciones_disponibles:
                datos_est = sub_predicciones[sub_predicciones["Estacion"] == est]
                if not datos_est.empty:
                    fig_est = construir_grafica_estacion(datos_est, est, modelos_seleccionados, horizonte_seleccionado)
                    st.plotly_chart(fig_est, use_container_width=True)
        else:
            datos_est = sub_predicciones[sub_predicciones["Estacion"] == seleccion_estacion]
            if not datos_est.empty:
                fig_est = construir_grafica_estacion(datos_est, seleccion_estacion, modelos_seleccionados, horizonte_seleccionado)
                st.plotly_chart(fig_est, use_container_width=True)
            else:
                st.warning(f"No hay registros disponibles para {NOMBRES_ESTACIONES.get(seleccion_estacion, seleccion_estacion)}.")

with pestana_mapa:
    st.subheader("Geolocalización de las Estaciones en la Cuenca del Río Orinoco")
    fig_mapa = construir_mapa_cuenca()
    st.plotly_chart(fig_mapa, use_container_width=True)

    st.markdown("### 📍 Detalle de las Estaciones Hidrométricas")
    c1, c2 = st.columns(2)
    estaciones_lista = list(ESTACIONES_GEO.items())

    for idx, (clave, info) in enumerate(estaciones_lista):
        columna = c1 if idx % 2 == 0 else c2
        with columna:
            columna.markdown(f"""
            <div class="geo-card">
                <span class="badge">{info['tipo']}</span>
                <h4>{info['nombre']}</h4>
                <p><b>Coordenadas:</b> Lat: {info['lat']:.4f}° N, Lon: {info['lon']:.4f}° W</p>
                <p>{info['descripcion']}</p>
            </div>
            """, unsafe_allow_html=True)
