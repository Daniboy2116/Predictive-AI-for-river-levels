from typing import Any, Dict, List, Optional, Union

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtesting import (
    BacktestValidationError, cargar_datos_backtest, clasificar_temporada,
    simular_evento,
)
from src.configuracion import (
    ESTACIONES,
    ESTACIONES_OBJETIVO,
    HORIZONTE_BACKTEST,
    LOOKBACKS_BACKTEST,
    MODO_PARTICION_BACKTEST,
    NOMBRES_ESTACIONES,
    PORCENTAJE_TRAIN_BACKTEST,
    UMBRALES_ALERTA,
    obtener_rango_holdout,
)
from src.inference import REGISTRO_MODELOS

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

COLORES_ESTACIONES = {
    "Ayacucho": "#a78bfa",
    "Caicara": "#38bdf8",
    "Ciudad_Bolivar": "#f59e0b",
    "Palua": "#22c55e",
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
.season-card {
    padding: 1rem 1.2rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.15);
}
.metric-box {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 12px;
    padding: 0.75rem 0.6rem;
    text-align: center;
    min-width: 0;
    overflow: visible;
}
.metric-box h3 {
    margin: 0;
    font-size: 1.35rem;
    font-weight: 700;
    color: #38bdf8;
    white-space: nowrap;
}
.metric-box p {
    margin: 0.25rem 0 0 0;
    color: #94a3b8;
    font-size: 0.82rem;
    line-height: 1.25;
    word-wrap: break-word;
}
</style>
"""


def estilo_grafica(fig: go.Figure, titulo: str, eje_y: str = "Nivel (m)", altura: int = 520) -> None:
    fig.update_layout(
        template="plotly_dark",
        height=altura,
        hovermode="x unified",
        hoverlabel=dict(
            namelength=-1,
            bgcolor="rgba(10, 21, 38, 0.95)",
            bordercolor="rgba(56, 189, 248, 0.4)",
            font=dict(size=12, color="#f8fafc"),
        ),
        title=dict(text=titulo, x=0.01, y=0.97, yanchor="top", font=dict(size=16, color="#f8fafc")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.015)",
        yaxis=dict(title=eje_y, gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(
            orientation="h",
            x=0.01,
            y=1.08,
            yanchor="bottom",
            font=dict(size=12, color="#cbd5e1"),
        ),
        margin=dict(l=20, r=20, t=85, b=20),
    )


@st.cache_data
def obtener_datos_hidrologicos():
    return cargar_datos_backtest()


def construir_grafica_historica(
    niveles: pd.DataFrame,
    observados: pd.DataFrame,
    estacion_sel: str,
    inicio_holdout: pd.Timestamp,
    fin_holdout: pd.Timestamp,
) -> go.Figure:
    """Construye la gráfica de series históricas delimitando con precisión el corte train/holdout."""
    fig = go.Figure()

    if estacion_sel == "Todas las estaciones":
        for est in ESTACIONES:
            col = COLORES_ESTACIONES.get(est, "#38bdf8")
            nombre = NOMBRES_ESTACIONES.get(est, est)
            s_obs = niveles[est].where(observados[est])
            fig.add_trace(go.Scatter(
                x=s_obs.index,
                y=s_obs.to_numpy(),
                name=nombre,
                mode="lines",
                line=dict(color=col, width=1.8),
                hovertemplate="%{y:.2f} m<extra></extra>",
                hoverlabel=dict(namelength=-1),
            ))
        titulo = "Serie Histórica del Nivel del Río Orinoco — Red de Monitoreo Completa"
    else:
        col = COLORES_ESTACIONES.get(estacion_sel, "#38bdf8")
        nombre = NOMBRES_ESTACIONES.get(estacion_sel, estacion_sel)

        # Segmento de entrenamiento (visto por los modelos)
        mascara_train = (niveles.index < inicio_holdout) & observados[estacion_sel]
        s_train = niveles.loc[mascara_train, estacion_sel]
        if not s_train.empty:
            fig.add_trace(go.Scatter(
                x=s_train.index,
                y=s_train.to_numpy(),
                name=f"Entrenamiento: {nombre} (70%)",
                mode="lines",
                line=dict(color=col, width=2.0),
                hovertemplate="%{y:.2f} m<extra></extra>",
                hoverlabel=dict(namelength=-1),
            ))

        # Segmento Holdout (estrictamente no visto durante el entrenamiento)
        mascara_holdout = (niveles.index >= inicio_holdout) & observados[estacion_sel]
        s_holdout = niveles.loc[mascara_holdout, estacion_sel]
        if not s_holdout.empty:
            fig.add_trace(go.Scatter(
                x=s_holdout.index,
                y=s_holdout.to_numpy(),
                name=f"Holdout Ciego: {nombre} (30% reservado)",
                mode="lines",
                line=dict(color="#94a3b8", width=1.8, dash="dot"),
                hovertemplate="%{y:.2f} m<extra></extra>",
                hoverlabel=dict(namelength=-1),
            ))

        if estacion_sel in UMBRALES_ALERTA:
            umbral = UMBRALES_ALERTA[estacion_sel]
            fig.add_hline(
                y=umbral,
                line=dict(color="#ef4444", dash="dash", width=1.8),
                annotation_text=f"Cota de alerta: {umbral:.1f} m",
                annotation_position="top left",
                annotation_font=dict(color="#ef4444"),
            )
        titulo = f"Serie Histórica Observada y Partición — {nombre}"

    # Línea vertical divisoria del Holdout
    fig.add_vline(
        x=inicio_holdout,
        line=dict(color="#f59e0b", width=2, dash="dash"),
        annotation_text="Línea de Partición (Inicio Holdout Ciego)",
        annotation_position="top left",
        annotation_font=dict(color="#f59e0b", size=11),
    )

    # Región sombreada del Holdout
    fig.add_vrect(
        x0=inicio_holdout,
        x1=fin_holdout,
        fillcolor="rgba(245, 158, 11, 0.08)",
        line_width=1,
        line_dash="dot",
        line_color="rgba(245, 158, 11, 0.4)",
        annotation_text="Conjunto Holdout (30% reservado para Backtesting)",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#f59e0b"),
    )

    estilo_grafica(fig, titulo, "Nivel (m)", altura=520)
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)))
    return fig


def construir_grafica_evento(
    historia: pd.DataFrame,
    ground_truth: pd.DataFrame,
    predicciones: pd.DataFrame,
    fecha_origen: pd.Timestamp,
    temporada: dict,
    nombre_est: str,
    nombres_modelos: Any = None,
) -> go.Figure:
    """Construye la visualización interactiva del evento puntual con historia, punto de corte y predicción multimodelo."""
    fig = go.Figure()

    # 1. Historia previa observada
    hist_obs = historia[historia["Observado"]].sort_values("Fecha")
    if not hist_obs.empty:
        fig.add_trace(go.Scatter(
            x=hist_obs["Fecha"],
            y=hist_obs["Nivel"],
            name="Historia observada",
            mode="lines+markers",
            line=dict(color="#94a3b8", width=2),
            marker=dict(size=4),
            hovertemplate="%{y:.2f} m<extra></extra>",
            hoverlabel=dict(namelength=-1),
        ))

    # 2. Ground Truth futuro observado
    gt_obs = ground_truth[ground_truth["Observado"]].sort_values("Fecha")
    if not gt_obs.empty:
        fig.add_trace(go.Scatter(
            x=gt_obs["Fecha"],
            y=gt_obs["Real"],
            name="Nivel Real (Ground Truth)",
            mode="lines+markers",
            line=dict(color="#10b981", width=3),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="%{y:.2f} m<extra></extra>",
            hoverlabel=dict(namelength=-1),
        ))

    # 3. Predicciones de los modelos
    if nombres_modelos is None:
        modelos_lista = sorted(predicciones["Modelo"].unique()) if "Modelo" in predicciones.columns else ["Modelo"]
    elif isinstance(nombres_modelos, str):
        modelos_lista = [nombres_modelos]
    else:
        modelos_lista = list(nombres_modelos)

    es_multimodelo = len(modelos_lista) > 1

    for mod in modelos_lista:
        if "Modelo" in predicciones.columns:
            sub_pred = predicciones[predicciones["Modelo"] == mod].sort_values("Horizonte")
        else:
            sub_pred = predicciones.sort_values("Horizonte")

        if sub_pred.empty:
            continue

        color_mod = COLORES_MODELOS.get(mod, "#38bdf8")

        if not es_multimodelo:
            p1_6 = sub_pred[sub_pred["Horizonte"] <= 6]
            p7_14 = sub_pred[sub_pred["Horizonte"] >= 7]

            if not p1_6.empty:
                fig.add_trace(go.Scatter(
                    x=p1_6["Fecha"],
                    y=p1_6["Prediccion"],
                    name=f"Pronóstico {mod} (h=1..6)",
                    mode="lines+markers",
                    line=dict(color=color_mod, width=2, dash="solid"),
                    marker=dict(size=6),
                    hovertemplate="%{y:.2f} m<extra></extra>",
                    hoverlabel=dict(namelength=-1),
                ))

            if not p7_14.empty:
                x_destacada = pd.concat([p1_6.tail(1), p7_14]) if not p1_6.empty else p7_14
                fig.add_trace(go.Scatter(
                    x=x_destacada["Fecha"],
                    y=x_destacada["Prediccion"],
                    name=f"Pronóstico {mod} (h=7..14 Prioritario)",
                    mode="lines+markers",
                    line=dict(color=color_mod, width=3.5, dash="solid"),
                    marker=dict(size=8, symbol="diamond"),
                    hovertemplate="%{y:.2f} m<extra></extra>",
                    hoverlabel=dict(namelength=-1),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=sub_pred["Fecha"],
                y=sub_pred["Prediccion"],
                name=f"{mod}",
                mode="lines+markers",
                line=dict(color=color_mod, width=2.5),
                marker=dict(size=6),
                hovertemplate="%{y:.2f} m<extra></extra>",
                hoverlabel=dict(namelength=-1),
            ))

    # 4. Línea de origen t0
    t0_dt = pd.Timestamp(fecha_origen)
    fig.add_vline(
        x=t0_dt,
        line=dict(color="#facc15", width=2, dash="dash"),
        annotation_text="Inicio del pronóstico (t₀)",
        annotation_position="top left",
        annotation_font=dict(color="#facc15", size=11),
    )

    # 5. Sombreado para región prioritaria h=7..14
    t7 = t0_dt + pd.Timedelta(days=7)
    t14 = t0_dt + pd.Timedelta(days=14)
    fig.add_vrect(
        x0=t7,
        x1=t14,
        fillcolor="rgba(56, 189, 248, 0.08)",
        line_width=1,
        line_dash="dot",
        line_color="rgba(56, 189, 248, 0.3)",
        annotation_text="Ventana prioritaria h=7..14",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#38bdf8"),
    )

    if nombre_est in UMBRALES_ALERTA:
        umbral = UMBRALES_ALERTA[nombre_est]
        fig.add_hline(
            y=umbral,
            line=dict(color="#ef4444", dash="dash", width=1.5),
            annotation_text=f"Alerta: {umbral:.1f} m",
            annotation_position="bottom right",
            annotation_font=dict(color="#ef4444", size=10),
        )

    if not es_multimodelo:
        titulo = f"Simulación Histórica: {modelos_lista[0]} en {NOMBRES_ESTACIONES.get(nombre_est, nombre_est)} (t₀ = {t0_dt.strftime('%d/%m/%Y')})"
    else:
        titulo = f"Comparativa Multimodelo ({len(modelos_lista)} modelos) en {NOMBRES_ESTACIONES.get(nombre_est, nombre_est)} (t₀ = {t0_dt.strftime('%d/%m/%Y')})"

    estilo_grafica(fig, titulo, "Nivel (m)", altura=500)
    return fig


def construir_mapa_cuenca() -> go.Figure:
    lats = [info["lat"] for info in ESTACIONES_GEO.values()]
    lons = [info["lon"] for info in ESTACIONES_GEO.values()]
    textos = [f"<b>{info['nombre']}</b><br>{info['tipo']}<br>{info['descripcion']}" for info in ESTACIONES_GEO.values()]

    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode="lines",
        line=dict(width=3.5, color="#38bdf8"),
        name="Cauce Principal del Río Orinoco",
        hoverinfo="none",
    ))

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


st.set_page_config(page_title="Nivel del Orinoco - Pronósticos & Backtesting", page_icon="🌊", layout="wide")
st.markdown(ESTILO_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <h1>🌊 Sistema de Predicción de Niveles del Río Orinoco</h1>
    <p>Visualización histórica y laboratorio interactivo de backtesting (1 a 14 días)</p>
</div>
""", unsafe_allow_html=True)

niveles_datos, observados_datos = obtener_datos_hidrologicos()

# Rango del holdout calculado cronológicamente (70% train / 30% holdout ciego)
inicio_holdout_calc, fin_holdout_calc = obtener_rango_holdout(
    niveles_datos, observados_datos, modo=MODO_PARTICION_BACKTEST, porcentaje_train=PORCENTAJE_TRAIN_BACKTEST,
)
ultimo_origen_valido = fin_holdout_calc - pd.Timedelta(days=HORIZONTE_BACKTEST)
inicio_train = niveles_datos.index.min()
fin_train = inicio_holdout_calc - pd.Timedelta(days=1)

opciones_estacion = ["Todas las estaciones"] + ESTACIONES

with st.sidebar:
    st.markdown("### 📍 Visualización Histórica")
    seleccion_estacion = st.selectbox(
        "Estación hidrológica",
        opciones_estacion,
        format_func=lambda e: "Todas las estaciones" if e == "Todas las estaciones" else NOMBRES_ESTACIONES.get(e, e),
    )

    st.markdown("---")
    st.markdown("### ℹ️ Partición del Dataset")
    modo_texto = f"Porcentual ({int(PORCENTAJE_TRAIN_BACKTEST*100)}% / {int((1-PORCENTAJE_TRAIN_BACKTEST)*100)}%)" if MODO_PARTICION_BACKTEST == "porcentaje" else "Fechas Fijas"
    st.markdown(f"**Modo:** `{modo_texto}`")
    st.markdown(f"**Entrenamiento:** `{inicio_train.strftime('%d/%m/%Y')} a {fin_train.strftime('%d/%m/%Y')}`")
    st.markdown(f"**Holdout Ciego:** `{inicio_holdout_calc.strftime('%d/%m/%Y')} a {fin_holdout_calc.strftime('%d/%m/%Y')}`")

    st.markdown("---")
    st.markdown("### 🔒 Garantía Metodológica")
    st.caption("Cero fuga de información (*No Data Leakage*). El conjunto de prueba permanece ciego y se evalúa exclusivamente a petición del operador en el simulador.")

    st.markdown("---")
    if st.button("🔄 Actualizar Datos / Caché", width="stretch"):
        st.cache_data.clear()
        st.rerun()

pestana_historia, pestana_mapa, pestana_backtest = st.tabs([
    "📊 Serie Histórica y Partición",
    "🗺️ Mapa del Río y Estaciones",
    "🧪 Simulador de Backtesting",
])

with pestana_historia:
    st.subheader("Registro Hidrológico Observado y Delimitación de Partición")
    st.markdown(
        "Esta vista presenta los niveles observados en la cuenca del Río Orinoco. "
        "La partición temporal delimita los datos que los modelos conocen durante el entrenamiento y "
        "reserva estrictamente el **30% final (Holdout)** como conjunto ciego para la evaluación interactiva."
    )

    fig_hist = construir_grafica_historica(
        niveles=niveles_datos,
        observados=observados_datos,
        estacion_sel=seleccion_estacion,
        inicio_holdout=inicio_holdout_calc,
        fin_holdout=fin_holdout_calc,
    )
    st.plotly_chart(fig_hist, width="stretch")

    st.markdown("#### 📈 Resumen Estadístico del Período de Entrenamiento")
    if seleccion_estacion == "Todas las estaciones":
        cols = st.columns(len(ESTACIONES))
        for idx, est in enumerate(ESTACIONES):
            s_train = niveles_datos.loc[niveles_datos.index < inicio_holdout_calc, est]
            s_train_obs = s_train[observados_datos.loc[niveles_datos.index < inicio_holdout_calc, est]]
            pct_obs = (len(s_train_obs) / len(s_train) * 100) if len(s_train) > 0 else 0
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-box">
                    <p style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{NOMBRES_ESTACIONES.get(est, est)}</p>
                    <h3 style="color: {COLORES_ESTACIONES.get(est, '#38bdf8')};">{s_train_obs.mean():.2f} m</h3>
                    <p>Media Train (Máx: {s_train_obs.max():.2f} m)</p>
                    <p style="color: #64748b; font-size: 0.75rem;">Cobertura obs: {pct_obs:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        s_train = niveles_datos.loc[niveles_datos.index < inicio_holdout_calc, seleccion_estacion]
        s_train_obs = s_train[observados_datos.loc[niveles_datos.index < inicio_holdout_calc, seleccion_estacion]]
        pct_obs = (len(s_train_obs) / len(s_train) * 100) if len(s_train) > 0 else 0
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
                <h3>{s_train_obs.mean():.2f} m</h3>
                <p>Nivel Medio (Train)</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-box">
                <h3>{s_train_obs.max():.2f} m</h3>
                <p>Nivel Máximo (Train)</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-box">
                <h3>{s_train_obs.min():.2f} m</h3>
                <p>Nivel Mínimo (Train)</p>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-box">
                <h3>{pct_obs:.1f}%</h3>
                <p>Días Observados Reales</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "💡 **¿Por qué no hay predicciones automáticas sobre el período Holdout en esta gráfica?**  \n"
        "Siguiendo las mejores prácticas de ciencia de datos e hidrología operativa, el conjunto de prueba (holdout) "
        "se mantiene como un **entorno ciego inalterado**. Los modelos no ejecutan inferencia masiva continua por defecto. "
        "En su lugar, el operador decide en la pestaña **🧪 Simulador de Backtesting** qué modelo evaluar, qué ventana de memoria (lookback) "
        "asignar y en qué fecha $t_0$ emitir el pronóstico a 14 días para auditar su desempeño real."
    )

with pestana_mapa:
    st.subheader("Geolocalización de las Estaciones en la Cuenca del Río Orinoco")
    fig_mapa = construir_mapa_cuenca()
    st.plotly_chart(fig_mapa, width="stretch")

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

with pestana_backtest:
    st.markdown("### 🧪 Simulador Interactivo de Backtesting Histórico (1 a 14 días)")
    st.markdown(
        "Evalúa el comportamiento de los modelos en cualquier fecha histórica $t_0$ dentro del **conjunto holdout**, "
        "revisando la historia previa observada, el pronóstico continuo a 14 días y el nivel real registrado."
    )

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.info(
            f"📅 **Holdout disponible:** {inicio_holdout_calc.strftime('%d/%m/%Y')} a {fin_holdout_calc.strftime('%d/%m/%Y')}  \n"
            f"📍 **Último origen válido para h=14:** `{ultimo_origen_valido.strftime('%d/%m/%Y')}`"
        )
    with info_col2:
        modo_texto = f"Porcentual ({int(PORCENTAJE_TRAIN_BACKTEST*100)}% Train / {int((1-PORCENTAJE_TRAIN_BACKTEST)*100)}% Holdout)" if MODO_PARTICION_BACKTEST == "porcentaje" else "Fechas Fijas del Proyecto"
        st.info(
            f"⚙️ **Modo de partición:** `{modo_texto}`  \n"
            f"🔒 **Garantía sin fuga:** Entrenamiento restringido estrictamente a $t < t_\\text{{holdout}}$."
        )

    with st.container():
        st.markdown("#### ⚙️ Parámetros de Simulación")

        modelos_backtest_disponibles = list(REGISTRO_MODELOS.keys())
        modelos_default_inicio = ["Ensamble", "RandomForest", "LightGBM", "XGBoost", "Ridge", "Persistencia"]

        # Callbacks para botones de selección rápida (se ejecutan antes de instanciar los checkboxes)
        def _marcar_todos():
            for m in modelos_backtest_disponibles:
                st.session_state[f"cb_mod_{m}"] = True

        def _marcar_defecto():
            for m in modelos_backtest_disponibles:
                st.session_state[f"cb_mod_{m}"] = (m in modelos_default_inicio)

        def _desmarcar_todos():
            for m in modelos_backtest_disponibles:
                st.session_state[f"cb_mod_{m}"] = False

        # Inicialización de estado para las casillas de cada modelo
        for m in modelos_backtest_disponibles:
            if f"cb_mod_{m}" not in st.session_state:
                st.session_state[f"cb_mod_{m}"] = (m in modelos_default_inicio)

        # 1. Encabezado de casillas y botones de acción rápida
        c_head, c_acc = st.columns([2.0, 2.0])
        with c_head:
            st.markdown("<p style='margin:0; font-size:0.95rem; font-weight:700; color:#38bdf8;'>🔘 Casillas de Selección de Modelos</p>", unsafe_allow_html=True)
            st.caption("Marca o desmarca directamente las casillas para agregar o quitar modelos del pronóstico.")
        with c_acc:
            b1, b2, b3 = st.columns(3)
            b1.button("✅ Todos", on_click=_marcar_todos, width="stretch", key="btn_btest_all")
            b2.button("⭐ Top Modelos", on_click=_marcar_defecto, width="stretch", key="btn_btest_def")
            b3.button("🗑️ Desmarcar todo", on_click=_desmarcar_todos, width="stretch", key="btn_btest_clr")

        # 2. Casillas organizadas por familias temáticas
        grupos_modelos = {
            "🏛️ Líneas Base & Lineal": ["Persistencia", "Climatologia", "AnomaliaAmortiguada", "Ridge"],
            "🌲 Ensambles de Árboles": ["RandomForest", "XGBoost", "LightGBM"],
            "📈 Estadísticos & Aditivos": ["SARIMAX", "AutoARIMA", "Prophet"],
            "🧠 Deep Learning & FM": ["TimesFM", "LSTM", "CNN1D", "Transformer", "Ensamble"],
        }

        cols_grupos = st.columns(4)
        for idx_g, (grupo_nombre, mods_grupo) in enumerate(grupos_modelos.items()):
            with cols_grupos[idx_g]:
                st.markdown(f"<p style='font-weight:600; font-size:0.84rem; color:#94a3b8; margin:4px 0 2px 0;'>{grupo_nombre}</p>", unsafe_allow_html=True)
                for m in mods_grupo:
                    if m in modelos_backtest_disponibles:
                        st.checkbox(m, key=f"cb_mod_{m}")

        # 3. Modelos seleccionados actuales
        modelos_sel = [m for m in modelos_backtest_disponibles if st.session_state.get(f"cb_mod_{m}", False)]
        n_sel = len(modelos_sel)

        if not modelos_sel:
            st.warning("⚠️ No hay ningún modelo seleccionado. Marca una o más casillas arriba para habilitar la simulación.")
        else:
            st.caption(f"✓ **{n_sel} modelo{'s' if n_sel != 1 else ''} activo{'s' if n_sel != 1 else ''} para la simulación:** {', '.join(modelos_sel)}")

        st.markdown("<hr style='margin: 12px 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        # 4. Parámetros complementarios: Estación, Fecha de corte t0, Lookback
        ctrl2, ctrl3, ctrl4 = st.columns([1.6, 1.2, 1.2])

        with ctrl2:
            opciones_est_btest = ["Todas las estaciones"] + ESTACIONES
            estacion_btest = st.selectbox(
                "Estación a evaluar",
                opciones_est_btest,
                index=opciones_est_btest.index("Ciudad_Bolivar"),
                format_func=lambda e: "Todas las estaciones" if e == "Todas las estaciones" else NOMBRES_ESTACIONES.get(e, e),
                key="sb_estacion_backtest",
            )

        with ctrl3:
            fecha_def = inicio_holdout_calc + (ultimo_origen_valido - inicio_holdout_calc) / 2
            fecha_origen_sel = st.date_input(
                "Fecha de origen (t₀)",
                value=fecha_def.date(),
                min_value=inicio_holdout_calc.date(),
                max_value=ultimo_origen_valido.date(),
                key="di_fecha_origen",
            )

        with ctrl4:
            modelos_con_60 = [m for m in modelos_sel if len(REGISTRO_MODELOS[m]["lookbacks_soportados"]) == 1]
            if modelos_con_60:
                lookback_sel = st.select_slider(
                    "Ventana histórica (lookback)",
                    options=LOOKBACKS_BACKTEST,
                    value=60,
                    disabled=True,
                    key="ss_lookback",
                )
                st.caption(f"ℹ️ {', '.join(modelos_con_60)} fija lookback a 60 d.")
            else:
                lookback_sel = st.select_slider(
                    "Ventana histórica (lookback)",
                    options=LOOKBACKS_BACKTEST,
                    value=60,
                    key="ss_lookback",
                )

        btn_texto = "🚀 Ejecutar simulación" if len(modelos_sel) <= 1 else f"🚀 Ejecutar simulación multimodelo ({len(modelos_sel)} modelos)"
        ejecutar_sim = st.button(btn_texto, type="primary", width="stretch", disabled=(len(modelos_sel) == 0))

    t0_timestamp = pd.Timestamp(fecha_origen_sel)
    info_temp = clasificar_temporada(t0_timestamp)

    st.markdown(f"""
    <div class="season-card" style="background: rgba({int(info_temp['color'][1:3], 16)}, {int(info_temp['color'][3:5], 16)}, {int(info_temp['color'][5:7], 16)}, 0.12); border-left: 6px solid {info_temp['color']};">
        <h4 style="margin: 0; color: {info_temp['color']};">📅 Fecha origen: {t0_timestamp.strftime('%d/%m/%Y')} | Temporada: {info_temp['nombre']}</h4>
        <p style="margin: 0.3rem 0 0 0; color: #e2e8f0; font-size: 0.9rem;">
            <b>Descripción:</b> {info_temp['descripcion']} | <b>Mes:</b> {info_temp['mes']} | <b>Día del año:</b> {info_temp['dia_del_anio']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    modelos_str_clave = "_".join(sorted(modelos_sel))
    clave_estado = f"sim_{modelos_str_clave}_{estacion_btest}_{t0_timestamp.strftime('%Y%m%d')}_{lookback_sel}"

    if ejecutar_sim:
        if not modelos_sel:
            st.warning("⚠️ Selecciona al menos un modelo para realizar la simulación.")
        else:
            n_mods = len(modelos_sel)
            with st.spinner(f"Entrenando/cargando {n_mods} modelo{'s' if n_mods > 1 else ''} ({', '.join(modelos_sel)}) y simulando evento a 14 días..."):
                try:
                    estaciones_a_simular = ESTACIONES if estacion_btest == "Todas las estaciones" else [estacion_btest]
                    res_evento = simular_evento(
                        nombre_modelo=modelos_sel,
                        niveles=niveles_datos,
                        observados=observados_datos,
                        fecha_origen=t0_timestamp,
                        lookback=lookback_sel,
                        estaciones=estaciones_a_simular,
                        horizonte_maximo=HORIZONTE_BACKTEST,
                        modo_split=MODO_PARTICION_BACKTEST,
                        inicio_holdout=inicio_holdout_calc,
                    )
                    st.session_state[clave_estado] = res_evento
                    st.success(f"✅ Simulación completada exitosamente para {n_mods} modelo{'s' if n_mods > 1 else ''}.")
                except BacktestValidationError as e:
                    st.error(f"❌ Error de validación: {e}")
                except Exception as e:
                    st.error(f"❌ Error durante la simulación: {e}")

    if clave_estado in st.session_state:
        simulacion = st.session_state[clave_estado]
        df_preds = simulacion["predicciones"]
        df_hist = simulacion["historia"]
        df_gt = simulacion["ground_truth"]
        df_mets = simulacion["metricas"]
        modelos_evaluados = simulacion.get("modelos", [simulacion.get("modelo")])

        st.markdown("---")
        mods_titulo = ", ".join(modelos_evaluados) if len(modelos_evaluados) > 1 else modelos_evaluados[0]
        st.markdown(f"### 📊 Resultados del Evento: {mods_titulo} desde {t0_timestamp.strftime('%d/%m/%Y')}")

        estaciones_simuladas = df_preds["Estacion"].unique()

        for est_sim in estaciones_simuladas:
            nombre_legible = NOMBRES_ESTACIONES.get(est_sim, est_sim)
            sub_preds = df_preds[df_preds["Estacion"] == est_sim]
            sub_hist = df_hist[df_hist["Estacion"] == est_sim]
            sub_gt = df_gt[df_gt["Estacion"] == est_sim]
            sub_mets = df_mets[df_mets["Estacion"] == est_sim]

            st.markdown(f"#### 📍 Estación: {nombre_legible}")

            modelos_en_est = sub_preds["Modelo"].unique()

            # Caso 1 modelo: tarjetas clásicas de resumen
            if len(modelos_en_est) == 1:
                sub_met = sub_mets.iloc[0]
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                with mc1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{sub_met['MAE']:.3f} m</h3>
                        <p>MAE Global (1..14)</p>
                    </div>
                    """, unsafe_allow_html=True)
                with mc2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{sub_met['RMSE']:.3f} m</h3>
                        <p>RMSE Global (1..14)</p>
                    </div>
                    """, unsafe_allow_html=True)
                with mc3:
                    r2_str = f"{sub_met['R2']:.2f}" if not pd.isna(sub_met['R2']) else "N/A"
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{r2_str}</h3>
                        <p>Coeficiente R²</p>
                    </div>
                    """, unsafe_allow_html=True)
                with mc4:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{sub_met['MAE_1_6']:.3f} m</h3>
                        <p>MAE Corto (h=1..6)</p>
                    </div>
                    """, unsafe_allow_html=True)
                with mc5:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>{sub_met['MAE_7_14']:.3f} m</h3>
                        <p>MAE Prioritario (h=7..14)</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Caso multimodelo: podio y tabla de ranking comparativo
                mets_ordenadas = sub_mets.sort_values("MAE").reset_index(drop=True)
                mejor_mod = mets_ordenadas.iloc[0]
                color_ganador = COLORES_MODELOS.get(mejor_mod["Modelo"], "#38bdf8")

                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.08); border-left: 5px solid {color_ganador}; padding: 0.8rem 1.2rem; border-radius: 10px; margin-bottom: 1rem;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">🏆 Mejor Desempeño: <span style="color: {color_ganador};">{mejor_mod['Modelo']}</span></span>
                    <span style="color: #cbd5e1; margin-left: 1.5rem;"><b>MAE Global:</b> {mejor_mod['MAE']:.3f} m | <b>MAE Prioritario (h=7..14):</b> {mejor_mod['MAE_7_14']:.3f} m | <b>RMSE:</b> {mejor_mod['RMSE']:.3f} m</span>
                </div>
                """, unsafe_allow_html=True)

                tabla_ranking = mets_ordenadas[["Modelo", "MAE", "RMSE", "R2", "MAE_1_6", "MAE_7_14"]].copy()
                tabla_ranking["Brecha vs Mejor"] = (tabla_ranking["MAE"] - mejor_mod["MAE"]).apply(lambda v: f"+{v*100:.1f} cm" if v > 0.0001 else "Líder 🥇")

                st.dataframe(
                    tabla_ranking.style.format({
                        "MAE": "{:.3f} m",
                        "RMSE": "{:.3f} m",
                        "R2": "{:.2f}",
                        "MAE_1_6": "{:.3f} m",
                        "MAE_7_14": "{:.3f} m",
                    }, na_rep="N/A"),
                    width="stretch",
                )

            # GRÁFICA MULTIMODELO (TODOS LOS MODELOS EN UNA SOLA GRÁFICA)
            fig_evento = construir_grafica_evento(
                historia=sub_hist,
                ground_truth=sub_gt,
                predicciones=sub_preds,
                fecha_origen=t0_timestamp,
                temporada=info_temp,
                nombre_est=est_sim,
                nombres_modelos=modelos_en_est,
            )
            st.plotly_chart(fig_evento, width="stretch")

            # TABLA COMPARATIVA DIARIA
            st.markdown("##### 📋 Comparativa Diaria: Nivel Real vs Modelos Seleccionados")
            fechas_u = sub_preds.drop_duplicates("Horizonte").sort_values("Horizonte")
            df_comp_diaria = pd.DataFrame({
                "Horizonte": fechas_u["Horizonte"].values,
                "Fecha": pd.to_datetime(fechas_u["Fecha"]).dt.strftime("%d/%m/%Y").values,
                "Nivel Real (m)": fechas_u["Real"].values,
            })
            for m_nom in modelos_en_est:
                sub_m = sub_preds[sub_preds["Modelo"] == m_nom].sort_values("Horizonte")
                df_comp_diaria[f"Pred {m_nom} (m)"] = sub_m["Prediccion"].values

            formatos_cols = {"Nivel Real (m)": "{:.3f}"}
            for m_nom in modelos_en_est:
                formatos_cols[f"Pred {m_nom} (m)"] = "{:.3f}"

            st.dataframe(
                df_comp_diaria.style.format(formatos_cols, na_rep="— (No Observado)")
                .apply(lambda row: ["background-color: rgba(56, 189, 248, 0.10)" if row["Horizonte"] >= 7 else "" for _ in row], axis=1),
                width="stretch",
            )
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("👆 Selecciona los parámetros y pulsa **Ejecutar simulación** para iniciar el análisis histórico del evento.")
