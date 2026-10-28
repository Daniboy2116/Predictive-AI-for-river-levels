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
    HORIZONTE_MAXIMO,
    HORIZONTE_PRINCIPAL,
    LOOKBACKS_BACKTEST,
    MODO_PARTICION_BACKTEST,
    NOMBRES_ESTACIONES,
    PORCENTAJE_TRAIN_BACKTEST,
    UMBRALES_ALERTA,
    obtener_rango_holdout,
)
from src.evaluate import RUTA_PREDICCIONES, cargar_predicciones
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

MODELOS_DEFAULT = ["Persistencia", "Ridge", "LightGBM", "AutoARIMA", "Prophet", "Ensamble"]

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
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-box h3 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-box p {
    margin: 0.2rem 0 0 0;
    color: #94a3b8;
    font-size: 0.85rem;
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


@st.cache_data
def obtener_datos_hidrologicos():
    return cargar_datos_backtest()


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


def construir_grafica_evento(
    historia: pd.DataFrame,
    ground_truth: pd.DataFrame,
    predicciones: pd.DataFrame,
    fecha_origen: pd.Timestamp,
    temporada: dict,
    nombre_est: str,
    nombre_modelo: str,
) -> go.Figure:
    """Construye la visualización interactiva del evento puntual con historia, punto de corte y predicción."""
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
        ))

    # 2. Ground Truth futuro observado
    gt_obs = ground_truth[ground_truth["Observado"]].sort_values("Fecha")
    if not gt_obs.empty:
        fig.add_trace(go.Scatter(
            x=gt_obs["Fecha"],
            y=gt_obs["Real"],
            name="Nivel Real (Ground Truth)",
            mode="lines+markers",
            line=dict(color="#10b981", width=2.5),
            marker=dict(size=6, symbol="circle"),
        ))

    # 3. Predicciones del modelo
    color_mod = COLORES_MODELOS.get(nombre_modelo, "#38bdf8")
    pred_ord = predicciones.sort_values("Horizonte")
    p1_6 = pred_ord[pred_ord["Horizonte"] <= 6]
    p7_14 = pred_ord[pred_ord["Horizonte"] >= 7]

    if not p1_6.empty:
        fig.add_trace(go.Scatter(
            x=p1_6["Fecha"],
            y=p1_6["Prediccion"],
            name=f"Pronóstico {nombre_modelo} (h=1..6)",
            mode="lines+markers",
            line=dict(color=color_mod, width=2, dash="solid"),
            marker=dict(size=6),
        ))

    if not p7_14.empty:
        # Enlace suave desde h=6
        x_destacada = pd.concat([p1_6.tail(1), p7_14]) if not p1_6.empty else p7_14
        fig.add_trace(go.Scatter(
            x=x_destacada["Fecha"],
            y=x_destacada["Prediccion"],
            name=f"Pronóstico {nombre_modelo} (h=7..14 Prioritario)",
            mode="lines+markers",
            line=dict(color=color_mod, width=3.5, dash="solid"),
            marker=dict(size=8, symbol="diamond"),
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
        annotation_text="Inicio de evaluación prioritaria: h=7",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#38bdf8"),
    )

    titulo = f"Simulación Histórica: {nombre_modelo} en {NOMBRES_ESTACIONES.get(nombre_est, nombre_est)} (t₀ = {t0_dt.strftime('%d/%m/%Y')})"
    estilo_grafica(fig, titulo, "Nivel (m)", altura=480)
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
    <p>Comparativa global y simulación de backtesting histórico interactivo (1 a 14 días)</p>
</div>
""", unsafe_allow_html=True)

predicciones = cargar_datos_predicciones()
niveles_datos, observados_datos = obtener_datos_hidrologicos()

# Rango del holdout calculado cronológicamente
inicio_holdout_calc, fin_holdout_calc = obtener_rango_holdout(
    niveles_datos, observados_datos, modo=MODO_PARTICION_BACKTEST, porcentaje_train=PORCENTAJE_TRAIN_BACKTEST,
)
ultimo_origen_valido = fin_holdout_calc - pd.Timedelta(days=HORIZONTE_BACKTEST)

estaciones_disponibles = sorted(predicciones["Estacion"].unique()) if not predicciones.empty else ESTACIONES_OBJETIVO
opciones_estacion = ["Todas las estaciones"] + estaciones_disponibles

with st.sidebar:
    st.markdown("### 📍 Selección Global")
    seleccion_estacion = st.selectbox(
        "Estación hidrológica (Global)",
        opciones_estacion,
        format_func=lambda e: "Todas las estaciones" if e == "Todas las estaciones" else NOMBRES_ESTACIONES.get(e, e),
    )

    st.markdown("---")
    st.markdown("### ⏱️ Horizonte Global")
    horizontes_disponibles = sorted(predicciones["Horizonte"].unique()) if not predicciones.empty else [HORIZONTE_PRINCIPAL]
    horizonte_default = HORIZONTE_PRINCIPAL if HORIZONTE_PRINCIPAL in horizontes_disponibles else horizontes_disponibles[-1]
    horizonte_seleccionado = st.select_slider(
        "Días de anticipación (h)",
        options=horizontes_disponibles,
        value=horizonte_default,
    )

    st.markdown("---")
    st.markdown("### 🤖 Modelos Globales")
    if not predicciones.empty:
        if seleccion_estacion == "Todas las estaciones":
            modelos_disponibles = sorted(predicciones["Modelo"].unique())
        else:
            modelos_disponibles = sorted(predicciones.loc[predicciones["Estacion"] == seleccion_estacion, "Modelo"].unique())
    else:
        modelos_disponibles = MODELOS_DEFAULT

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

pestana_grafica, pestana_mapa, pestana_backtest = st.tabs([
    "📈 Nivel Real vs Predicción de Modelos",
    "🗺️ Mapa del Río y Estaciones",
    "🧪 Simulador de Backtesting",
])

with pestana_grafica:
    if predicciones.empty:
        st.warning("⚠️ No se encontraron predicciones guardadas en `outputs/predictions.csv`.")
        st.info("Ejecuta `python main.py` en la terminal para generar los pronósticos batch completos.")
    elif not modelos_seleccionados:
        st.info("💡 Por favor, selecciona al menos un modelo en la barra lateral.")
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

with pestana_backtest:
    st.markdown("### 🧪 Simulador Interactivo de Backtesting Histórico (1 a 14 días)")
    st.markdown(
        "Evalúa el comportamiento de los modelos en cualquier fecha histórica $t_0$ dentro del **conjunto holdout**, "
        "revisando la historia previa observada, el pronóstico continuo a 14 días y el nivel real registrado."
    )

    # Panel de metadatos de la partición
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

    # Panel superior de controles
    with st.container():
        st.markdown("#### ⚙️ Parámetros de Simulación")
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.5, 1.5, 1.5, 1.5])

        modelos_backtest_disponibles = list(REGISTRO_MODELOS.keys())
        with ctrl1:
            modelo_btest = st.selectbox(
                "Modelo a evaluar",
                modelos_backtest_disponibles,
                index=modelos_backtest_disponibles.index("Ridge") if "Ridge" in modelos_backtest_disponibles else 0,
                key="sb_modelo_backtest",
            )

        with ctrl2:
            opciones_est_btest = ["Todas las estaciones"] + ESTACIONES
            estacion_btest = st.selectbox(
                "Estación",
                opciones_est_btest,
                index=opciones_est_btest.index("Ciudad_Bolivar"),
                format_func=lambda e: "Todas las estaciones" if e == "Todas las estaciones" else NOMBRES_ESTACIONES.get(e, e),
                key="sb_estacion_backtest",
            )

        with ctrl3:
            # Fecha por defecto: fecha media del holdout
            fecha_def = inicio_holdout_calc + (ultimo_origen_valido - inicio_holdout_calc) / 2
            fecha_origen_sel = st.date_input(
                "Fecha de origen (t₀)",
                value=fecha_def.date(),
                min_value=inicio_holdout_calc.date(),
                max_value=ultimo_origen_valido.date(),
                key="di_fecha_origen",
            )

        with ctrl4:
            lookbacks_permitidos = REGISTRO_MODELOS[modelo_btest]["lookbacks_soportados"]
            lookback_val = 60 if 60 in lookbacks_permitidos else lookbacks_permitidos[0]
            lookback_sel = st.select_slider(
                "Ventana histórica (lookback)",
                options=LOOKBACKS_BACKTEST,
                value=lookback_val,
                disabled=(len(lookbacks_permitidos) == 1),
                key="ss_lookback",
            )
            if len(lookbacks_permitidos) == 1:
                st.caption(f"ℹ️ {modelo_btest} requiere lookback fijo de {lookbacks_permitidos[0]} días.")

        col_ejecutar, _ = st.columns([1.5, 4.5])
        with col_ejecutar:
            ejecutar_sim = st.button("🚀 Ejecutar simulación", type="primary", use_container_width=True)

    # Tarjeta de Temporada Hidrológica para t0
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

    # Ejecución o recuperación del estado
    clave_estado = f"sim_{modelo_btest}_{estacion_btest}_{t0_timestamp.strftime('%Y%m%d')}_{lookback_sel}"

    if ejecutar_sim:
        with st.spinner(f"Entrenando/cargando {modelo_btest} y simulando evento a 14 días..."):
            try:
                estaciones_a_simular = ESTACIONES if estacion_btest == "Todas las estaciones" else [estacion_btest]
                res_evento = simular_evento(
                    nombre_modelo=modelo_btest,
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
                st.success("✅ Simulación completada exitosamente.")
            except BacktestValidationError as e:
                st.error(f"❌ Error de validación: {e}")
            except Exception as e:
                st.error(f"❌ Error durante la simulación: {e}")

    # Renderizar resultados si existen en session_state
    if clave_estado in st.session_state:
        simulacion = st.session_state[clave_estado]
        df_preds = simulacion["predicciones"]
        df_hist = simulacion["historia"]
        df_gt = simulacion["ground_truth"]
        df_mets = simulacion["metricas"]

        st.markdown("---")
        st.markdown(f"### 📊 Resultados del Evento: {modelo_btest} desde {t0_timestamp.strftime('%d/%m/%Y')}")

        estaciones_simuladas = df_preds["Estacion"].unique()

        for est_sim in estaciones_simuladas:
            nombre_legible = NOMBRES_ESTACIONES.get(est_sim, est_sim)
            sub_preds = df_preds[df_preds["Estacion"] == est_sim]
            sub_hist = df_hist[df_hist["Estacion"] == est_sim]
            sub_gt = df_gt[df_gt["Estacion"] == est_sim]
            sub_met = df_mets[df_mets["Estacion"] == est_sim].iloc[0]

            st.markdown(f"#### 📍 Estación: {nombre_legible}")

            # Métricas en tarjetas
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

            if sub_met.get("advertencia"):
                st.warning(f"⚠️ {sub_met['advertencia']}")

            # Gráfica del evento
            fig_evento = construir_grafica_evento(
                historia=sub_hist,
                ground_truth=sub_gt,
                predicciones=sub_preds,
                fecha_origen=t0_timestamp,
                temporada=info_temp,
                nombre_est=est_sim,
                nombre_modelo=modelo_btest,
            )
            st.plotly_chart(fig_evento, use_container_width=True)

            # Tabla comparativa diaria
            st.markdown("##### 📋 Detalle Diario del Pronóstico vs Real")
            tabla_mostrar = sub_preds[[
                "Horizonte", "Fecha", "Real", "Prediccion", "Error", "ErrorAbsoluto", "Observado",
            ]].copy()
            tabla_mostrar["Fecha"] = pd.to_datetime(tabla_mostrar["Fecha"]).dt.strftime("%d/%m/%Y")

            st.dataframe(
                tabla_mostrar.style.format({
                    "Real": "{:.3f}",
                    "Prediccion": "{:.3f}",
                    "Error": "{:+.3f}",
                    "ErrorAbsoluto": "{:.3f}",
                }, na_rep="— (No Observado)")
                .apply(lambda row: ["background-color: rgba(56, 189, 248, 0.12)" if row["Horizonte"] >= 7 else "" for _ in row], axis=1),
                use_container_width=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("👆 Selecciona los parámetros y pulsa **Ejecutar simulación** para iniciar el análisis histórico del evento.")
