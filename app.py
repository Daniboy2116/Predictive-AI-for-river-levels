"""Dashboard Streamlit: pronóstico del nivel del Orinoco y comparación de modelos.

Ejecutar:  python -m streamlit run app.py
(antes hay que generar los resultados con:  python main.py)
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.configuracion import (
    CAMBIOS_DE_REFERENCIA, ESTACIONES, HORIZONTE_PRINCIPAL, NOMBRES_ESTACIONES, UMBRALES_ALERTA,
)
from src.contribucion import RUTA_CONTRIBUCION, cargar_contribucion
from src.evaluate import RUTA_METRICAS, RUTA_PREDICCIONES, cargar_metricas, cargar_predicciones
from src.preprocessing import leer_excel, revisar_calidad

COLORES_MODELOS = {
    "Persistencia": "#94a3b8", "Climatologia": "#64748b", "AnomaliaAmortiguada": "#cbd5e1",
    "Ridge": "#38bdf8", "RandomForest": "#22c55e", "XGBoost": "#10b981", "LightGBM": "#a3e635",
    "SARIMAX": "#f59e0b", "Prophet": "#fb923c", "LSTM": "#a78bfa", "CNN1D": "#f472b6",
    "Transformer": "#60a5fa", "Ensamble": "#facc15",
}
COLORES_ESTACIONES = {"Ayacucho": "#a78bfa", "Caicara": "#38bdf8", "Ciudad_Bolivar": "#f59e0b", "Palua": "#22c55e"}
MODELOS_POR_DEFECTO = ["Persistencia", "Ridge", "LightGBM", "Ensamble"]
ESTILO = """
<style>
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #0e3a5c 0%, transparent 60%),
         linear-gradient(160deg, #050b18 0%, #0a1628 60%, #06101f 100%); color: #e2e8f0; }
section[data-testid="stSidebar"] { background: rgba(10, 22, 40, .9); border-right: 1px solid rgba(56, 189, 248, .15); }
.titulo h1 { font-size: 2.2rem; font-weight: 700; margin: 0;
         background: linear-gradient(90deg, #38bdf8, #818cf8, #22d3ee); -webkit-background-clip: text;
         -webkit-text-fill-color: transparent; }
.titulo p { color: #94a3b8; margin-top: .2rem; }
.tarjeta { background: rgba(255,255,255,.04); border: 1px solid rgba(56,189,248,.18);
        border-radius: 16px; padding: 1rem 1.2rem; }
.tarjeta .etiqueta { color: #94a3b8; font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }
.tarjeta .valor { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; }
.tarjeta .detalle { color: #38bdf8; font-size: .85rem; }
</style>
"""


# --------------------------------------------------------------------------
# Carga de datos (se guarda en caché para no leer los archivos en cada clic)
# --------------------------------------------------------------------------
def fechas_de_modificacion() -> tuple:
    """Fecha de última modificación de cada archivo de resultados (0 si no existe)."""
    fechas = []
    for ruta in [RUTA_PREDICCIONES, RUTA_METRICAS, RUTA_CONTRIBUCION]:
        if ruta.exists():
            fechas.append(ruta.stat().st_mtime)
        else:
            fechas.append(0.0)
    return tuple(fechas)


def leer_resultados(fechas: tuple) -> tuple:
    """Lee predicciones de prueba, métricas y contribución de variables desde ``outputs/``.

    El argumento ``fechas`` no se usa dentro de la función: sirve para que la caché se
    renueve sola cuando ``main.py`` vuelve a escribir los archivos.
    """
    return cargar_predicciones("prueba"), cargar_metricas(), cargar_contribucion()


def leer_datos_y_calidad() -> tuple:
    """Lee el Excel crudo y la revisión de calidad (saltos detectados)."""
    crudos = leer_excel()
    return crudos, revisar_calidad(crudos)


# [JUSTIFICACIÓN]: st.cache_data suele usarse como decorador; aquí lo aplicamos como una
# función normal para mantener el estilo del proyecto (sin decoradores).
leer_resultados_en_cache = st.cache_data(leer_resultados)
leer_datos_y_calidad_en_cache = st.cache_data(leer_datos_y_calidad)


def tarjeta(columna, etiqueta: str, valor: str, detalle: str) -> None:
    """Dibuja una tarjeta con un indicador."""
    html = ('<div class="tarjeta"><div class="etiqueta">' + etiqueta + '</div>'
            '<div class="valor">' + valor + '</div><div class="detalle">' + detalle + '</div></div>')
    columna.markdown(html, unsafe_allow_html=True)


def nombre_estacion(estacion: str) -> str:
    """Nombre bonito de una estación para mostrar."""
    return NOMBRES_ESTACIONES[estacion]


def estilo_grafica(figura: go.Figure, titulo: str, titulo_eje_y: str, altura: int = 480) -> None:
    """Aplica el estilo oscuro común a todas las gráficas."""
    figura.update_layout(
        template="plotly_dark", height=altura, hovermode="x unified",
        title=dict(text=titulo, x=0.01, y=0.98, yanchor="top", font=dict(size=17)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
        yaxis_title=titulo_eje_y, legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"),
        margin=dict(l=10, r=10, t=95, b=10),
    )


# --------------------------------------------------------------------------
# Página
# --------------------------------------------------------------------------
st.set_page_config(page_title="Nivel del Orinoco", page_icon="🌊", layout="wide")
st.markdown(ESTILO, unsafe_allow_html=True)
st.markdown('<div class="titulo"><h1>🌊 Pronóstico del nivel del Orinoco</h1>'
            '<p>Pronóstico directo a h días con estaciones aguas arriba · prueba 2020-2025 · '
            'entrenamiento 1974-2014 · validación 2015-2019</p></div>', unsafe_allow_html=True)

predicciones, metricas, contribucion = leer_resultados_en_cache(fechas_de_modificacion())
if predicciones.empty:
    st.warning("No hay resultados todavía. Ejecuta primero: `python main.py`")
    st.stop()

# ----------------------------------------------------------------- Barra lateral
with st.sidebar:
    st.header("Configuración")
    estaciones_disponibles = []
    for estacion in ESTACIONES:
        if estacion in predicciones["Estacion"].unique():
            estaciones_disponibles.append(estacion)
    estacion = st.selectbox("Estación", estaciones_disponibles, format_func=nombre_estacion, key="estacion")

    horizontes = sorted(predicciones["Horizonte"].unique())
    horizonte_inicial = HORIZONTE_PRINCIPAL
    if horizonte_inicial not in horizontes:
        horizonte_inicial = horizontes[-1]
    horizonte = st.select_slider("Horizonte (días de anticipación)", options=horizontes,
                                 value=horizonte_inicial, key="horizonte")

    st.subheader("Modelos")
    modelos_disponibles = sorted(predicciones.loc[predicciones["Estacion"] == estacion, "Modelo"].unique())
    seleccionados = []
    for modelo in modelos_disponibles:
        marcado = st.checkbox(modelo, value=modelo in MODELOS_POR_DEFECTO, key="modelo_" + modelo)
        if marcado:
            seleccionados.append(modelo)

    if st.button("Recargar resultados", key="recargar"):
        st.cache_data.clear()
        st.rerun()

# ----------------------------------------------------------------- Filtrado
de_la_estacion = predicciones[(predicciones["Estacion"] == estacion) & (predicciones["Horizonte"] == horizonte)]
reales = de_la_estacion.drop_duplicates("Fecha").set_index("Fecha").sort_index()
reales_observados = reales[reales["Observado"]]["Real"]

metricas_horizonte = metricas[(metricas["Estacion"] == estacion) & (metricas["Horizonte"] == horizonte)]
metricas_horizonte = metricas_horizonte.sort_values("MAE").reset_index(drop=True)
metricas_seleccionadas = metricas_horizonte[metricas_horizonte["Modelo"].isin(seleccionados)]

# ----------------------------------------------------------------- Indicadores
columna1, columna2, columna3, columna4 = st.columns(4)
ultimo_dia = reales_observados.index[-1]
tarjeta(columna1, "Último nivel observado", "%.2f m" % reales_observados.iloc[-1], ultimo_dia.strftime("%d-%m-%Y"))
mejor = metricas_horizonte.iloc[0]
tarjeta(columna2, "Mejor modelo a " + str(horizonte) + " días", mejor["Modelo"], "MAE %.1f cm" % (100 * mejor["MAE"]))
tarjeta(columna3, "Mejora vs. persistencia", "%.0f %%" % mejor["Mejora_vs_persistencia"],
        "reducción del MAE a " + str(horizonte) + " días")
tarjeta(columna4, "NSE del mejor", "%.3f" % mejor["NSE"], "1 = perfecto · 0 = media")
st.write("")

pestana_pronostico, pestana_metricas, pestana_horizonte, pestana_variables, pestana_datos = st.tabs(
    ["Pronóstico", "Métricas", "Error por horizonte", "Contribución de variables", "Datos y calidad"])

# ----------------------------------------------------------------- Pronóstico
with pestana_pronostico:
    figura = go.Figure()
    figura.add_trace(go.Scatter(x=reales_observados.index, y=reales_observados.to_numpy(), name="Nivel observado",
                                mode="lines", line=dict(color="#e2e8f0", width=2.5)))
    for modelo in seleccionados:
        del_modelo = de_la_estacion[de_la_estacion["Modelo"] == modelo].sort_values("Fecha")
        figura.add_trace(go.Scatter(x=del_modelo["Fecha"], y=del_modelo["Prediccion"], name=modelo, mode="lines",
                                    opacity=0.9, line=dict(color=COLORES_MODELOS.get(modelo), width=1.5, dash="dot")))
    if estacion in UMBRALES_ALERTA:
        figura.add_hline(y=UMBRALES_ALERTA[estacion], line=dict(color="#ef4444", dash="dash"),
                         annotation_text="Umbral de alerta %.1f m" % UMBRALES_ALERTA[estacion])
    estilo_grafica(figura, "Nivel observado vs. pronóstico a " + str(horizonte) + " días — " + nombre_estacion(estacion),
                   "Nivel (m)", altura=540)
    figura.update_layout(xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)))
    st.plotly_chart(figura, theme=None)
    st.caption("Cada punto de un modelo es el pronóstico emitido " + str(horizonte) +
               " días antes de la fecha del eje. Solo se muestran días con nivel observado.")

# ----------------------------------------------------------------- Métricas
with pestana_metricas:
    st.subheader("Métricas en prueba a " + str(horizonte) + " días (de mejor a peor MAE)")
    if metricas_seleccionadas.empty:
        st.info("Selecciona al menos un modelo en la barra lateral.")
    else:
        tabla = pd.DataFrame()
        tabla["Modelo"] = metricas_seleccionadas["Modelo"]
        tabla["MAE (cm)"] = 100 * metricas_seleccionadas["MAE"]
        tabla["RMSE (cm)"] = 100 * metricas_seleccionadas["RMSE"]
        tabla["NSE"] = metricas_seleccionadas["NSE"]
        tabla["Mejora vs. persistencia (%)"] = metricas_seleccionadas["Mejora_vs_persistencia"]
        tabla.index = range(1, len(tabla) + 1)
        formato = {"MAE (cm)": "{:.1f}", "RMSE (cm)": "{:.1f}", "NSE": "{:.3f}", "Mejora vs. persistencia (%)": "{:.1f}"}
        tabla_con_estilo = tabla.style.format(formato)
        tabla_con_estilo = tabla_con_estilo.highlight_min(subset=["MAE (cm)", "RMSE (cm)"], color="#14532d")
        tabla_con_estilo = tabla_con_estilo.highlight_max(subset=["NSE"], color="#14532d")
        st.dataframe(tabla_con_estilo)
    st.caption("MAE: error medio absoluto. RMSE: penaliza más los errores grandes. NSE: eficiencia de "
               "Nash-Sutcliffe. Mejora vs. persistencia: cuánto se reduce el MAE frente a 'el río seguirá igual'.")

# ----------------------------------------------------------------- Error por horizonte
with pestana_horizonte:
    figura = go.Figure()
    metricas_estacion = metricas[metricas["Estacion"] == estacion]
    for modelo in seleccionados:
        del_modelo = metricas_estacion[metricas_estacion["Modelo"] == modelo].sort_values("Horizonte")
        figura.add_trace(go.Scatter(x=del_modelo["Horizonte"], y=100 * del_modelo["MAE"], name=modelo,
                                    mode="lines+markers", line=dict(color=COLORES_MODELOS.get(modelo), width=2)))
    estilo_grafica(figura, "MAE en prueba según el horizonte — " + nombre_estacion(estacion), "MAE (cm)")
    figura.update_layout(hovermode="closest", xaxis=dict(title="Horizonte (días)", tickvals=horizontes))
    st.plotly_chart(figura, theme=None)

# ----------------------------------------------------------------- Contribución de variables
with pestana_variables:
    if estacion not in contribucion:
        st.info("Ejecuta `python main.py --familias contribucion` para calcular este análisis.")
    else:
        figura = go.Figure()
        for nombre_modelo, colores in [("Ridge", "#38bdf8"), ("LightGBM", "#a3e635")]:
            resultados = contribucion[estacion][nombre_modelo]
            configuraciones = list(resultados.keys())
            valores = []
            textos = []
            for configuracion in configuraciones:
                valores.append(100 * resultados[configuracion])
                textos.append("%.1f" % (100 * resultados[configuracion]))
            figura.add_trace(go.Bar(y=configuraciones, x=valores, name=nombre_modelo, orientation="h",
                                    marker_color=colores, text=textos, textposition="outside"))
        estilo_grafica(figura, "MAE a " + str(HORIZONTE_PRINCIPAL) + " días al quitar cada grupo de variables — " +
                       nombre_estacion(estacion), "")
        figura.update_layout(barmode="group", hovermode="closest", xaxis_title="MAE en prueba (cm)",
                             yaxis=dict(autorange="reversed"))
        st.plotly_chart(figura, theme=None)
        st.caption("Si al quitar un grupo el error sube mucho, ese grupo aporta mucha información.")

# ----------------------------------------------------------------- Datos y calidad
with pestana_datos:
    crudos, revision = leer_datos_y_calidad_en_cache()
    figura = go.Figure()
    for nombre in ESTACIONES:
        figura.add_trace(go.Scatter(x=crudos.index, y=crudos[nombre], name=nombre_estacion(nombre), mode="lines",
                                    line=dict(color=COLORES_ESTACIONES[nombre], width=1)))
    confirmados = revision[revision["Confirmado"]].drop_duplicates(["Estacion", "Fecha"])
    for _, fila in confirmados.iterrows():
        figura.add_vline(x=fila["Fecha"], line=dict(color="#ef4444", dash="dot"))
    estilo_grafica(figura, "Registro diario sin corregir (líneas rojas: cambios de referencia detectados)", "Nivel (m)")
    figura.update_layout(xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)))
    st.plotly_chart(figura, theme=None)

    columna_izquierda, columna_derecha = st.columns(2)
    with columna_izquierda:
        st.subheader("Saltos detectados")
        revision_para_mostrar = revision.copy()
        revision_para_mostrar["Fecha"] = revision_para_mostrar["Fecha"].dt.strftime("%Y-%m-%d")
        st.dataframe(revision_para_mostrar, hide_index=True)
        st.caption("Confirmado = aparece frente a las dos estaciones vecinas. Los no confirmados suelen ser "
                   "el reflejo del salto de una vecina.")
    with columna_derecha:
        st.subheader("Correcciones aplicadas")
        correcciones = pd.DataFrame(CAMBIOS_DE_REFERENCIA, columns=["Estacion", "Fecha", "Salto (m)"])
        st.dataframe(correcciones, hide_index=True)
        st.subheader("Días sin dato")
        st.dataframe(crudos.isna().sum().rename("Días sin dato"))
