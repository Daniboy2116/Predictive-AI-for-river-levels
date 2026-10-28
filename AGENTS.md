# AGENTS.md — Predicción de Niveles de Ríos

## Rol
Actuar como Ingeniero Senior de IA y Ciencia de Datos. Código modular, tipado, documentado (docstrings en español) y reproducible (semillas fijas).

## Objetivo
Pronosticar el nivel del río Orinoco (en metros) a **15 días vista** en `Ciudad_Bolivar` y `Palua`,
aprovechando la información de las estaciones aguas arriba. El análisis de referencia está en
`Nivel_Orinoco_Analisis_y_Modelos.pdf` (persistencia: MAE 109,2 cm / 82,0 cm; LightGBM: 38,5 cm / 36,1 cm).

## Estaciones
De aguas arriba a aguas abajo: `Ayacucho`, `Caicara`, `Ciudad_Bolivar`, `Palua` (nombres de columna en `data/datos_rios.xlsx`).
- Estaciones objetivo: `Ciudad_Bolivar` y `Palua` (configurable en `src/configuracion.py`).
- Las cuatro estaciones se usan como variables de entrada.

## Datos
- Datos reales: `data/niveles_reales.xlsx`, hoja `Niveles` (2009-01-01 a 2025-02-24), generado por la auditoría de `data/auditoria/`.
  No usar la hoja `Entrenar` (Ciudad Bolívar corrida 1 día en 2020-2021) ni los archivos imputados (Ayacucho 2007-2008 es sintético).
  Los cambios de referencia de Ciudad Bolívar 1987 y 2019 resultaron ser pulsos transitorios, no escalones permanentes.
- `create_dummy_data.py` → `data/datos_rios.xlsx` queda solo para probar el control de calidad (saltos conocidos).
- Columna `Fecha` diaria (1974-01-01 a 2025-12-31) + una columna numérica por estación (niveles en metros).
- El dataset simulado reproduce lo descrito en el PDF: ciclo anual con máximo en agosto y mínimo en marzo,
  onda de crecida que viaja aguas abajo (Ayacucho → Ciudad Bolívar ≈ 18 días, Caicara → Ciudad Bolívar ≈ 4 días,
  Ciudad Bolívar y Palúa simultáneas), anomalías con memoria larga (~70 días), huecos de datos y dos
  cambios de referencia en Ciudad Bolívar (15-05-1987: −2,20 m; 01-01-2019: +1,00 m).

## Estructura
```
create_dummy_data.py
data/datos_rios.xlsx
src/__init__.py
src/configuracion.py        # estaciones, horizontes, fechas de corte, rutas, semilla
src/preprocessing.py        # carga, control de calidad, imputación, climatología, variables, split, escalado
src/evaluate.py             # MAE, RMSE, NSE, mejora vs persistencia, exportación CSV/JSON
src/contribucion.py         # ablación por grupos de variables (Figura 8 del PDF)
src/models/__init__.py
src/models/linea_base.py    # Persistencia, Climatología armónica, Anomalía amortiguada
src/models/lineal_model.py  # Regresión Ridge
src/models/tree_models.py   # RandomForest, XGBoost, LightGBM
src/models/arima_model.py   # SARIMAX (anomalía + exógenas aguas arriba)
src/models/google_model.py  # Prophet
src/models/neural_models.py # LSTM, CNN1D causal dilatada, Transformer (salida multi-horizonte, atajo lineal)
src/models/ensamble.py      # Ensamble apilado (stacking)
tests/                      # pruebas pytest (fugas de información, split, métricas)
outputs/                    # predictions.csv, predicciones_validacion.csv, metrics.json, escaladores
app.py                      # Streamlit + Plotly (tema en .streamlit/config.toml)
main.py                     # entrenamiento y evaluación secuencial
proyecto_completo.ipynb     # flujo completo, educativo, de principio a fin
requirements.txt
pytest.ini
```

## Reglas técnicas
1. **Preprocesamiento:** fecha → `datetime` como índice con frecuencia diaria; corregir los cambios de referencia
   declarados en la configuración; imputar con `interpolate(method="time")` pero **evaluar solo en días observados**.
2. **Pronóstico directo:** ŷ(t+h) = y(t) + f(x(t)). El modelo aprende el **cambio** y(t+h) − y(t) usando solo
   información disponible hasta el día t. Se pronostica y evalúa cada horizonte de 1 a 15 días (principal: 15).
   Desde el último dato disponible se emite además el **pronóstico operativo** de los próximos 15 días
   (se guarda en `predictions.csv` con `Real` vacío y `Observado = False`; no entra en las métricas).
3. **Variables (79):** cíclicas (seno/coseno k = 1..3 del día del año en t+h y cambio climatológico esperado),
   rezagos y(t−L) con L ∈ {0, 1, 2, 3, 5, 7, 10, 15, 20, 30}, tendencias Δw = y(t) − y(t−w) con
   w ∈ {1, 3, 7, 15, 30}, medias móviles de 7 y 30 días y anomalía. Rezagos, tendencias y suavizado se calculan
   para las cuatro estaciones.
4. **Split cronológico estricto por años:** train 2009–2016, validación 2017–2019, prueba 2020–2025 (según la
   fecha objetivo t+h). **Nunca** mezclar aleatoriamente. La climatología armónica y los escaladores
   (Z-Score, guardados con joblib) se ajustan **solo con train** para evitar *data leakage*.
   Todos los modelos se entrenan con train; validación sirve para *early stopping*, selección y para ajustar el ensamble.
5. **Modelos:** líneas base (persistencia, climatología armónica, anomalía amortiguada); Ridge; RandomForest,
   XGBoost y LightGBM; SARIMAX; LSTM, CNN 1D (Conv1D causal dilatada + MaxPooling1D) y Transformer (self-attention);
   Prophet; ensamble apilado.
6. **Evaluación:** métricas sobre prueba en metros: MAE, RMSE, NSE (Nash-Sutcliffe) y mejora del MAE respecto a la
   persistencia. Guardar `outputs/predictions.csv` y `outputs/metrics.json`.
7. **App:** Streamlit, selector de estación y horizonte en sidebar, checkboxes de modelos. Pestaña
   "Pronóstico por día": se elige cualquier fecha de emisión (prueba o último dato) y cuántos días ver (1-15)
   y se muestra la trayectoria de cada modelo junto al nivel observado. Además: gráfica Plotly Real vs
   Predicciones con umbral de alerta, tabla de métricas y curva de error por horizonte.
8. **Código educativo:** imports normales al inicio del archivo (sin importaciones perezosas, `try/except ImportError`
   ni *fallbacks*), sin decoradores ni sintaxis hipercompacta. Cada modelo sigue 4 pasos marcados con encabezados
   (`# === PASO 1: PREPARACIÓN DE DATOS ===`, definición, entrenamiento, predicción + inversión de escala).
   Si un modelo se aparta de la plantilla, añadir un comentario `# [JUSTIFICACIÓN]: ...`.
9. Avanzar paso a paso; validar cada módulo antes de continuar (`python -m pytest`).
