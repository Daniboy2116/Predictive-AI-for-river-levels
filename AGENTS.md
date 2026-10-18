# AGENTS.md — Predicción de Niveles de Ríos

## Rol
Actuar como Ingeniero Senior de IA y Ciencia de Datos. Código modular, tipado, documentado (docstrings en español) y reproducible (semillas fijas).

## Estaciones
`Palua`, `Orinoco`, `Caicara`, `Ciudad_Bolivar`, `Ayacucho` (nombres de columna en `data/datos_rios.xlsx`).

## Datos
- Mientras no exista el Excel real, usar `create_dummy_data.py` → `data/datos_rios.xlsx`.
- Columna `Fecha` diaria (2020-01-01 a 2025-12-31) + una columna numérica por estación (niveles en metros).
- El dataset simulado incluye estacionalidad anual, tendencia leve, ruido y ~2% de NaN para probar la imputación.

## Estructura
```
create_dummy_data.py
data/datos_rios.xlsx
src/__init__.py
src/preprocessing.py      # carga, interpolación temporal, lags t-1..t-7, escalado, split 70/15/15
src/evaluate.py           # RMSE, MAE, R², exportación CSV/JSON
src/models/__init__.py
src/models/arima_model.py # ARIMA / Auto-ARIMA
src/models/tree_models.py # RandomForest, XGBoost
src/models/neural_models.py # LSTM, CNN1D, Transformer
src/models/google_model.py  # TimesFM / Prophet (fallback)
outputs/                  # predictions.csv, metrics.json
app.py                    # Streamlit + Plotly
main.py                   # entrenamiento y evaluación secuencial
requirements.txt
```

## Reglas técnicas
1. **Preprocesamiento:** fecha → `datetime` como índice; `interpolate(method="time")`; lags t-1..t-7; escalado Min-Max o Z-Score guardando escaladores (joblib) para desnormalizar.
2. **Split cronológico estricto** 70% train / 15% val / 15% test. **Nunca** mezclar aleatoriamente. Ajustar escaladores solo con train (evitar *data leakage*).
3. **Modelos:** ARIMA; RandomForestRegressor y XGBRegressor con lags; LSTM, CNN 1D (Conv1D + MaxPooling1D) y Transformer (self-attention); TimesFM o Prophet.
4. **Evaluación:** métricas sobre test en escala original: RMSE, MAE, R². Guardar `outputs/predictions.csv` y `outputs/metrics.json`.
5. **App:** Streamlit, selector de río en sidebar, checkboxes de modelos, gráfica Plotly Real vs Predicciones, tabla de métricas.
6. Las dependencias pesadas (TimesFM, Prophet, TensorFlow/PyTorch) deben importarse de forma perezosa y fallar con mensajes claros.
7. Avanzar paso a paso; validar cada módulo antes de continuar.
