# Predicción del nivel del río Orinoco a 15 días

Pronóstico del nivel diario del Orinoco en **Ciudad Bolívar** y **Palúa** con 15 días de anticipación,
usando también las estaciones aguas arriba (**Puerto Ayacucho** y **Caicara**). Se comparan líneas base,
modelos estadísticos, ensamblados de árboles, redes neuronales y un ensamble apilado.

> ⚠️ Mientras no lleguen los archivos reales, `create_dummy_data.py` genera datos **simulados** con las
> propiedades del análisis exploratorio (`Nivel_Orinoco_Analisis_y_Modelos.pdf`). Los resultados de abajo
> son sobre esos datos simulados y solo sirven para validar el flujo.

## Inicio rápido

```bash
pip install -r requirements.txt
python main.py                      # datos, control de calidad, todos los modelos (~30-40 min en CPU)
python -m streamlit run app.py      # dashboard en http://localhost:8501
python -m pytest                    # pruebas (fugas de información, split, métricas)
```

Para ejecutar solo algunas familias de modelos:

```bash
python main.py --familias lineas_base lineal arboles ensamble
```

Familias: `lineas_base`, `lineal`, `arboles`, `sarimax`, `prophet`, `redes`, `ensamble`, `contribucion`.
Cada módulo de `src/models/` también se puede ejecutar solo, por ejemplo `python -m src.models.tree_models`.

## Metodología

| Paso | Qué se hace |
|------|-------------|
| Control de calidad | Detecta cambios del cero de la regla comparando cada estación con dos vecinas; corrige los confirmados en `src/configuracion.py` |
| Imputación | Interpolación temporal; las métricas se calculan **solo con días observados** |
| Objetivo | Pronóstico directo: ŷ(t+h) = y(t) + f(x(t)); el modelo aprende el **cambio** a h días |
| Variables (79) | Cíclicas (7), rezagos (40), tendencias (20) y suavizado (12) de las cuatro estaciones, todas conocidas en t |
| Partición | Train 1974–2014 · validación 2015–2019 · prueba 2020–2025 (por fecha objetivo) |
| Sin fugas | Climatología y escaladores ajustados solo con train; prueba automática de fuga de información futura |
| Métricas | MAE, RMSE, NSE (Nash-Sutcliffe) y mejora del MAE frente a la persistencia |

### Modelos

| Familia | Modelos | Archivo |
|---------|---------|---------|
| Líneas base | Persistencia, Climatología armónica, Anomalía amortiguada | `src/models/linea_base.py` |
| Lineal | Ridge | `src/models/lineal_model.py` |
| Árboles | Random Forest, XGBoost, LightGBM | `src/models/tree_models.py` |
| Estadístico | SARIMAX (anomalía ARMA + exógenas aguas arriba) | `src/models/arima_model.py` |
| Google/Meta | Prophet con regresores aguas arriba | `src/models/google_model.py` |
| Redes neuronales | LSTM, CNN 1D causal dilatada, Transformer (salida de 15 horizontes) | `src/models/neural_models.py` |
| Ensamble | Apilamiento con pesos no negativos ajustados en validación | `src/models/ensamble.py` |

## Usar los datos reales

1. Guardar el Excel en `data/datos_rios.xlsx` con una columna `Fecha` (diaria) y una columna por estación:
   `Ayacucho`, `Caicara`, `Ciudad_Bolivar`, `Palua` (niveles en metros).
2. Ejecutar `python main.py --familias lineas_base`: el control de calidad lista los saltos detectados.
3. Verificar cada salto confirmado con quien opera la estación y declararlo en `CAMBIOS_DE_REFERENCIA`
   (`src/configuracion.py`). Si el salto es real, se corrige; si no, se deja fuera de la lista.
4. Ejecutar `python main.py` completo.

## Estructura

```
create_dummy_data.py        # datos simulados (mientras no estén los reales)
data/datos_rios.xlsx
src/configuracion.py        # estaciones, horizontes, fechas de corte, cambios de referencia
src/preprocessing.py        # carga, control de calidad, imputación, climatología, variables, split, escalado
src/evaluate.py             # métricas y exportación a outputs/
src/contribucion.py         # análisis de contribución por grupos de variables
src/models/                 # un archivo por familia de modelos (4 pasos cada uno)
tests/                      # pruebas pytest
app.py                      # dashboard Streamlit + Plotly
main.py                     # flujo completo
proyecto_completo.ipynb     # recorrido educativo de principio a fin
```
