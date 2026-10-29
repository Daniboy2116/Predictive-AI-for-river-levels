# Predicción del nivel del río Orinoco a 15 días

Pronóstico del nivel diario del Orinoco en **Ciudad Bolívar** y **Palúa** con 15 días de anticipación,
usando también las estaciones aguas arriba (**Puerto Ayacucho** y **Caicara**). Se comparan líneas base,
modelos estadísticos, ensamblados de árboles, redes neuronales y un ensamble apilado.

> ⚠️ Mientras no lleguen los archivos reales, `create_dummy_data.py` genera datos **simulados** con las
> propiedades del análisis exploratorio (`Nivel_Orinoco_Analisis_y_Modelos.pdf`). Los resultados de abajo
> son sobre esos datos simulados y solo sirven para validar el flujo.

## Qué ofrece

- **Pronóstico de cualquier día hasta 15 días:** en el dashboard (pestaña *Pronóstico por día*) se elige una
  fecha de emisión y cuántos días ver (1 a 15) y aparece la trayectoria de cada modelo junto al nivel observado.
  Con la fecha más reciente se ve el **pronóstico operativo**: los 15 días siguientes al último dato.
- **Evaluación honesta por horizonte:** MAE, RMSE y NSE para cada horizonte de 1 a 15 días en 2020–2025.
- **Comparación de 13 modelos** y análisis de qué variables aportan más.

## Resultados con los datos simulados (MAE a 15 días, prueba 2020–2025)

| Modelo | Ciudad Bolívar | Palúa | Mejora vs. persistencia (CB) |
|--------|---------------:|------:|-----------------------------:|
| Persistencia | 126,0 cm | 91,8 cm | — |
| Climatología armónica | 128,2 cm | 113,2 cm | −2 % |
| Anomalía amortiguada | 51,7 cm | 38,8 cm | 59 % |
| SARIMAX | 48,1 cm | 31,4 cm | 62 % |
| Random Forest | 38,7 cm | 29,3 cm | 69 % |
| XGBoost | 37,5 cm | 28,4 cm | 70 % |
| LightGBM | 37,1 cm | 28,5 cm | 71 % |
| Prophet | 34,7 cm | 27,3 cm | 72 % |
| Transformer | 33,6 cm | 27,3 cm | 73 % |
| CNN 1D | 33,1 cm | 27,3 cm | 74 % |
| LSTM | 33,0 cm | 27,3 cm | 74 % |
| Ridge | 31,0 cm | **24,1 cm** | 75 % |
| **Ensamble** | **30,9 cm** | 24,6 cm | **75 %** |

Las líneas base son muy parecidas a las del PDF (persistencia 109/82 cm, anomalía amortiguada 55/48 cm).
Con estos datos simulados, que son casi lineales por construcción, Ridge es tan bueno como los modelos
complejos; con los datos reales el orden puede cambiar (en el PDF, LightGBM supera ligeramente a Ridge).
Quitar las otras estaciones empeora el MAE de Ridge de 31,2 a 36,9 cm: la mayor ganancia está aguas arriba.

## Inicio rápido

```bash
pip install -r requirements.txt
python main.py                      # datos, control de calidad, todos los modelos (~30 min en CPU)
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
| Partición | Train 2009–2016 · validación 2017–2019 · prueba 2020–feb 2025 (por fecha objetivo) |
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

1. Los datos reales están en `data/niveles_reales.xlsx`, hoja `Niveles` (columna `Fecha` diaria de 2009-01-01 a
   2025-02-24 y una columna por estación, en metros). Salen de la auditoría de `data/auditoria/` (fuente: hojas por
   estación de `DATOS RIO ORINOCO.xlsx`; sin imputados, sin días aberrantes; serie completa 1935-2025 en
   `data/auditoria/dataset_real_1935_2025.csv`). Hoja, ruta y correspondencia de encabezados: `src/configuracion.py`
   (`RUTA_DATOS`, `HOJA_DATOS`, `COLUMNAS_EXCEL`). Las filas finales sin datos se ignoran.
2. Ejecutar `python main.py --familias lineas_base`: el control de calidad lista los saltos detectados.
3. Verificar cada salto confirmado con quien opera la estación y declararlo en `CAMBIOS_DE_REFERENCIA`
   (`src/configuracion.py`). Si el salto es real, se corrige; si no, se deja fuera de la lista.
4. Ejecutar `python main.py` completo.

## Estructura

```
create_dummy_data.py        # datos simulados (solo para las pruebas del control de calidad)
data/niveles_reales.xlsx    # datos reales auditados (hoja Niveles)
data/auditoria/             # informe, scripts y serie real 1935-2025
data/datos_rios.xlsx        # datos simulados
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
