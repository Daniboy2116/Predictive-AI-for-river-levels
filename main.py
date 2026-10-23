"""Pipeline completo: genera datos (si faltan), entrena y evalúa todos los modelos.

Uso:
    python main.py

Los modelos se ejecutan uno detrás de otro y cada uno guarda sus resultados en
``outputs/predictions.csv`` y ``outputs/metrics.json``. Después se puede abrir
el dashboard con:  python -m streamlit run app.py
"""
import time
from pathlib import Path

import create_dummy_data
from src.evaluate import cargar_metricas
from src.models import arima_model, google_model, neural_models, tree_models

RUTA_DATOS = Path(__file__).resolve().parent / "data" / "datos_rios.xlsx"


def main() -> None:
    """Ejecuta secuencialmente todos los modelos y muestra un resumen de métricas."""
    # 1) Si todavía no existe el Excel real, generamos el dataset simulado
    if not RUTA_DATOS.exists():
        create_dummy_data.main()

    # 2) Entrenar y evaluar cada familia de modelos (en todas las estaciones)
    inicio = time.perf_counter()

    print("\n=== ARIMA ===")
    arima_model.ejecutar()

    print("\n=== ÁRBOLES (Random Forest y XGBoost) ===")
    tree_models.ejecutar()

    print("\n=== REDES NEURONALES (LSTM, CNN1D y Transformer) ===")
    neural_models.ejecutar()

    print("\n=== PROPHET ===")
    google_model.ejecutar()

    duracion = time.perf_counter() - inicio
    print("\nTiempo total: %.1f segundos" % duracion)

    # 3) Resumen final ordenado de mejor a peor RMSE en cada estación
    tabla = cargar_metricas()
    tabla = tabla.sort_values(["Estacion", "RMSE"])
    print("\n=== RESUMEN (ordenado por RMSE) ===")
    print(tabla.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
