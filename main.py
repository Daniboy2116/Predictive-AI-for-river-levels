"""Pipeline completo: datos, control de calidad, modelos, ensamble y resumen.

Uso:
    python main.py                              # todo (tarda varios minutos)
    python main.py --familias lineas_base lineal arboles ensamble   # solo algunas familias

Familias disponibles (se ejecutan en este orden):
    lineas_base, lineal, arboles, sarimax, prophet, redes, ensamble, contribucion

Cada familia guarda sus resultados en ``outputs/`` sin borrar los de las demás.
Después se puede abrir el dashboard con:  python -m streamlit run app.py
"""
import argparse
import time

from src import contribucion
from src.configuracion import CAMBIOS_DE_REFERENCIA, HORIZONTE_PRINCIPAL
from src.evaluate import cargar_metricas
from src.models import arima_model, ensamble, google_model, linea_base, lineal_model, neural_models, tree_models
from src.preprocessing import leer_excel, revisar_calidad

FAMILIAS = {
    "lineas_base": linea_base.ejecutar,
    "lineal": lineal_model.ejecutar,
    "arboles": tree_models.ejecutar,
    "arima": arima_model.ejecutar,
    "sarimax": arima_model.ejecutar,
    "timesfm": google_model.ejecutar_timesfm,
    "prophet": google_model.ejecutar_prophet,
    "redes": neural_models.ejecutar,
    "ensamble": ensamble.ejecutar,
    "contribucion": contribucion.ejecutar,
}


def leer_argumentos() -> argparse.Namespace:
    """Lee de la línea de comandos qué familias de modelos ejecutar."""
    lector = argparse.ArgumentParser(description="Entrena y evalúa los modelos de nivel del Orinoco.")
    lector.add_argument("--familias", nargs="+", choices=list(FAMILIAS.keys()), default=list(FAMILIAS.keys()),
                        help="familias a ejecutar (por defecto, todas)")
    return lector.parse_args()


def mostrar_control_de_calidad() -> None:
    """Imprime los saltos detectados y los cambios de referencia que se están corrigiendo."""
    revision = revisar_calidad(leer_excel())
    confirmados = revision[revision["Confirmado"]].drop_duplicates(["Estacion", "Fecha"])
    print("Saltos confirmados (frente a dos estaciones vecinas):")
    if confirmados.empty:
        print("  ninguno")
    else:
        print(confirmados[["Estacion", "Fecha", "Salto"]].to_string(index=False))

    print("Cambios de referencia que se corrigen (src/configuracion.py):")
    for estacion, fecha, salto in CAMBIOS_DE_REFERENCIA:
        print("  ", estacion, fecha, "%+.2f m" % salto)
    print("Si aparece un salto confirmado que no está en la lista, hay que verificarlo y añadirlo.")


def main() -> None:
    """Ejecuta el flujo completo y muestra el resumen a 15 días."""
    argumentos = leer_argumentos()

    # 1) Control de calidad (lee la hoja HOJA_DATOS de RUTA_DATOS)
    print("\n=== CONTROL DE CALIDAD ===")
    mostrar_control_de_calidad()

    # 2) Entrenar y evaluar cada familia de modelos
    inicio = time.perf_counter()
    for nombre_familia in FAMILIAS:
        if nombre_familia in argumentos.familias:
            print("\n=== " + nombre_familia.upper() + " ===")
            FAMILIAS[nombre_familia]()
    duracion = time.perf_counter() - inicio
    print("\nTiempo total: %.1f minutos" % (duracion / 60))

    # 3) Resumen: horizonte principal, ordenado de mejor a peor MAE
    tabla = cargar_metricas()
    tabla = tabla[tabla["Horizonte"] == HORIZONTE_PRINCIPAL]
    tabla = tabla.sort_values(["Estacion", "MAE"])
    tabla["MAE_cm"] = 100 * tabla["MAE"]
    tabla["RMSE_cm"] = 100 * tabla["RMSE"]
    print("\n=== RESUMEN A " + str(HORIZONTE_PRINCIPAL) + " DÍAS (prueba 2020-2025) ===")
    columnas = ["Estacion", "Modelo", "MAE_cm", "RMSE_cm", "NSE", "Mejora_vs_persistencia"]
    print(tabla[columnas].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
