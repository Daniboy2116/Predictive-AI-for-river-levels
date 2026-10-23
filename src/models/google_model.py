"""Prophet (Meta, muy usado junto a herramientas de Google Cloud) con regresores aguas arriba.

Prophet descompone la serie en tendencia + estacionalidad anual + efecto lineal de
"regresores externos". Para pronosticar a h días de forma directa:

- ``ds`` es la fecha objetivo t + h y ``y`` el nivel ese día.
- Los regresores son valores conocidos en el día de emisión t: el nivel y la
  tendencia de 7 días de las cuatro estaciones.

Así Prophet aprende "cuánto pesa" lo que pasa hoy río arriba en el nivel de dentro
de h días, encima del ciclo anual.

Ejecutar:  python -m src.models.google_model
"""
import logging

import pandas as pd
from prophet import Prophet

from src.configuracion import ESTACIONES, ESTACIONES_OBJETIVO, HORIZONTES
from src.evaluate import armar_predicciones, evaluar_y_guardar
from src.preprocessing import cargar_datos, preparar_datos

# Prophet (y su motor cmdstanpy) imprimen muchos mensajes; mostramos solo avisos importantes.
# cmdstanpy reinicia su nivel si no tiene "handlers", por eso le añadimos uno que no hace nada
# (los avisos siguen llegando a la consola a través del registro raíz).
registro_cmdstanpy = logging.getLogger("cmdstanpy")
registro_cmdstanpy.addHandler(logging.NullHandler())
registro_cmdstanpy.setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# Variables de la tabla (conocidas en t) que se usan como regresores
REGRESORES = []
for _estacion in ESTACIONES:
    REGRESORES.append(_estacion + "_rezago_0")
    REGRESORES.append(_estacion + "_tendencia_7")


def tabla_prophet(tabla: pd.DataFrame) -> pd.DataFrame:
    """Convierte una tabla de ``preparar_datos`` al formato de Prophet: ds, y + regresores."""
    formato = pd.DataFrame()
    formato["ds"] = pd.to_datetime(tabla["fecha_objetivo"]).to_numpy()
    formato["y"] = tabla["nivel_objetivo"].to_numpy()
    for regresor in REGRESORES:
        formato[regresor] = tabla[regresor].to_numpy()
    return formato


def ejecutar_prophet(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str) -> dict:
    """Entrena y evalúa Prophet para una estación en todos los horizontes.

    Returns:
        Métricas de prueba por horizonte (ver ``evaluar_y_guardar``).
    """
    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === PASO 1: PREPARACIÓN DE DATOS ===
        # [JUSTIFICACIÓN]: Prophet no acepta matrices X / y. Exige un DataFrame con las
        # columnas "ds" (fecha) e "y" (valor), más una columna por regresor. Trabaja en
        # metros y estandariza los regresores internamente, por eso no se usa x_train.
        datos = preparar_datos(niveles, observados, estacion, horizonte)
        entrenamiento = tabla_prophet(datos["tabla_train"])

        # === PASO 2: DEFINICIÓN DEL MODELO ===
        modelo = Prophet(
            growth="flat",                  # sin tendencia: la media decenal es estable (PDF)
            yearly_seasonality=10,          # ciclo anual de crecida / estiaje (10 armónicos)
            weekly_seasonality=False,       # el río no depende del día de la semana
            daily_seasonality=False,        # tenemos un dato por día, no por hora
        )
        for regresor in REGRESORES:
            modelo.add_regressor(regresor)

        # === PASO 3: ENTRENAMIENTO ===
        # [JUSTIFICACIÓN]: Prophet no tiene early stopping; se entrena solo con train
        # (como el resto) para que validación quede libre para el ensamble.
        modelo.fit(entrenamiento)

        # === PASO 4: PREDICCIÓN (YA EN METROS) ===
        # [JUSTIFICACIÓN]: no hay que invertir ninguna escala porque Prophet se entrenó en metros.
        for conjunto in ["val", "test"]:
            tabla = datos["tabla_" + conjunto]
            prediccion = modelo.predict(tabla_prophet(tabla).drop(columns=["y"]))
            nivel_predicho = prediccion["yhat"].to_numpy()   # "yhat" es la predicción central
            resultado = armar_predicciones(tabla, nivel_predicho, estacion, "Prophet", horizonte)
            if conjunto == "val":
                predicciones_val.append(resultado)
            else:
                predicciones_test.append(resultado)

    return evaluar_y_guardar(estacion, "Prophet", pd.concat(predicciones_val), pd.concat(predicciones_test))


def ejecutar() -> None:
    """Entrena y evalúa Prophet en todas las estaciones objetivo."""
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_prophet(niveles, observados, estacion)


if __name__ == "__main__":
    ejecutar()
