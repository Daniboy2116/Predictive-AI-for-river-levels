"""Ensamble apilado (stacking, Wolpert 1992): un metamodelo combina a los demás.

Idea: cada modelo se equivoca de forma distinta. Si combinamos sus pronósticos con
buenos pesos, los errores se compensan en parte.

- Los modelos base se entrenaron SOLO con train (1974-2014).
- El metamodelo aprende los pesos con sus predicciones de VALIDACIÓN (2015-2019),
  datos que ningún modelo base usó para ajustar sus parámetros.
- La evaluación final es en PRUEBA (2020-2025), igual que el resto.

Metamodelo: cambio = Σ w_m · cambio_m, con pesos w_m ≥ 0 (regresión lineal con
pesos positivos y sin intercepto), donde cambio_m es lo que el modelo m pronostica
que subirá o bajará el río respecto a hoy.

Requisito: haber ejecutado antes los modelos base (``python main.py`` lo hace en orden).

Ejecutar:  python -m src.models.ensamble
"""
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.configuracion import ESTACIONES_OBJETIVO, HORIZONTE_PRINCIPAL, HORIZONTES
from src.evaluate import cargar_predicciones, evaluar_y_guardar

NOMBRE_ENSAMBLE = "Ensamble"
# La persistencia aporta y(t) (cambio cero); la climatología no usa la situación actual
MODELOS_EXCLUIDOS = ["Persistencia", "Climatologia", NOMBRE_ENSAMBLE]


def tabla_ancha(predicciones: pd.DataFrame, estacion: str, horizonte: int) -> pd.DataFrame:
    """Una fila por fecha objetivo y una columna por modelo (más ``Real`` y ``Observado``)."""
    seleccion = predicciones[(predicciones["Estacion"] == estacion) & (predicciones["Horizonte"] == horizonte)]
    ancha = seleccion.pivot(index="Fecha", columns="Modelo", values="Prediccion")
    reales = seleccion.drop_duplicates("Fecha").set_index("Fecha")
    ancha["Real"] = reales["Real"]
    ancha["Observado"] = reales["Observado"].astype(bool)
    return ancha


def ejecutar_ensamble(estacion: str) -> dict:
    """Ajusta el metamodelo de una estación (un juego de pesos por horizonte) y lo evalúa.

    Returns:
        Métricas de prueba por horizonte (ver ``evaluar_y_guardar``).
    """
    todas_val = cargar_predicciones("validacion")
    todas_test = cargar_predicciones("prueba")

    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === PASO 1: PREPARACIÓN DE DATOS ===
        # [JUSTIFICACIÓN]: las "variables" del metamodelo son las predicciones de los
        # modelos base, expresadas como cambio respecto al nivel de hoy (persistencia).
        ancha_val = tabla_ancha(todas_val, estacion, horizonte)
        ancha_test = tabla_ancha(todas_test, estacion, horizonte)
        modelos_base = []
        for modelo in ancha_val.columns:
            es_modelo = modelo not in ["Real", "Observado"] and modelo not in MODELOS_EXCLUIDOS
            if es_modelo and modelo in ancha_test.columns:
                modelos_base.append(modelo)

        nivel_hoy_val = ancha_val["Persistencia"]
        nivel_hoy_test = ancha_test["Persistencia"]
        x_val = ancha_val[modelos_base].sub(nivel_hoy_val, axis=0)
        y_val = ancha_val["Real"] - nivel_hoy_val
        x_test = ancha_test[modelos_base].sub(nivel_hoy_test, axis=0)

        # === PASO 2: DEFINICIÓN DEL MODELO ===
        metamodelo = LinearRegression(positive=True, fit_intercept=False)

        # === PASO 3: ENTRENAMIENTO ===
        # [JUSTIFICACIÓN]: se entrena con VALIDACIÓN (no con train): en train los modelos
        # base ya vieron los datos y parecerían mejores de lo que son.
        observados = ancha_val["Observado"]
        metamodelo.fit(x_val[observados], y_val[observados])
        if horizonte == HORIZONTE_PRINCIPAL:
            pesos = pd.Series(metamodelo.coef_, index=modelos_base).sort_values(ascending=False)
            pesos = pesos[pesos > 0.001].round(3)
            print(estacion.ljust(15), "pesos del ensamble (h=" + str(horizonte) + "):", pesos.to_dict())

        # === PASO 4: PREDICCIÓN (YA EN METROS) ===
        for conjunto, ancha, x, nivel_hoy in [("val", ancha_val, x_val, nivel_hoy_val),
                                              ("test", ancha_test, x_test, nivel_hoy_test)]:
            nivel_predicho = nivel_hoy.to_numpy() + metamodelo.predict(x)
            resultado = pd.DataFrame({
                "Fecha": ancha.index,
                "Estacion": estacion,
                "Modelo": NOMBRE_ENSAMBLE,
                "Horizonte": horizonte,
                "Real": ancha["Real"].to_numpy(),
                "Prediccion": nivel_predicho,
                "Observado": ancha["Observado"].to_numpy(),
            })
            if conjunto == "val":
                predicciones_val.append(resultado)
            else:
                predicciones_test.append(resultado)

    return evaluar_y_guardar(estacion, NOMBRE_ENSAMBLE, pd.concat(predicciones_val), pd.concat(predicciones_test))


def ejecutar() -> None:
    """Ajusta y evalúa el ensamble en todas las estaciones objetivo."""
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_ensamble(estacion)


if __name__ == "__main__":
    ejecutar()
