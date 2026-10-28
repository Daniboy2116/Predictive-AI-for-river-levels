import pandas as pd
from sklearn.linear_model import LinearRegression

from src.configuracion import ESTACIONES_OBJETIVO, HORIZONTE_PRINCIPAL, HORIZONTES
from src.evaluate import cargar_predicciones, evaluar_y_guardar

NOMBRE_ENSAMBLE = "Ensamble"
# La persistencia aporta y(t) (cambio cero); la climatología no usa la situación actual
MODELOS_EXCLUIDOS = ["Persistencia", "Climatologia", "TimesFM", NOMBRE_ENSAMBLE]


def tabla_ancha(predicciones: pd.DataFrame, estacion: str, horizonte: int) -> pd.DataFrame:
    seleccion = predicciones[(predicciones["Estacion"] == estacion) & (predicciones["Horizonte"] == horizonte)]
    ancha = seleccion.pivot(index="Fecha", columns="Modelo", values="Prediccion")
    reales = seleccion.drop_duplicates("Fecha").set_index("Fecha")
    ancha["Real"] = reales["Real"]
    ancha["Observado"] = reales["Observado"].astype(bool)
    return ancha


def ejecutar_ensamble(estacion: str) -> dict:
    todas_val = cargar_predicciones("validacion")
    todas_test = cargar_predicciones("prueba")

    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        # === Preparación de datos ===
        ancha_val = tabla_ancha(todas_val, estacion, horizonte)
        ancha_test = tabla_ancha(todas_test, estacion, horizonte)
        modelos_base = [
            m for m in ancha_val.columns
            if m not in ["Real", "Observado"] and m not in MODELOS_EXCLUIDOS and m in ancha_test.columns
        ]

        ancha_test = ancha_test.dropna(subset=modelos_base + ["Persistencia"])
        nivel_hoy_val = ancha_val["Persistencia"]
        nivel_hoy_test = ancha_test["Persistencia"]
        x_val = ancha_val[modelos_base].sub(nivel_hoy_val, axis=0)
        y_val = ancha_val["Real"] - nivel_hoy_val
        x_test = ancha_test[modelos_base].sub(nivel_hoy_test, axis=0)

        # === Definición del modelo ===
        metamodelo = LinearRegression(positive=True, fit_intercept=False)

        # === Entrenamiento ===
        observados = ancha_val["Observado"]
        metamodelo.fit(x_val[observados], y_val[observados])
        if horizonte == HORIZONTE_PRINCIPAL:
            pesos = pd.Series(metamodelo.coef_, index=modelos_base).sort_values(ascending=False)
            pesos = pesos[pesos > 0.001].round(3)
            print(f"{estacion.ljust(15)} pesos del ensamble (h={horizonte}): {pesos.to_dict()}")

        # === Evaluación ===
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
    for estacion in ESTACIONES_OBJETIVO:
        ejecutar_ensamble(estacion)


if __name__ == "__main__":
    ejecutar()
