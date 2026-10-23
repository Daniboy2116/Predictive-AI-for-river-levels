"""Redes neuronales en PyTorch que pronostican los próximos 15 días de una sola vez.

Arquitecturas:
- LSTM: red recurrente que lee la ventana día a día y "recuerda" lo importante.
- CNN 1D: convoluciones causales dilatadas (Bai et al., 2018) + MaxPooling1D: cada capa
  mira más lejos en el pasado (1, 2, 4 días de separación) con pocos parámetros.
- Transformer: self-attention; cada día de la ventana decide a qué otros días atender.

Las tres reciben ventanas de 60 días con 10 canales (nivel y anomalía de las cuatro
estaciones + calendario) y devuelven 15 valores: el cambio escalado del nivel a
1, 2, ..., 15 días.

Atajo lineal y entrenamiento en dos etapas: cada red va acompañada de un "atajo"
lineal (una capa Linear sobre los últimos 31 días, equivalente a un Ridge). Primero
se entrena solo el atajo; después se congela y la red aprende únicamente lo que el
modelo lineal no explica. Si no hay nada no lineal que aprender, la red se queda
cerca de cero y el resultado no empeora respecto al modelo lineal.

Ejecutar:  python -m src.models.neural_models
"""
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.configuracion import ESTACIONES_OBJETIVO, HORIZONTES, SEMILLA
from src.evaluate import armar_predicciones, evaluar_y_guardar
from src.preprocessing import cambio_a_nivel, cargar_datos, preparar_datos, preparar_ventanas

# Hiperparámetros comunes a las tres redes
LONGITUD_VENTANA = 60    # días de historia que ve la red
MAXIMO_HORIZONTE = 15    # la red predice los cambios a 1..15 días
EPOCAS_MAXIMAS = 150     # vueltas completas máximas sobre los datos de entrenamiento
TAMANO_LOTE = 64         # cuántas muestras se usan en cada actualización de pesos
TASA_APRENDIZAJE = 0.001
PACIENCIA = 10           # épocas sin mejorar en validación antes de detenerse
DIAS_ATAJO = 31          # días recientes que ve el atajo lineal

NOMBRES_REDES = ["LSTM", "CNN1D", "Transformer"]


# ==========================================================================
# ARQUITECTURAS
# [JUSTIFICACIÓN]: en PyTorch cada red se define como una clase con dos partes:
#   __init__ -> qué capas tiene la red.
#   forward  -> en qué orden pasan los datos por esas capas.
# ==========================================================================
class RedLSTM(nn.Module):
    """LSTM de 2 capas seguida de capas densas."""

    def __init__(self, numero_canales: int, numero_salidas: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=numero_canales, hidden_size=64, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.capa_densa = nn.Linear(64, 64)
        self.activacion = nn.ReLU()
        self.capa_salida = nn.Linear(64, numero_salidas)

    def forward(self, x):
        # x: (lote, 60 días, 10 canales)
        salidas_lstm, _ = self.lstm(x)              # (lote, 60, 64)
        ultimo_dia = salidas_lstm[:, -1, :]         # estado tras leer el día t: (lote, 64)
        h = self.activacion(self.capa_densa(ultimo_dia))
        return self.capa_salida(h)                  # (lote, 15)


class BloqueConvolucionCausal(nn.Module):
    """Conv1D causal dilatada: la salida del día d solo mira los días d, d - dil, d - 2·dil."""

    def __init__(self, canales_entrada: int, canales_salida: int, dilatacion: int):
        super().__init__()
        tamano_nucleo = 3
        relleno_izquierdo = (tamano_nucleo - 1) * dilatacion
        self.relleno = nn.ConstantPad1d((relleno_izquierdo, 0), 0.0)   # solo a la izquierda (pasado)
        self.convolucion = nn.Conv1d(canales_entrada, canales_salida, kernel_size=tamano_nucleo,
                                     dilation=dilatacion)
        self.activacion = nn.ReLU()

    def forward(self, x):
        return self.activacion(self.convolucion(self.relleno(x)))


class RedCNN1D(nn.Module):
    """Tres convoluciones causales dilatadas (1, 2, 4) con MaxPooling1D y capas densas."""

    def __init__(self, numero_canales: int, numero_salidas: int):
        super().__init__()
        self.bloque1 = BloqueConvolucionCausal(numero_canales, 32, dilatacion=1)
        self.bloque2 = BloqueConvolucionCausal(32, 32, dilatacion=2)
        self.pooling1 = nn.MaxPool1d(kernel_size=2)          # 60 días -> 30
        self.bloque3 = BloqueConvolucionCausal(32, 32, dilatacion=4)
        self.pooling2 = nn.MaxPool1d(kernel_size=2)          # 30 -> 15
        self.capa_densa = nn.Linear(32 * 15, 64)
        self.activacion = nn.ReLU()
        self.capa_salida = nn.Linear(64, numero_salidas)

    def forward(self, x):
        # Conv1d espera (lote, canales, días), así que intercambiamos los ejes 1 y 2
        h = x.permute(0, 2, 1)                      # (lote, 10, 60)
        h = self.bloque1(h)                         # (lote, 32, 60)
        h = self.bloque2(h)                         # (lote, 32, 60)
        h = self.pooling1(h)                        # (lote, 32, 30)
        h = self.bloque3(h)                         # (lote, 32, 30)
        h = self.pooling2(h)                        # (lote, 32, 15)
        h = h.reshape(h.shape[0], -1)               # aplanar: (lote, 480)
        h = self.activacion(self.capa_densa(h))
        return self.capa_salida(h)                  # (lote, 15)


class RedTransformer(nn.Module):
    """Encoder Transformer (self-attention) con una posición aprendida para cada día."""

    def __init__(self, numero_canales: int, numero_salidas: int):
        super().__init__()
        dimension = 32
        self.embedding = nn.Linear(numero_canales, dimension)   # cada día -> vector de 32 números
        # Un vector aprendido por posición para que la red sepa qué día es cada dato
        self.posicion = nn.Parameter(torch.zeros(1, LONGITUD_VENTANA, dimension))
        capa_encoder = nn.TransformerEncoderLayer(d_model=dimension, nhead=4, dim_feedforward=64,
                                                  dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(capa_encoder, num_layers=2, enable_nested_tensor=False)
        self.capa_densa = nn.Linear(dimension, 64)
        self.activacion = nn.ReLU()
        self.capa_salida = nn.Linear(64, numero_salidas)

    def forward(self, x):
        h = self.embedding(x) + self.posicion       # (lote, 60, 32)
        h = self.encoder(h)                         # (lote, 60, 32)
        ultimo_dia = h[:, -1, :]                    # representación del día t: (lote, 32)
        h = self.activacion(self.capa_densa(ultimo_dia))
        return self.capa_salida(h)                  # (lote, 15)


def crear_red(nombre_red: str, numero_canales: int, numero_salidas: int) -> nn.Module:
    """Devuelve una red nueva según su nombre."""
    if nombre_red == "LSTM":
        return RedLSTM(numero_canales, numero_salidas)
    if nombre_red == "CNN1D":
        return RedCNN1D(numero_canales, numero_salidas)
    if nombre_red == "Transformer":
        return RedTransformer(numero_canales, numero_salidas)
    raise ValueError("Red desconocida: " + nombre_red)


class RedConAtajoLineal(nn.Module):
    """Suma de un atajo lineal (últimos ``DIAS_ATAJO`` días) y una red no lineal."""

    def __init__(self, red: nn.Module, numero_canales: int, numero_salidas: int):
        super().__init__()
        self.red = red
        self.atajo = nn.Linear(DIAS_ATAJO * numero_canales, numero_salidas)
        # La red empieza respondiendo cero: al principio el modelo es solo el atajo lineal
        nn.init.zeros_(self.red.capa_salida.weight)
        nn.init.zeros_(self.red.capa_salida.bias)

    def forward(self, x):
        dias_recientes = x[:, -DIAS_ATAJO:, :]                       # (lote, 31, 10)
        entrada_atajo = dias_recientes.reshape(x.shape[0], -1)      # aplanar: (lote, 310)
        return self.atajo(entrada_atajo) + self.red(x)              # (lote, 15)


def entrenar(modelo: nn.Module, parametros, x_train: torch.Tensor, y_train: torch.Tensor,
             x_val: torch.Tensor, y_val: torch.Tensor) -> tuple:
    """Bucle de entrenamiento con early stopping; solo se actualizan ``parametros``.

    Returns:
        Tupla ``(épocas usadas, mejor pérdida de validación)``. Al terminar, el modelo
        queda con los pesos de la mejor época (o los iniciales, si ninguna época mejora).
    """
    funcion_perdida = nn.MSELoss()   # error cuadrático medio (sobre los 15 horizontes)
    optimizador = torch.optim.Adam(parametros, lr=TASA_APRENDIZAJE)
    generador = torch.Generator().manual_seed(SEMILLA)
    numero_muestras = len(x_train)
    # Punto de partida: si ninguna época mejora la validación, el modelo queda como estaba
    prediccion_inicial = torch.tensor(predecir_por_lotes(modelo, x_val))
    mejor_perdida_val = funcion_perdida(prediccion_inicial, y_val).item()
    mejores_pesos = copy.deepcopy(modelo.state_dict())
    epocas_sin_mejora = 0
    epocas_usadas = 0

    for epoca in range(EPOCAS_MAXIMAS):
        epocas_usadas = epoca + 1
        # 1) Entrenar una época completa, lote a lote
        modelo.train()
        orden_aleatorio = torch.randperm(numero_muestras, generator=generador)
        for inicio in range(0, numero_muestras, TAMANO_LOTE):
            indices_lote = orden_aleatorio[inicio:inicio + TAMANO_LOTE]
            optimizador.zero_grad()                              # borrar gradientes anteriores
            prediccion_lote = modelo(x_train[indices_lote])      # predicción
            perdida = funcion_perdida(prediccion_lote, y_train[indices_lote])
            perdida.backward()                                   # calcular gradientes
            optimizador.step()                                   # actualizar pesos

        # 2) Medir el error en validación (sin actualizar pesos)
        prediccion_val = torch.tensor(predecir_por_lotes(modelo, x_val))
        perdida_val = funcion_perdida(prediccion_val, y_val).item()

        # 3) Early stopping
        if perdida_val < mejor_perdida_val:
            mejor_perdida_val = perdida_val
            mejores_pesos = copy.deepcopy(modelo.state_dict())
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora = epocas_sin_mejora + 1
            if epocas_sin_mejora >= PACIENCIA:
                break

    modelo.load_state_dict(mejores_pesos)   # nos quedamos con la mejor época
    return epocas_usadas, mejor_perdida_val


def predecir_por_lotes(modelo: nn.Module, ventanas: torch.Tensor) -> np.ndarray:
    """Predice en lotes grandes (sin calcular gradientes) para no agotar la memoria."""
    modelo.eval()
    partes = []
    with torch.no_grad():
        for inicio in range(0, len(ventanas), 1024):
            partes.append(modelo(ventanas[inicio:inicio + 1024]).numpy())
    return np.concatenate(partes)


def ejecutar_red_neuronal(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str,
                          nombre_red: str) -> dict:
    """Entrena y evalúa una red (LSTM, CNN1D o Transformer) para una estación.

    Returns:
        Métricas de prueba por horizonte (ver ``evaluar_y_guardar``).
    """
    # Semillas fijas: misma inicialización de pesos y mismo orden de lotes en cada ejecución
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    # === PASO 1: PREPARACIÓN DE DATOS ===
    # [JUSTIFICACIÓN]: las redes secuenciales no usan la tabla de 79 variables, sino la
    # historia "en bruto": tensores de 3 dimensiones (muestras, días, canales). Y en vez
    # de un modelo por horizonte, una sola red predice los 15 cambios a la vez.
    datos = preparar_ventanas(niveles, observados, estacion, LONGITUD_VENTANA, MAXIMO_HORIZONTE)
    ventanas = torch.tensor(datos["ventanas"])
    objetivos = torch.tensor(datos["cambios_escalados"])
    x_train = ventanas[datos["en_train"]]
    y_train = objetivos[datos["en_train"]]
    x_val = ventanas[datos["en_val"]]
    y_val = objetivos[datos["en_val"]]

    # === PASO 2: DEFINICIÓN DEL MODELO ===
    red = crear_red(nombre_red, ventanas.shape[2], MAXIMO_HORIZONTE)
    modelo = RedConAtajoLineal(red, ventanas.shape[2], MAXIMO_HORIZONTE)

    # === PASO 3: ENTRENAMIENTO ===
    # [JUSTIFICACIÓN]: a diferencia de scikit-learn (un simple .fit()), en PyTorch el
    # entrenamiento se escribe a mano (función entrenar): épocas de lotes barajados,
    # error de validación al final de cada época y early stopping. Barajamos el orden de
    # las VENTANAS (no la serie): cada ventana ya contiene sus 60 días en orden, así que
    # no se rompe la cronología ni hay fuga de información.
    # [JUSTIFICACIÓN]: se entrena en dos etapas. Si se entrenan juntas, la parte no lineal
    # aprende más rápido, se sobreajusta y el early stopping corta antes de que la parte
    # lineal (la más útil) termine de aprender.
    # Etapa 1: solo el atajo lineal (la red está congelada y responde cero)
    for parametro in modelo.red.parameters():
        parametro.requires_grad = False
    epocas_lineal, perdida_lineal = entrenar(modelo, modelo.atajo.parameters(), x_train, y_train, x_val, y_val)
    # Etapa 2: se congela el atajo y la red aprende lo que el modelo lineal no explica
    for parametro in modelo.atajo.parameters():
        parametro.requires_grad = False
    for parametro in modelo.red.parameters():
        parametro.requires_grad = True
    epocas_red, perdida_red = entrenar(modelo, modelo.red.parameters(), x_train, y_train, x_val, y_val)
    print(estacion.ljust(15), nombre_red.ljust(12),
          "MSE validación (escalado): atajo lineal %.4f (%d épocas) -> con la red %.4f (%d épocas)"
          % (perdida_lineal, epocas_lineal, perdida_red, epocas_red), flush=True)

    # === PASO 4: PREDICCIÓN E INVERSIÓN DE ESCALA ===
    # Se predice para todos los días de emisión; luego cada horizonte toma su columna.
    cambios_escalados = predecir_por_lotes(modelo, ventanas)
    cambios_metros = datos["escalador_objetivo"].inverse_transform(cambios_escalados)
    cambios_metros = pd.DataFrame(cambios_metros, index=datos["origenes"], columns=range(1, MAXIMO_HORIZONTE + 1))

    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        tablas = preparar_datos(niveles, observados, estacion, horizonte)
        for conjunto in ["val", "test"]:
            tabla = tablas["tabla_" + conjunto]
            cambio_predicho = cambios_metros.loc[tabla.index, horizonte].to_numpy()
            nivel_predicho = cambio_a_nivel(tabla["nivel_actual"], cambio_predicho)
            resultado = armar_predicciones(tabla, nivel_predicho, estacion, nombre_red, horizonte)
            if conjunto == "val":
                predicciones_val.append(resultado)
            else:
                predicciones_test.append(resultado)

    return evaluar_y_guardar(estacion, nombre_red, pd.concat(predicciones_val), pd.concat(predicciones_test))


def ejecutar() -> None:
    """Entrena y evalúa las tres redes en todas las estaciones objetivo."""
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        for nombre_red in NOMBRES_REDES:
            ejecutar_red_neuronal(niveles, observados, estacion, nombre_red)


if __name__ == "__main__":
    ejecutar()
