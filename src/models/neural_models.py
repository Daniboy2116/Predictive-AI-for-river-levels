"""Redes neuronales en PyTorch para predecir el nivel del día siguiente (t+1).

Arquitecturas:
- LSTM: red recurrente que "recuerda" lo ocurrido en los días anteriores.
- CNN 1D: filtros convolucionales que detectan patrones locales (Conv1D + MaxPooling1D).
- Transformer: mecanismo de self-attention que pondera qué días pasados importan más.

Las tres reciben ventanas de 7 días con forma (muestras, 7 pasos, 1 variable) y
devuelven un único valor: el nivel escalado del día siguiente.

Ejecutar:  python -m src.models.neural_models
"""
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.evaluate import evaluar_y_guardar
from src.preprocessing import (
    ESTACIONES, NUMERO_DE_LAGS, SEMILLA, cargar_datos, invertir_escala, preparar_estacion,
)

# Hiperparámetros de entrenamiento comunes a las tres redes
EPOCAS_MAXIMAS = 200     # vueltas completas máximas sobre los datos de entrenamiento
TAMANO_LOTE = 32         # cuántas muestras se usan en cada actualización de pesos
TASA_APRENDIZAJE = 0.001
PACIENCIA = 20           # épocas sin mejorar en validación antes de detenerse

NOMBRES_REDES = ["LSTM", "CNN1D", "Transformer"]


# ==========================================================================
# ARQUITECTURAS
# [JUSTIFICACIÓN]: en PyTorch cada red se define como una clase con dos partes:
#   __init__ -> qué capas tiene la red.
#   forward  -> en qué orden pasan los datos por esas capas.
# ==========================================================================
class RedLSTM(nn.Module):
    """LSTM de 2 capas seguida de capas densas."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.capa_densa = nn.Linear(64, 32)
        self.activacion = nn.ReLU()
        self.capa_salida = nn.Linear(32, 1)

    def forward(self, x):
        # x: (lote, 7 pasos, 1)
        salidas_lstm, _ = self.lstm(x)              # (lote, 7, 64)
        ultimo_paso = salidas_lstm[:, -1, :]        # nos quedamos con el último día: (lote, 64)
        h = self.activacion(self.capa_densa(ultimo_paso))
        y = self.capa_salida(h)                     # (lote, 1)
        return y.view(-1)                           # (lote,)


class RedCNN1D(nn.Module):
    """Dos capas Conv1D + MaxPooling1D seguidas de capas densas."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool1d(kernel_size=2)  # 7 pasos -> 3 pasos
        self.activacion = nn.ReLU()
        self.capa_densa = nn.Linear(64 * 3, 32)
        self.capa_salida = nn.Linear(32, 1)

    def forward(self, x):
        # Conv1d espera (lote, canales, pasos), así que intercambiamos los ejes 1 y 2
        x = x.permute(0, 2, 1)                      # (lote, 1, 7)
        h = self.activacion(self.conv1(x))          # (lote, 64, 7)
        h = self.activacion(self.conv2(h))          # (lote, 64, 7)
        h = self.pooling(h)                         # (lote, 64, 3)
        h = h.reshape(h.shape[0], -1)               # aplanar: (lote, 192)
        h = self.activacion(self.capa_densa(h))
        y = self.capa_salida(h)
        return y.view(-1)


class RedTransformer(nn.Module):
    """Encoder Transformer (self-attention) con posición aprendida para cada día."""

    def __init__(self):
        super().__init__()
        dimension = 32
        self.embedding = nn.Linear(1, dimension)    # cada nivel -> vector de 32 números
        # Un vector aprendido por posición para que la red sepa qué día es cada dato
        self.posicion = nn.Parameter(torch.zeros(1, NUMERO_DE_LAGS, dimension))
        capa_encoder = nn.TransformerEncoderLayer(d_model=dimension, nhead=4,
                                                  dim_feedforward=64, dropout=0.1,
                                                  batch_first=True)
        self.encoder = nn.TransformerEncoder(capa_encoder, num_layers=2)
        self.capa_densa = nn.Linear(dimension, 32)
        self.activacion = nn.ReLU()
        self.capa_salida = nn.Linear(32, 1)

    def forward(self, x):
        h = self.embedding(x) + self.posicion       # (lote, 7, 32)
        h = self.encoder(h)                         # (lote, 7, 32)
        ultimo_paso = h[:, -1, :]                   # (lote, 32)
        h = self.activacion(self.capa_densa(ultimo_paso))
        y = self.capa_salida(h)
        return y.view(-1)


def lags_a_tensor(x_lags: pd.DataFrame) -> torch.Tensor:
    """Convierte la tabla de lags en un tensor (muestras, 7 pasos, 1) en orden cronológico.

    Las columnas vienen como lag_1 (ayer), lag_2, ..., lag_7 (hace una semana).
    Las redes secuenciales deben leer del pasado al presente: lag_7, ..., lag_1.
    """
    columnas_cronologicas = []
    for k in range(NUMERO_DE_LAGS, 0, -1):
        columnas_cronologicas.append("lag_" + str(k))

    matriz = x_lags[columnas_cronologicas].to_numpy(dtype=np.float32)
    matriz = matriz.reshape(len(matriz), NUMERO_DE_LAGS, 1)
    return torch.tensor(matriz)


def ejecutar_red_neuronal(df: pd.DataFrame, estacion: str, nombre_red: str) -> dict:
    """Entrena y evalúa una red neuronal (LSTM, CNN1D o Transformer) para una estación.

    Args:
        df: DataFrame completo devuelto por ``cargar_datos``.
        estacion: nombre de la estación.
        nombre_red: ``"LSTM"``, ``"CNN1D"`` o ``"Transformer"``.

    Returns:
        Diccionario con las métricas (RMSE, MAE, R2) en metros.
    """
    # Semillas fijas: misma inicialización de pesos y mismo orden de lotes en cada ejecución
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    # === PASO 1: PREPARACIÓN DE DATOS ===
    # [JUSTIFICACIÓN]: PyTorch trabaja con tensores, no con DataFrames. Además las redes
    # secuenciales esperan 3 dimensiones: (muestras, pasos de tiempo, variables).
    datos = preparar_estacion(df, estacion)
    x_train = lags_a_tensor(datos["x_train"])
    y_train = torch.tensor(datos["y_train"].to_numpy(dtype=np.float32))
    x_val = lags_a_tensor(datos["x_val"])
    y_val = torch.tensor(datos["y_val"].to_numpy(dtype=np.float32))
    x_test = lags_a_tensor(datos["x_test"])

    # === PASO 2: DEFINICIÓN DEL MODELO ===
    if nombre_red == "LSTM":
        modelo = RedLSTM()
    elif nombre_red == "CNN1D":
        modelo = RedCNN1D()
    elif nombre_red == "Transformer":
        modelo = RedTransformer()
    else:
        raise ValueError("Red desconocida: " + nombre_red)

    funcion_perdida = nn.MSELoss()   # error cuadrático medio
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)

    # === PASO 3: ENTRENAMIENTO ===
    # [JUSTIFICACIÓN]: a diferencia de scikit-learn (un simple .fit()), en PyTorch el
    # entrenamiento se escribe a mano con un bucle de épocas:
    #   1. Se recorren los datos de train en lotes pequeños y se actualizan los pesos.
    #   2. Al final de cada época se mide el error en validación.
    #   3. Se guardan los mejores pesos y se para si no mejora en PACIENCIA épocas
    #      ("early stopping").
    # Barajamos el orden de las VENTANAS (no la serie): cada ventana ya contiene sus
    # 7 días en orden, así que no se rompe la cronología ni hay fuga de información.
    generador = torch.Generator().manual_seed(SEMILLA)
    numero_muestras = len(x_train)
    mejor_perdida_val = float("inf")
    mejores_pesos = copy.deepcopy(modelo.state_dict())
    epocas_sin_mejora = 0

    for epoca in range(EPOCAS_MAXIMAS):
        # 3.1) Entrenar una época completa, lote a lote
        modelo.train()
        orden_aleatorio = torch.randperm(numero_muestras, generator=generador)
        for inicio in range(0, numero_muestras, TAMANO_LOTE):
            indices_lote = orden_aleatorio[inicio:inicio + TAMANO_LOTE]
            x_lote = x_train[indices_lote]
            y_lote = y_train[indices_lote]

            optimizador.zero_grad()                  # borrar gradientes anteriores
            y_pred_lote = modelo(x_lote)             # predicción
            perdida = funcion_perdida(y_pred_lote, y_lote)
            perdida.backward()                       # calcular gradientes
            optimizador.step()                       # actualizar pesos

        # 3.2) Medir el error en validación (sin actualizar pesos)
        modelo.eval()
        with torch.no_grad():
            perdida_val = funcion_perdida(modelo(x_val), y_val).item()

        # 3.3) Early stopping
        if perdida_val < mejor_perdida_val:
            mejor_perdida_val = perdida_val
            mejores_pesos = copy.deepcopy(modelo.state_dict())
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora = epocas_sin_mejora + 1
            if epocas_sin_mejora >= PACIENCIA:
                break

    modelo.load_state_dict(mejores_pesos)   # nos quedamos con la mejor época

    # === PASO 4: PREDICCIÓN E INVERSIÓN DE ESCALA ===
    modelo.eval()
    with torch.no_grad():
        y_pred_escalado = modelo(x_test).numpy()
    y_pred = invertir_escala(datos["escalador"], y_pred_escalado)
    y_real = datos["y_test_metros"]

    metricas = evaluar_y_guardar(estacion, nombre_red, datos["fechas_test"], y_real, y_pred)
    return metricas


def ejecutar() -> None:
    """Entrena y evalúa las tres redes en todas las estaciones."""
    df = cargar_datos()
    for estacion in ESTACIONES:
        for nombre_red in NOMBRES_REDES:
            ejecutar_red_neuronal(df, estacion, nombre_red)


if __name__ == "__main__":
    ejecutar()
