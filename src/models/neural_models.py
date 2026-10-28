import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.configuracion import ESTACIONES_OBJETIVO, HORIZONTES, SEMILLA
from src.evaluate import armar_predicciones, evaluar_y_guardar
from src.preprocessing import cambio_a_nivel, cargar_datos, preparar_datos, preparar_ventanas

LONGITUD_VENTANA = 60
MAXIMO_HORIZONTE = 15
EPOCAS_MAXIMAS = 150
TAMANO_LOTE = 64
TASA_APRENDIZAJE = 0.001
PACIENCIA = 10
DIAS_ATAJO = 31

NOMBRES_REDES = ["LSTM", "CNN1D", "Transformer"]


class RedLSTM(nn.Module):
    def __init__(self, numero_canales: int, numero_salidas: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=numero_canales, hidden_size=64, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.capa_densa = nn.Linear(64, 64)
        self.activacion = nn.ReLU()
        self.capa_salida = nn.Linear(64, numero_salidas)

    def forward(self, x):
        salidas_lstm, _ = self.lstm(x)
        ultimo_dia = salidas_lstm[:, -1, :]
        h = self.activacion(self.capa_densa(ultimo_dia))
        return self.capa_salida(h)


class BloqueConvolucionCausal(nn.Module):
    def __init__(self, canales_entrada: int, canales_salida: int, dilatacion: int):
        super().__init__()
        tamano_nucleo = 3
        relleno_izquierdo = (tamano_nucleo - 1) * dilatacion
        self.relleno = nn.ConstantPad1d((relleno_izquierdo, 0), 0.0)
        self.convolucion = nn.Conv1d(canales_entrada, canales_salida, kernel_size=tamano_nucleo,
                                     dilation=dilatacion)
        self.activacion = nn.ReLU()

    def forward(self, x):
        return self.activacion(self.convolucion(self.relleno(x)))


class RedCNN1D(nn.Module):
    def __init__(self, numero_canales: int, numero_salidas: int):
        super().__init__()
        self.bloque1 = BloqueConvolucionCausal(numero_canales, 32, dilatacion=1)
        self.bloque2 = BloqueConvolucionCausal(32, 32, dilatacion=2)
        self.pooling1 = nn.MaxPool1d(kernel_size=2)
        self.bloque3 = BloqueConvolucionCausal(32, 32, dilatacion=4)
        self.pooling2 = nn.MaxPool1d(kernel_size=2)
        self.capa_densa = nn.Linear(32 * 15, 64)
        self.activacion = nn.ReLU()
        self.capa_salida = nn.Linear(64, numero_salidas)

    def forward(self, x):
        h = x.permute(0, 2, 1)
        h = self.bloque1(h)
        h = self.bloque2(h)
        h = self.pooling1(h)
        h = self.bloque3(h)
        h = self.pooling2(h)
        h = h.reshape(h.shape[0], -1)
        h = self.activacion(self.capa_densa(h))
        return self.capa_salida(h)


class RedTransformer(nn.Module):
    def __init__(self, numero_canales: int, numero_salidas: int):
        super().__init__()
        dimension = 32
        self.embedding = nn.Linear(numero_canales, dimension)
        self.posicion = nn.Parameter(torch.zeros(1, LONGITUD_VENTANA, dimension))
        capa_encoder = nn.TransformerEncoderLayer(d_model=dimension, nhead=4, dim_feedforward=64,
                                                  dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(capa_encoder, num_layers=2, enable_nested_tensor=False)
        self.capa_densa = nn.Linear(dimension, 64)
        self.activacion = nn.ReLU()
        self.capa_salida = nn.Linear(64, numero_salidas)

    def forward(self, x):
        h = self.embedding(x) + self.posicion
        h = self.encoder(h)
        ultimo_dia = h[:, -1, :]
        h = self.activacion(self.capa_densa(ultimo_dia))
        return self.capa_salida(h)


def crear_red(nombre_red: str, numero_canales: int, numero_salidas: int) -> nn.Module:
    if nombre_red == "LSTM":
        return RedLSTM(numero_canales, numero_salidas)
    if nombre_red == "CNN1D":
        return RedCNN1D(numero_canales, numero_salidas)
    if nombre_red == "Transformer":
        return RedTransformer(numero_canales, numero_salidas)
    raise ValueError(f"Red desconocida: {nombre_red}")


class RedConAtajoLineal(nn.Module):
    def __init__(self, red: nn.Module, numero_canales: int, numero_salidas: int):
        super().__init__()
        self.red = red
        self.atajo = nn.Linear(DIAS_ATAJO * numero_canales, numero_salidas)
        nn.init.zeros_(self.red.capa_salida.weight)
        nn.init.zeros_(self.red.capa_salida.bias)

    def forward(self, x):
        dias_recientes = x[:, -DIAS_ATAJO:, :]
        entrada_atajo = dias_recientes.reshape(x.shape[0], -1)
        return self.atajo(entrada_atajo) + self.red(x)


def entrenar(modelo: nn.Module, parametros, x_train: torch.Tensor, y_train: torch.Tensor,
             x_val: torch.Tensor, y_val: torch.Tensor) -> tuple:
    funcion_perdida = nn.MSELoss()
    optimizador = torch.optim.Adam(parametros, lr=TASA_APRENDIZAJE)
    generador = torch.Generator().manual_seed(SEMILLA)
    numero_muestras = len(x_train)
    
    prediccion_inicial = torch.tensor(predecir_por_lotes(modelo, x_val))
    mejor_perdida_val = funcion_perdida(prediccion_inicial, y_val).item()
    mejores_pesos = copy.deepcopy(modelo.state_dict())
    epocas_sin_mejora = 0
    epocas_usadas = 0

    for epoca in range(EPOCAS_MAXIMAS):
        epocas_usadas = epoca + 1
        modelo.train()
        orden_aleatorio = torch.randperm(numero_muestras, generator=generador)
        for inicio in range(0, numero_muestras, TAMANO_LOTE):
            indices_lote = orden_aleatorio[inicio:inicio + TAMANO_LOTE]
            optimizador.zero_grad()
            prediccion_lote = modelo(x_train[indices_lote])
            perdida = funcion_perdida(prediccion_lote, y_train[indices_lote])
            perdida.backward()
            optimizador.step()

        prediccion_val = torch.tensor(predecir_por_lotes(modelo, x_val))
        perdida_val = funcion_perdida(prediccion_val, y_val).item()

        if perdida_val < mejor_perdida_val:
            mejor_perdida_val = perdida_val
            mejores_pesos = copy.deepcopy(modelo.state_dict())
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= PACIENCIA:
                break

    modelo.load_state_dict(mejores_pesos)
    return epocas_usadas, mejor_perdida_val


def predecir_por_lotes(modelo: nn.Module, ventanas: torch.Tensor) -> np.ndarray:
    modelo.eval()
    partes = []
    with torch.no_grad():
        for inicio in range(0, len(ventanas), 1024):
            partes.append(modelo(ventanas[inicio:inicio + 1024]).numpy())
    return np.concatenate(partes)


def ejecutar_red_neuronal(niveles: pd.DataFrame, observados: pd.DataFrame, estacion: str,
                          nombre_red: str) -> dict:
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    # === Preparación de datos ===
    datos = preparar_ventanas(niveles, observados, estacion, LONGITUD_VENTANA, MAXIMO_HORIZONTE)
    ventanas = torch.tensor(datos["ventanas"])
    objetivos = torch.tensor(datos["cambios_escalados"])
    x_train = ventanas[datos["en_train"]]
    y_train = objetivos[datos["en_train"]]
    x_val = ventanas[datos["en_val"]]
    y_val = objetivos[datos["en_val"]]

    # === Definición del modelo ===
    red = crear_red(nombre_red, ventanas.shape[2], MAXIMO_HORIZONTE)
    modelo = RedConAtajoLineal(red, ventanas.shape[2], MAXIMO_HORIZONTE)

    # === Entrenamiento ===
    for parametro in modelo.red.parameters():
        parametro.requires_grad = False
    epocas_lineal, perdida_lineal = entrenar(modelo, modelo.atajo.parameters(), x_train, y_train, x_val, y_val)

    for parametro in modelo.atajo.parameters():
        parametro.requires_grad = False
    for parametro in modelo.red.parameters():
        parametro.requires_grad = True
    epocas_red, perdida_red = entrenar(modelo, modelo.red.parameters(), x_train, y_train, x_val, y_val)
    print(f"{estacion.ljust(15)} {nombre_red.ljust(12)} MSE val: atajo lineal {perdida_lineal:.4f} ({epocas_lineal} épocas) -> red {perdida_red:.4f} ({epocas_red} épocas)", flush=True)

    # === Evaluación ===
    cambios_escalados = predecir_por_lotes(modelo, ventanas)
    cambios_metros = datos["escalador_objetivo"].inverse_transform(cambios_escalados)
    cambios_metros = pd.DataFrame(cambios_metros, index=datos["origenes"], columns=range(1, MAXIMO_HORIZONTE + 1))

    predicciones_val = []
    predicciones_test = []
    for horizonte in HORIZONTES:
        tablas = preparar_datos(niveles, observados, estacion, horizonte)
        for conjunto in ["val", "test", "futuro"]:
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
    niveles, observados = cargar_datos()
    for estacion in ESTACIONES_OBJETIVO:
        for nombre_red in NOMBRES_REDES:
            ejecutar_red_neuronal(niveles, observados, estacion, nombre_red)


if __name__ == "__main__":
    ejecutar()
