# Auditoría de datos — niveles diarios del río Orinoco

**Fecha:** 28/10/2026  
**Datos:** 11 archivos en `data/` (CSV y XLSX), 4 estaciones: Puerto Ayacucho, Caicara, Ciudad Bolívar y Palúa.  
**Entregables:** este informe, `salida/dataset_real.csv` y los scripts de `scripts/` (todos ejecutados; todas las cifras salen de ellos).

---

## Resumen ejecutivo

1. **La fuente original es el libro `DATOS RIO ORINOCO.xlsx`**, hoja por estación. Los cuatro CSV "raw" y el XLSX `ayacucho_raw.xlsx` son extractos exactos de esas hojas para 1974-2025: 100 % de los valores coinciden en las fechas comunes (diferencias máximas de 0,005 m por redondeo).
2. **No hay señales de cm/m, columnas cruzadas ni offsets globales.** Se compararon 116 pares archivo-estación: en todos, el mejor desfase global es 0 días, el cociente de medianas es 1,00 y no hay ningún candidato a factor ×100. La única anomalía sistemática es un **desfase local de 1 día en Ciudad Bolívar dentro de `DATOS:Entrenar`**, entre 2020-01-01 y 2021-12-30 (638 fechas), donde `Entrenar(t) = raw(t+1)`.
3. **Los archivos "imputados" no son datos reales.** El imputado de Ayacucho rellena 700 fechas: el hueco completo 2007-01-01..2008-11-30. El bloque de 700 días es sintético (media 46,46 m, sin saltos >0,5 m, máxima racha constante de 2 días). Los tres archivos imputados son idénticos entre sí. Además, dejan sin rellenar 17 fechas de Palúa en la ventana 1974-2025 (12 días 29-feb y 5 días de 1993), así que ni siquiera están completos.
4. **Control de calidad:** 86 saltos >0,5 m (Ayacucho 19, Caicara 13, Ciudad Bolívar 17, Palúa 37). Se identificaron **8 pulsos compatibles con aperturas de compuertas (a)** en Palúa, **28 días aberrantes (b)** que se pusieron a NaN, y **8 eventos de escalón (c)**, incluidos los dos cambios de referencia conocidos de Ciudad Bolívar (**1987-05-15, −2,20 m** y **2019-01-01, +1,00 m**, ambos exactos). Ningún escalón se comporta como un offset permanente: la serie los reabsorbe en semanas o, como máximo, uno o dos años, por lo que se reportan pero **no se corrigen**.
5. **Cobertura:** el mejor bloque continuo reciente es **2009-2024**: 16/16 años completos en las 4 estaciones, con cobertura mínima de 99,949 % (solo faltan 3 días 29-feb de Palúa: 2012, 2016 y 2020; Ayacucho está completo). Se propone **train 2009-2018, validación 2019-2021, prueba 2022-2024**. El cambio de referencia de 2019 y el candidato de Ayacucho 2020-06-20 quedan aislados en validación, y el entrenamiento queda limpio de eventos (a)/(b)/(c).
6. **`salida/dataset_real.csv`**: 32.928 filas diarias (1935-01-01..2025-02-24), columnas `Fecha,Ayacucho,Caicara,Ciudad_Bolivar,Palua`, solo valores reales, 28 días (b) a NaN, sin imputaciones y sin corregir (a)/(c).

---

## 1. Inventario

### 1.1 Archivos y hojas

| Archivo / hoja | Formato | Filas | Columnas | Estaciones | Rango | Valores | % NaN |
|---|---|---:|---:|---|---|---:|---:|
| `ayacucho_raw.csv` | CSV ancho (años en columnas, 366 filas) | 366 | 52 | Ayacucho | 1974-01-01..2025-02-24 | 17.983 | 3,747 |
| `caicara_raw.csv` | CSV ancho | 366 | 52 | Caicara | 1974-01-01..2025-02-24 | 18.683 | 0,000 |
| `ciudad_bolivar_raw.csv` | CSV ancho | 366 | 52 | Ciudad Bolívar | 1974-01-01..2025-02-24 | 18.683 | 0,000 |
| `palua_raw.csv` | CSV ancho | 366 | 52 | Palúa | 1974-01-01..2025-02-24 | 18.666 | 0,091 |
| `ayacucho_raw.xlsx` | XLSX ancho (sin encabezado de años) | 366 | 52 | Ayacucho | 1974-01-01..2025-02-24 | 17.983 | 3,747 |
| `DATOS RIO ORINOCO.xlsx` → `Puerto Ayacucho` | Hoja XLSX ancha (mes/día + años) | 366 | 66 | Ayacucho | 1963-06-28..2025-02-24 | 21.823 | 3,108 |
| `DATOS RIO ORINOCO.xlsx` → `Caicara` | Hoja XLSX ancha | 366 | 61 | Caicara | 1968-03-01..2025-02-24 | 19.701 | 5,352 |
| `DATOS RIO ORINOCO.xlsx` → `Ciudad Bolivar` | Hoja XLSX ancha | 366 | 94 | Ciudad Bolívar | 1935-01-01..2025-02-24 | 32.928 | 0,000 |
| `DATOS RIO ORINOCO.xlsx` → `Palua` | Hoja XLSX ancha | 366 | 86 | Palúa | 1943-01-01..2025-02-24 | 29.981 | 0,083 |
| `DATOS RIO ORINOCO.xlsx` → `Entrenar` | Hoja larga (fecha + 4 estaciones + `Pred`) | 5.934 | 40 | 4 estaciones | 2009-01-01..2025-02-24 | 23.593 | 0 / 0 / 0 / 0,051 |
| `dataset_combinado.csv` | CSV largo | 18.683 | 5 | 4 estaciones | 1974-01-01..2025-02-24 | 74.015 | Ayacucho 3,747; Palúa 0,091 |
| `dataset_hasta_2006.csv` | CSV largo | 12.053 | 5 | 4 estaciones | 1974-01-01..2006-12-31 | 48.199 | Palúa 0,108 |
| `dataset_imputado_simpleml.csv` | CSV largo | 18.683 | 5 | 4 estaciones | 1974-01-01..2025-02-24 | 74.715 | Palúa 0,091 |
| `dataset_imputado_simpleml.xlsx` | XLSX largo | 18.683 | 5 | 4 estaciones | 1974-01-01..2025-02-24 | 74.715 | Palúa 0,091 |
| `dataset-orinoco-imputado (1).xlsx` | XLSX largo (1 hoja) | 18.683 | 5 | 4 estaciones | 1974-01-01..2025-02-24 | 74.715 | Palúa 0,091 |

Notas:

- Los CSV anchos tienen una columna por año (1974-2025) y 366 filas de día del año. En años no bisiestos la fila 60 (29-feb) está vacía; el cargador la salta para no correr las fechas.
- `DATOS:Entrenar` tiene 5.934 filas: 5.899 fechas con datos de estaciones (2009-01-01..2025-02-24) y 35 filas finales (2025-02-25..2025-03-31) con solo la columna `Pred`; esas 35 no son datos reales y no se usan.
- El porcentaje de NaN de los archivos largos se calcula sobre la ventana de cada uno. `dataset_combinado` conserva el hueco de Ayacucho 2007-2008 (3,747 % de NaN); en los imputados Ayacucho queda sin NaN solo porque el relleno es sintético, no porque existan datos reales.
- **29-feb:** las cuatro hojas tienen una fila para el 29-feb. En Caicara y Ciudad Bolívar esa fila lleva `DIAS=60` y valores en todos los bisiestos; en Ayacucho y Palúa la columna `DIAS` está vacía, pero la fila existe y **sí tiene valores** en Ayacucho (todos los bisiestos 1964-2024 salvo 2008) y en Palúa solo 2024 (1,31 m). El cargador corregido ya no descarta esa fila. Los valores son compatibles con una interpolación lineal entre el 28-feb y el 1-mar (desvío ≤ 0,03 m); se conservan porque están en el libro, pero se advierte que podrían ser reconstruidos, no observados.

### 1.2 Estadísticas por estación (fuente raw)

| Estación | Hoja fuente | Rango | n reales | Mín (m) | Mediana (m) | Máx (m) | Unidad |
|---|---|---|---:|---:|---:|---:|---|
| Ayacucho | `Puerto Ayacucho` | 1963-06-28..2025-02-24 | 21.823 | 39,36 | 45,80 | 54,28 | m |
| Caicara | `Caicara` | 1968-03-01..2025-02-24 | 19.701 | 20,53 | 28,17 | 36,45 | m |
| Ciudad Bolívar | `Ciudad Bolivar` | 1935-01-01..2025-02-24 | 32.928 | 1,58 | 9,58 | 18,34 | m |
| Palúa | `Palua` | 1943-01-01..2025-02-24 | 29.981 | 0,41 | 5,53 | 12,58 | m |

Los extractos de 1974-2025 dan mínimos algo más altos (Ayacucho 39,68; Ciudad Bolívar 1,88; Palúa 0,63) porque la historia anterior a 1974 contiene los valores más bajos. Eso confirma que el libro, y no los CSV, es la fuente completa.

**Unidades:** metros. Los rangos (0,41-54,28) son plausibles para niveles de río; si fueran centímetros, los valores serían 41-5.428 cm, imposibles para estas estaciones. No hay indicios de mezcla de unidades.

---

## 2. Comparación entre archivos

### 2.1 Método

Para cada estación se compararon **todos los pares de fuentes** en las fechas comunes (`salida/comparacion_pares.csv`, 116 pares). Se calculó:

- nº de fechas comunes y rango común;
- % de valores idénticos con tolerancia |dif| < 0,005 m;
- diferencia media, mediana y máxima absoluta;
- mejor desfase de fecha probando −5..+5 días;
- cociente de medianas (para detectar ×100) y bandera `posible_x100`.

### 2.2 Resultado global

- **116 pares comparados.**
- **0 pares con posible factor ×100.**
- **0 pares con diferencia media > 0,05 m.**
- **En los 116 pares, el mejor desfase global es 0 días** y el cociente de medianas es 1,00.
- Todas las fuentes coinciden al 100 % (|dif| < 0,005) con el raw en las fechas comunes, salvo:
  - `DATOS:Entrenar` en **Ciudad Bolívar**: 89,18 % de coincidencia global, por un desfase local de 1 día en 2020-2021 (ver 2.4);
  - `DATOS:Entrenar` en **Caicara**: 99,983 % (una sola fecha, 2019-12-29, con 0,08 m de diferencia);
  - **Palúa**: 3 valores con diferencia de 0,005 m (redondeo) entre los CSV/derivados y la hoja del libro.

Que el mejor lag global sea 0 no contradice el desfase de Entrenar: ese desfase afecta solo 2 de los 16 años, así que globalmente compensa y el mejor ajuste sigue siendo 0. El desfase se detecta al comparar por tramos (2.4).

### 2.3 Tabla resumen frente al raw

| Estación | Fuente(s) | n común | % idénticos | Dif. media | Dif. máx. abs. | Lag | Ratio medianas |
|---|---|---:|---:|---:|---:|---:|---:|
| Ayacucho | `ayacucho_raw.csv` / `ayacucho_raw.xlsx` | 17.983 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Ayacucho | `dataset_combinado` / imputados (3) | 17.983 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Ayacucho | `dataset_hasta_2006` | 12.053 | 100,000 | 0,000 | 0,000 | 0 | 1,00 |
| Ayacucho | `DATOS:Entrenar` | 5.899 | 100,000 | 0,000 | 0,000 | 0 | 1,00 |
| Caicara | `caicara_raw.csv` / combinado / imputados | 18.683 | 100,000 | 0,000 | 0,003 | 0 | 1,00 |
| Caicara | `dataset_hasta_2006` | 12.053 | 100,000 | 0,000 | 0,003 | 0 | 1,00 |
| Caicara | `DATOS:Entrenar` | 5.899 | 99,983 | 0,000 | 0,080 | 0 | 1,00 |
| Ciudad Bolívar | `ciudad_bolivar_raw.csv` / combinado / imputados | 18.683 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Ciudad Bolívar | `dataset_hasta_2006` | 12.053 | 100,000 | 0,000 | 0,004 | 0 | 1,00 |
| Ciudad Bolívar | `DATOS:Entrenar` | 5.899 | 89,185 | 0,000 | 0,440 | 0 | 1,00 |
| Palúa | `palua_raw.csv` / combinado / imputados | 18.666 | 99,984 | 0,000 | 0,005 | 0 | 1,00 |
| Palúa | `dataset_hasta_2006` | 12.040 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Palúa | `DATOS:Entrenar` | 5.896 | 100,000 | 0,000 | 0,000 | 0 | 1,00 |

La tabla completa de los 116 pares está en `salida/comparacion_pares.csv`; la comparación de cada fuente contra el raw, en `salida/comparacion_vs_raw.csv`.

### 2.4 El desfase local de `Entrenar` en Ciudad Bolívar

`DATOS:Entrenar` no es un archivo raw: es una tabla de entrenamiento derivada. En Ciudad Bolívar, entre **2020-01-01 y 2021-12-30**, sus valores están corridos un día:

- 638 fechas distintas (todas las del tramo) frente al raw;
- `Entrenar(t) = raw(t+1)` en el 99,86 % de los 730 días del tramo;
- comparado contra `raw(t)` solo coincide el 12,6 %.

| Fecha | `Entrenar` | `raw(t)` | `raw(t+1)` | Coincide con |
|---|---:|---:|---:|---|
| 2019-12-31 | 7,38 | 7,38 | 7,34 | raw(t) |
| 2020-01-01 | 7,30 | 7,34 | 7,30 | raw(t+1) |
| 2020-01-02 | 7,26 | 7,30 | 7,26 | raw(t+1) |
| 2021-12-30 | 7,14 | 7,20 | 7,14 | raw(t+1) |
| 2021-12-31 | 7,14 | 7,14 | 7,20 | raw(t) |
| 2022-01-01 | 7,20 | 7,20 | 7,00 | raw(t) |

El resto de `Entrenar` (2009-2019 y 2022-2025) coincide con el raw. Conclusión: el desfase es un error de construcción de esa tabla, no un cambio de fecha del río. No se usa `Entrenar` como fuente.

### 2.5 Duplicados exactos

- `dataset_imputado_simpleml.csv` == `dataset_imputado_simpleml.xlsx` == `dataset-orinoco-imputado (1).xlsx` (verificado con `Series.equals`).
- `dataset_combinado.csv` restringido a ≤2006 == `dataset_hasta_2006.csv` (verificado).
- `ayacucho_raw.csv` == `ayacucho_raw.xlsx` en valores (mismas 17.983 fechas).

---

## 3. Fuente original e imputados

### 3.1 Fuente raw por estación

| Estación | Fuente original | Por qué |
|---|---|---|
| Ayacucho | `DATOS RIO ORINOCO.xlsx` → `Puerto Ayacucho` | Es la serie más larga (1963-2025) y coincide con `ayacucho_raw.csv`/`.xlsx` en las fechas comunes de 1974-2025. |
| Caicara | `DATOS RIO ORINOCO.xlsx` → `Caicara` | Serie 1968-2025; los CSV son el extracto 1974-2025. |
| Ciudad Bolívar | `DATOS RIO ORINOCO.xlsx` → `Ciudad Bolivar` | Serie 1935-2025; los CSV son el extracto 1974-2025. |
| Palúa | `DATOS RIO ORINOCO.xlsx` → `Palua` | Serie 1943-2025; los CSV son el extracto 1974-2025. |

Los cuatro CSV "raw" y `ayacucho_raw.xlsx` son **copias redundantes en las fechas comunes** del libro para 1974-2025, no fuentes independientes. Incluyen los mismos 29-feb que trae el libro (12 en la ventana 1974-2025 para Ayacucho, 1 para Palúa en 2024); no añaden valores nuevos. La única diferencia son redondeos de 0,005 m en 3 fechas de Palúa.

### 3.2 Qué es cada derivado

| Archivo | Contenido | ¿Tiene valores no reales? |
|---|---|---|
| `dataset_combinado.csv` | Concatenación de las 4 series raw 1974-2025 | No añade valores; conserva el hueco de Ayacucho 2007-2008 como NaN y los 29-feb que trae el libro. |
| `dataset_hasta_2006.csv` | `dataset_combinado` recortado a 1974-2006 | No añade valores; conserva los 29-feb del libro. |
| `dataset_imputado_simpleml.*` / `dataset-orinoco-imputado (1).xlsx` | `dataset_combinado` + relleno de Ayacucho 2007-2008 | Sí: 700 fechas sintéticas de Ayacucho (el hueco 2007-01-01..2008-11-30). |
| `DATOS:Entrenar` | Tabla de entrenamiento 2009-2025 | Sí: 638 fechas de Ciudad Bolívar corridas 1 día; 35 valores `Pred` futuros. Los 29-feb coinciden con el raw. |

### 3.3 Valores no reales (imputados o interpolados)

| Estación | Archivo | Fechas sin raw | Detalle |
|---|---|---|---:|
| Ayacucho | imputados (3) | 700 | 2007-01-01..2008-11-30. |
| Palúa | imputados | 17 fechas sin dato | 12 días 29-feb (1976-2020) + 1993-02-27, 1993-02-28, 1993-10-29, 1993-10-30, 1993-10-31. **No están rellenadas.** |
| Ciudad Bolívar | `DATOS:Entrenar` | 638 | 2020-01-01..2021-12-30, corridas 1 día. |

Los 29-feb de Ayacucho (12 en 1974-2025), Palúa (2024) y los de Caicara/Ciudad Bolívar **ya están en el libro** y no se cuentan como valores añadidos. Su desvío respecto de la interpolación lineal entre el 28-feb y el 1-mar es ≤ 0,03 m, así que podrían ser reconstruidos y no observados; se conservan porque no hay forma de distinguirlos y porque el mismo patrón aparece en las cuatro estaciones. Si se aplica un filtro estricto de "solo observado", habría que marcar todos los 29-feb, no solo los de Ayacucho/Palúa.

El bloque imputado de Ayacucho 2007-2008 tiene:

- 700 días, 2007-01-01..2008-11-30;
- mínimo 41,05 m, máximo 51,72 m, media 46,46 m, mediana 46,54 m;
- **0 saltos diarios > 0,5 m** y una racha constante máxima de solo 2 días;
- medias anuales 2007 = 46,40, 2008 = 46,47, vecinas 2006 = 46,40 y 2009 = 45,23.

Es un relleno suave y plausible, pero **no es un dato observado**. El bloque no se usa en `dataset_real.csv`.

### 3.4 Conclusión sobre los derivados

- El único contenido "real" de los combinados/imputados es el que ya está en el libro.
- Los imputados **no completan** Palúa (les faltan 17 fechas en la ventana 1974-2025) y añaden 700 valores sintéticos en Ayacucho.
- `Entrenar` no sirve como fuente de Ciudad Bolívar por el desfase 2020-2021.
- Para modelado, la base limpia es `salida/dataset_real.csv`; si se necesita imputar, debe hacerse después y marcando explícitamente los valores sintéticos.

---

## 4. Control de calidad

### 4.1 Saltos de un día a otro > 0,5 m

| Estación | Saltos > 0,5 m | (a) compuerta | (b) error | (c) escalón | (d) real/temporada |
|---|---:|---:|---:|---:|---:|
| Ayacucho | 19 | 0 | 5 | 2 | 12 |
| Caicara | 13 | 0 | 2 | 1 | 10 |
| Ciudad Bolívar | 17 | 0 | 1 | 4 | 12 |
| Palúa | 37 | 8 | 5 | 1 | 23 |
| **Total** | **86** | **8** | **13** | **8** | **57** |

La columna (b) cuenta **eventos de salto**, no días: además hay días aberrantes que no superan 0,5 m o que forman bloques planos. En total se eliminaron **28 días (b)**: Ayacucho 8, Caicara 8, Ciudad Bolívar 3, Palúa 9.

### 4.2 (a) Pulsos compatibles con aperturas de compuertas (Palúa)

Criterio: subida brusca **solo en Palúa** (Ciudad Bolívar no la replica), con residuo positivo a 5 días que luego decae. Se concentran en abril-julio (época de llenado), salvo 2019-12.

| Fecha | Salto (m) | Salto vecina CB (m) | Residuo +5 d (m) | Residuo +30 d (m) | Evidencia |
|---|---:|---:|---:|---:|---|
| 1971-04-23 | +0,58 | +0,12 | +0,62 | −0,86 | Pulso local que se disipa. |
| 1971-05-18 | +0,58 | +0,26 | +0,43 | −0,30 | Pulso local. |
| 1972-05-01 | +1,16 | +0,30 | +0,27 | −0,43 | Pulso grande, decae. |
| 1974-04-08 | +0,73 | 0,00 | +0,65 | −1,02 | El más limpio: CB no se mueve. |
| 1983-05-27 | +0,51 | +0,09 | +0,54 | −0,38 (a 90 d) | Decae lentamente. |
| 1994-07-17 | +0,58 | +0,11 | +0,46 | −0,07 | Pulso local. |
| 1997-06-28 | +0,58 | +0,10 | +0,63 | +0,12 | Decae. |
| 2019-12-19 | +0,58 | −0,12 | +0,60 | +0,35 | Pulso local; no se corrige. |

No hay registro externo de compuertas en los archivos, así que la clasificación es **compatible con (a)**, no una confirmación documental. Se dejan intactos.

### 4.3 (b) Errores de digitación: días aberrantes puestos a NaN

Criterio: un valor (o un bloque corto) que se aparta del nivel previo y **vuelve** al nivel anterior al día siguiente. Se eliminaron 28 días.

| Estación | Fecha(s) | Días | Tamaño del pico (m) | Valor(es) | Vecinos (antes / después) |
|---|---|---:|---:|---|---|
| Ayacucho | 1982-06-07 | 1 | +0,535 | 49,86 | 49,26 / 49,39 |
| Ayacucho | 1983-01-02 | 1 | −0,960 | 41,78 | 42,74 / 42,74 |
| Ayacucho | 1989-09-17 | 1 | +0,500 | 48,92 | 48,51 / 48,33 |
| Ayacucho | 1992-12-27 | 1 | +0,610 | 43,99 | 43,51 / 43,25 |
| Ayacucho | 1994-04-08 | 1 | −0,975 | 41,02 | 41,88 / 42,11 |
| Ayacucho | 1994-12-29 | 1 | −0,980 | 43,08 | 44,17 / 43,95 |
| Ayacucho | 1997-01-30 | 1 | +0,900 | 42,97 | 42,20 / 41,94 |
| Ayacucho | 2006-09-24 | 1 | +0,495 | 48,97 | 48,56 / 48,39 |
| Caicara | 1969-01-27..28 | 2 | −1,025 / +1,235 | 21,32 / 22,93 | 21,76 / 22,07 |
| Caicara | 1969-04-02..06 | 5 | bloque plano | 21,03 (5 días) | 21,93 / 22,02 |
| Caicara | 1969-11-29 | 1 | −0,510 | 26,97 | 27,45 / 27,51 |
| Ciudad Bolívar | 1979-11-21 | 1 | +0,600 | 11,80 | 11,30 / 11,10 |
| Ciudad Bolívar | 1997-07-13 | 1 | −1,015 | 12,02 | 12,84 / 13,23 |
| Ciudad Bolívar | 2000-01-20 | 1 | +0,520 | 4,97 | 4,50 / 4,40 |
| Palúa | 1949-05-11 | 1 | +0,500 | 3,83 | 3,13 / 3,53 |
| Palúa | 1970-07-10 | 1 | −0,445 | 8,63 | 9,02 / 9,13 |
| Palúa | 1971-04-27 | 1 | −0,520 | 2,45 | 2,95 / 2,99 |
| Palúa | 1972-03-03 | 1 | +0,430 | 2,50 | 1,97 / 2,17 |
| Palúa | 1972-03-20 | 1 | +0,470 | 2,64 | 2,02 / 2,32 |
| Palúa | 1974-02-04 | 1 | +0,575 | 2,45 | 1,79 / 1,96 |
| Palúa | 1975-07-25 | 1 | +0,505 | 9,67 | 9,18 / 9,15 |
| Palúa | 1988-12-02 | 1 | +0,520 | 8,20 | 7,75 / 7,61 |
| Palúa | 2021-10-29 | 1 | +0,995 | 9,79 | 8,73 / 8,86 |

Casos especiales:

- **Caicara 1969-01-27..28:** dos días malos consecutivos; el nivel vuelve el 29 (22,07) al valor previo (21,76). Se eliminan ambos.
- **Caicara 1969-04-02..06:** 5 días exactamente iguales a 21,03 entre 21,93 y 22,02; es un bloque pegado, no una meseta real. Se eliminan los 5.
- **Caicara 1969-11-29:** el salto de +0,54 del 30 es el retorno; el día aberrante es el 29 (26,97).
- **Palúa 2021-10-29:** el 28 (8,73) es una caída leve y el 29 (9,79) el pico; se elimina solo el 29.

**Casos borderline que se dejaron como reales:** Ciudad Bolívar 2010-11-22 (+0,52 m, pico +0,42) y 2016-10-23 (−0,54 m, pico −0,41) tienen retorno incompleto y saltos apenas por debajo del criterio; se clasificaron como (d). Si se quiere un criterio más agresivo, pueden marcarse como (b), pero no se hizo para no borrar fluctuaciones posibles.

### 4.4 (c) Cambios de referencia del cero

Se comparó cada estación con su vecina aplicando el retardo hidráulico: Caicara→Ciudad Bolívar ≈ 4 días, Ayacucho→Ciudad Bolívar ≈ 18 días, Ciudad Bolívar↔Palúa ≈ 0 días.

**Los dos cambios conocidos se verifican exactamente:**

| Fecha | Estación | Salto | Valor antes → después | Δ vecina (m) | Persistencia |
|---|---|---:|---|---:|---|
| 1987-05-15 | Ciudad Bolívar | **−2,20** | 7,52 → 5,32 | +0,12 | No permanente: CB−Palúa anual 1986 = 3,795; 1987 = 3,142; 1988 = 3,411; 1989 = 3,490. Se reabsorbe. |
| 2019-01-01 | Ciudad Bolívar | **+1,00** | 5,60 → 6,60 | −0,18 | No permanente: CB−Palúa anual 2018 = 4,007; 2019 = 3,666; 2020 = 3,638; 2021 = 3,898. Se reabsorbe. |

**Otros escalones candidatos** (salto grande, vecina casi sin cambio, sin retorno inmediato):

| Fecha | Estación | Salto (m) | Δ vecina (m) | Persistencia / evidencia |
|---|---|---:|---:|---|
| 1986-10-01 | Ayacucho | −1,10 | 0,00 (CB+18) | El 30-sep ya trae un pico de +0,50 m (49,98) y el 1-oct cae −1,10 m. Residuo anual 1986 = 36,273; 1987 = 37,205; 1988 = 36,330 (sin offset fijo). |
| 1987-01-01 | Caicara | −1,02 | −0,19 (CB+4) | Residuo anual 1986 = 19,189; 1987 = 19,132; 1988 = 18,428. No persiste. |
| 1972-04-04 | Palúa | −0,94 | −0,01 (CB) | CB−Palúa anual 1972 = 3,872; 1973 = 3,286; 1974 = 3,307. Se recupera. |
| 2000-11-17 | Ciudad Bolívar | −0,56 | +0,20 (Palúa) | CB−Palúa anual 1999 = 3,930; 2000 = 3,582; 2001 = 3,399. Parcial, no fijo. |
| 2020-06-20 | Ayacucho | +0,84 | +0,14 (CB+18) | Residuo anual 2019 = 36,656; 2020 = 36,498; 2021 = 36,222. No persiste. |

Además, **2019-01-11** (Ciudad Bolívar, −0,54; Palúa −0,03) es parte de la recuperación del escalón de 2019-01-01, no un cambio independiente.

**Conclusión:** se confirman los dos cambios conocidos, pero **ninguno se mantiene como un offset permanente** al comparar medias anuales; la serie los reabsorbe en semanas o, como máximo, uno o dos años. No se encontró otro escalón permanente. Por instrucción de la tarea, **no se aplica ninguna corrección** a (c): solo se reporta.

### 4.5 Valores constantes repetidos (posible relleno)

| Estación | Rachas ≥5 días | Días totales en rachas | Racha máxima | Valor de la racha máxima | Fecha | Percentil del valor | Rachas ≥7 d y ≥p75 |
|---|---:|---:|---:|---:|---|---:|---:|
| Ayacucho | 33 | 194 | 12 | 40,46 | 2019-03-07..18 | 5,4 | 1 |
| Caicara | 39 | 230 | 10 | 33,16 | 1979-07-22..31 | 87,8 | 5 |
| Ciudad Bolívar | 70 | 464 | 19 | 2,78 | 2020-03-25..04-12 | 5,7 | 12 |
| Palúa | 83 | 513 | 16 | 1,80 | 1996-04-08..23 | 17,3 | 9 |

Las rachas más sospechosas de relleno son las de **aguas altas** (≥7 días y valor ≥ percentil 75):

- **Ciudad Bolívar:** 1959-08-27 (15,95 m, 13 d), 1971-08-20 (16,12 m, 12 d), 1953-08-16 (16,95 m, 11 d), 1937-08-23 (15,28 m, 9 d), 1959-08-06 (15,63 m, 9 d), 1960-08-27 (16,03 m, 9 d).
- **Palúa:** 1944-09-02 (11,03 m, 11 d), 1943-08-16 (12,18 m, 10 d), 1948-08-11 (10,88 m, 10 d), 1948-07-24 (10,53 m, 9 d), 1947-08-03 (10,83 m, 8 d), 1949-07-13 (9,43 m, 8 d), 1945-09-08 (11,23 m, 7 d), 1946-07-26 (11,13 m, 7 d), 1989-08-08 (9,79 m, 7 d).
- **Caicara:** 1979-07-22 (33,16 m, 10 d), 1975-09-09 (33,18 m, 9 d), 1969-07-22 (32,40 m, 8 d), 1968-09-07 (33,02 m, 7 d), 1988-08-15 (33,08 m, 7 d).
- **Ayacucho:** 2021-07-10 (51,36 m, 7 d).

No se corrigen: una meseta de crecida puede ser real. Se marcan como sospechosas en `salida/constantes.csv` (columna `posible_relleno`). La lista completa tiene 27 rachas marcadas (Ayacucho 1, Caicara 5, Ciudad Bolívar 12, Palúa 9).

### 4.6 Valores fuera de rango físico

No se encontraron valores negativos, ceros ni valores fuera del rango físico plausible:

| Estación | Mín | Máx | Negativos/cero | Percentil 99,9 |
|---|---:|---:|---:|---:|
| Ayacucho | 39,36 | 54,28 | 0 | 53,66 |
| Caicara | 20,53 | 36,45 | 0 | 36,28 |
| Ciudad Bolívar | 1,58 | 18,34 | 0 | 18,02 |
| Palúa | 0,41 | 12,58 | 0 | 12,23 |

### 4.7 Decisiones de limpieza

- **(b):** 28 días a NaN. Lista exacta en `salida/dias_b_eliminar.csv`. No se interpolan.
- **(a):** no se corrigen; se conservan los pulsos.
- **(c):** no se aplican; se conservan los valores originales y se reportan.
- **29-feb:** se conservan los valores que trae el libro (Ayacucho en todos los bisiestos salvo 2008; Palúa solo 2024; Caicara/CB en todos). No se añaden interpolaciones nuevas; los 29-feb que faltan en Palúa quedan NaN.
- **Huecos:** quedan NaN; no se rellenan.
- **Imputados:** no se usan.

---

## 5. Cobertura y rango recomendado

### 5.1 Cobertura por año (1974-2025)

Todos los años no listados tienen **100 %** en las 4 estaciones. En los bisiestos, Ayacucho tiene **100 %** (el libro trae el 29-feb) y Palúa **99,73 %** (le falta el 29-feb), salvo 2024, que también tiene 100 % en Palúa. 2008 se detalla aparte.

| Año | Ayacucho | Caicara | Ciudad Bolívar | Palúa | Motivo |
|---|---:|---:|---:|---:|---|
| 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2012, 2016, 2020 | 100,00 | 100,00 | 100,00 | 99,73 | Bisiestos: Palúa no tiene 29-feb; Ayacucho sí. |
| 2024 | 100,00 | 100,00 | 100,00 | 100,00 | Bisiesto completo: Palúa sí tiene 2024-02-29 (1,31 m). |
| 1993 | 100,00 | 100,00 | 100,00 | **98,63** | Faltan 5 días: 1993-02-27/28 y 1993-10-29/30/31. |
| 2007 | **0,00** | 100,00 | 100,00 | 100,00 | Hueco completo de Ayacucho. |
| 2008 | **8,47** | 100,00 | 100,00 | 99,73 | Ayacucho solo tiene diciembre (31 días). |
| 2025 | 15,07 | 15,07 | 15,07 | 15,07 | Año parcial: hasta 2025-02-24 (55 días). |
| Resto 1974-2024 | 100,00 | 100,00 | 100,00 | 100,00 | — |

La tabla año a año completa (1935-2025, con días observados y faltantes) está en `salida/cobertura_anual.csv`; la matriz 1974-2025, en el Anexo C.

### 5.2 Cobertura antes de 1974 (resumen)

| Etapa | Ayacucho | Caicara | Ciudad Bolívar | Palúa |
|---|---:|---:|---:|---:|
| 1935-1942 | — | — | 100 % | — |
| 1943-1962 | — | — | 100 % | 100 % (99,73 en bisiestos) |
| 1963 | 51,23 % (inicia 28-jun) | — | 100 % | 100 % |
| 1964-1967 | 100 % | — | 100 % | 100 % (99,73 en bisiestos) |
| 1968 | 100 % | 83,61 % (inicia 1-mar) | 100 % | 99,73 % |
| 1969 | 100 % | 100 % | 100 % | 100 % |
| 1970 | 100 % | 95,07 % (hueco 6-23 abr) | 100 % | 100 % |
| 1971-1973 | 100 % | **0 %** | 100 % | 100 % (99,73 en 1972) |

### 5.3 Huecos internos > 1 día

| Estación | Desde | Hasta | Días |
|---|---|---|---:|
| Ayacucho | 2007-01-01 | 2008-11-30 | **700** |
| Caicara | 1971-01-01 | 1973-12-31 | **1.096** |
| Caicara | 1970-04-06 | 1970-04-23 | 18 |
| Palúa | 1993-10-29 | 1993-10-31 | 3 |
| Palúa | 1993-02-27 | 1993-02-28 | 2 |

Además hay días faltantes aislados: los 29-feb que faltan en Palúa (20 entre 1944 y 2020; 2024 sí está), no listados aquí. El detalle completo está en `salida/huecos.csv`.

### 5.4 Rangos candidatos

| Desde | Hasta | Años | Años completos (4 est.) | Ayacucho | Caicara | Ciudad Bolívar | Palúa | Cobertura mínima |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1963 | 2006 | 44 | 33 | 98,892 | 81,333 | 100,000 | 99,900 | 81,333 |
| 1968 | 2006 | 39 | 33 | 100,000 | 91,759 | 100,000 | 99,895 | 91,759 |
| **1974** | **2006** | **33** | **32** | **100,000** | **100,000** | **100,000** | **99,892** | **99,892** |
| 1974 | 2024 | 51 | 48 | 96,242 | 100,000 | 100,000 | 99,909 | 96,242 |
| 1988 | 2006 | 19 | 18 | 100,000 | 100,000 | 100,000 | 99,856 | 99,856 |
| 1988 | 2024 | 37 | 34 | 94,821 | 100,000 | 100,000 | 99,896 | 94,821 |
| 2000 | 2024 | 25 | 23 | 92,335 | 100,000 | 100,000 | 99,934 | 92,335 |
| **2009** | **2024** | **16** | **16** | **100,000** | **100,000** | **100,000** | **99,949** | **99,949** |
| 2010 | 2024 | 15 | 15 | 100,000 | 100,000 | 100,000 | 99,945 | 99,945 |
| 1974 | 2018 | 45 | 42 | 95,741 | 100,000 | 100,000 | 99,903 | 95,741 |

### 5.5 Rango recomendado y partición

**Rango recomendado: 2009-2024 (16 años).**

Justificación:

1. **4 estaciones completas:** 16/16 años con cobertura ≥99,949 % en las 4 estaciones. Solo faltan 3 días 29-feb de Palúa (2012, 2016 y 2020); Ayacucho está completo.
2. **Bloque continuo y reciente:** evita el hueco de Ayacucho 2007-2008 y el de Caicara 1971-1973.
3. **Entrenamiento limpio:** 2009-2018 no contiene ningún evento (a), (b) ni (c) de los detectados. Los dos eventos (c) del bloque (Ciudad Bolívar 2019-01-01 y el candidato Ayacucho 2020-06-20) caen en **validación**, no en entrenamiento.
4. **Prueba limpia:** 2022-2024 no contiene eventos (a)/(b)/(c); solo fluctuaciones reales de temporada.

| Partición | Años | Ayacucho | Caicara | Ciudad Bolívar | Palúa | Días |
|---|---|---:|---:|---:|---:|---:|
| Train | 2009-2018 (10) | 100,000 | 100,000 | 100,000 | 99,945 | 3.652 |
| Validación | 2019-2021 (3) | 100,000 | 100,000 | 100,000 | 99,909 | 1.096 |
| Prueba | 2022-2024 (3) | 100,000 | 100,000 | 100,000 | 100,000 | 1.096 |

**Advertencia para validación:** 2019-01-01 (Ciudad Bolívar, +1,00 m), 2019-12-19 (Palúa, pulso (a)) y 2020-06-20 (Ayacucho, candidato (c) de +0,84 m) están dentro de validación. No se corrigen; si se quiere una validación "limpia", se pueden excluir esos días concretos. Si se excluye todo 2019, la validación queda en 2 años (2020-2021).

### 5.6 Alternativa si se necesita más historia

**1974-2006 (33 años):** cobertura mínima 99,892 %, 32/33 años completos (Palúa 1993 tiene 5 días faltantes). Partición sugerida: train 1974-1995, validación 1996-2001, prueba 2002-2006.

Ventaja: 33 años y cuatro estaciones. Desventaja: el entrenamiento contiene los eventos (c) de 1986-10-01 (Ayacucho), 1987-01-01 (Caicara) y 1987-05-15 (Ciudad Bolívar), además de los pulsos (a) de 1974, 1983, 1994 y 1997. Si se elige esta alternativa, hay que documentar que el modelo aprende esos escalones tal como están.

**No recomendable:** 1963-2006 o 1968-2006 (Caicara tiene 0 % en 1971-1973); 1974-2024 o 2000-2024 (incluyen el hueco de Ayacucho 2007-2008).

---

## 6. `salida/dataset_real.csv`

**Especificación:**

- 32.928 filas, una por día, 1935-01-01..2025-02-24 (rango completo disponible).
- Columnas exactas: `Fecha,Ayacucho,Caicara,Ciudad_Bolivar,Palua`.
- Fecha en formato `YYYY-MM-DD`, ordenada y sin duplicados.
- Valores en metros, tomados de las hojas raw del libro.
- 28 días (b) a NaN (`salida/dias_b_eliminar.csv`).
- Sin imputaciones, sin interpolaciones de 29-feb, sin correcciones (a)/(c).
- NaN donde no hay dato real (antes del inicio de cada estación, huecos, 29-feb de Ayacucho/Palúa).

**Conteos:**

| Columna | Valores reales | NaN | Primer dato | Último dato |
|---|---:|---:|---|---|
| Ayacucho | 21.800 | 11.128 | 1963-06-28 | 2025-02-24 |
| Caicara | 19.693 | 13.235 | 1968-03-01 | 2025-02-24 |
| Ciudad Bolívar | 32.925 | 3 | 1935-01-01 | 2025-02-24 |
| Palúa | 29.971 | 2.957 | 1943-01-01 | 2025-02-24 |

Los NaN de Ciudad Bolívar son exactamente los 3 días (b) eliminados; los de Caicara, sus 8 días (b) más el período previo 1935-1968 y el hueco 1970-1973. Los de Ayacucho y Palúa incluyen, además, los períodos previos a su inicio, los 29-feb y (en Palúa) los huecos de 1993.

**Verificaciones ejecutadas (`scripts/06_dataset_real.py`):**

- 32.928 filas, 0 fechas duplicadas, diferencia diaria siempre 1 día.
- 1987-05-15 = 5,32 y 1987-05-14 = 7,52 (cambio de referencia conservado, no corregido).
- 2019-01-01 = 6,60 y 2018-12-31 = 5,60 (conservado).
- 1974-04-08 = 2,12 (pulso (a) conservado).
- 2007-06-01 = NaN y 2008-12-01 = 46,79 (hueco no rellenado).
- 2024-02-29 = NaN en Ayacucho y 3,54 en Ciudad Bolívar (29-feb real solo donde existe).
- 1969-04-03 = NaN (bloque (b) eliminado) y 1969-04-07 = 22,02.

**Limitación:** el archivo es solo de datos reales. Para modelos que necesiten una serie completa, la imputación debe hacerse *después* del corte train/validación/prueba y ajustarse **solo con el train**, para no filtrar información del futuro. Los archivos imputados existentes no permiten verificar esa separación: el bloque 2007-2008 se rellenó de una vez y no se documenta qué información usó el modelo; no deberían usarse como verdad.

---

## 7. Conclusiones y recomendaciones

1. **Usar el libro como fuente**, no los CSV "raw" (que son copias) ni los imputados. Los CSV sirven solo como verificación.
2. **No usar `DATOS:Entrenar` para Ciudad Bolívar** sin corregir el desfase 2020-2021; para las otras estaciones coincide con el raw.
3. **No tratar los imputados como datos reales:** el bloque 2007-2008 de Ayacucho es sintético y los imputados dejan 17 fechas de Palúa sin cubrir.
4. **Eliminar los 28 días (b)** al entrenar; el `dataset_real.csv` ya los tiene a NaN.
5. **No corregir (a) ni (c)** en la serie base. Si el objetivo es detectar anomalías, los eventos (c) son útiles como variables/exclusiones; si el objetivo es predecir niveles, conviene marcarlos y evaluar el modelo por separado en esos tramos.
6. **Entrenar en 2009-2024** con la partición 2009-2018 / 2019-2021 / 2022-2024. Si se necesita más historia, usar 1974-2006 y documentar los escalones de 1986-1987.
7. **Cuidado con la fuga temporal:** cualquier imputación, normalización o selección de variables debe ajustarse solo con el train. El `dataset_real.csv` está pensado para eso: conserva los NaN y no rellena nada.

---

## Anexo A. Comparación completa de cada fuente contra el raw

Archivo: `salida/comparacion_vs_raw.csv` (29 filas). Columnas: estación, fuente, n común, % idénticos, diferencia media, diferencia máxima absoluta, mejor lag global, cociente de medianas.

| Estación | Fuente | n común | % idénticos | Dif. media | Dif. máx. | Lag | Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| Ayacucho | `DATOS:Entrenar` | 5.899 | 100,000 | 0,000 | 0,000 | 0 | 1,00 |
| Ayacucho | `ayacucho_raw.csv` | 17.971 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Ayacucho | `ayacucho_raw.xlsx` | 17.971 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Ayacucho | imputado (1).xlsx | 17.971 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Ayacucho | `dataset_combinado.csv` | 17.971 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Ayacucho | `dataset_hasta_2006.csv` | 12.045 | 100,000 | 0,000 | 0,000 | 0 | 1,00 |
| Ayacucho | imputado `.csv` | 17.971 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Ayacucho | imputado `.xlsx` | 17.971 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Caicara | `DATOS:Entrenar` | 5.899 | 99,983 | 0,000 | 0,080 | 0 | 1,00 |
| Caicara | `caicara_raw.csv` | 18.683 | 100,000 | 0,000 | 0,003 | 0 | 1,00 |
| Caicara | imputado (1).xlsx | 18.683 | 100,000 | 0,000 | 0,003 | 0 | 1,00 |
| Caicara | `dataset_combinado.csv` | 18.683 | 100,000 | 0,000 | 0,003 | 0 | 1,00 |
| Caicara | `dataset_hasta_2006.csv` | 12.053 | 100,000 | 0,000 | 0,003 | 0 | 1,00 |
| Caicara | imputado `.csv` | 18.683 | 100,000 | 0,000 | 0,003 | 0 | 1,00 |
| Caicara | imputado `.xlsx` | 18.683 | 100,000 | 0,000 | 0,003 | 0 | 1,00 |
| Ciudad Bolívar | `DATOS:Entrenar` | 5.899 | 89,185 | 0,000 | 0,440 | 0 | 1,00 |
| Ciudad Bolívar | `ciudad_bolivar_raw.csv` | 18.683 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Ciudad Bolívar | imputado (1).xlsx | 18.683 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Ciudad Bolívar | `dataset_combinado.csv` | 18.683 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Ciudad Bolívar | `dataset_hasta_2006.csv` | 12.053 | 100,000 | 0,000 | 0,004 | 0 | 1,00 |
| Ciudad Bolívar | imputado `.csv` | 18.683 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Ciudad Bolívar | imputado `.xlsx` | 18.683 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Palúa | `DATOS:Entrenar` | 5.896 | 100,000 | 0,000 | 0,000 | 0 | 1,00 |
| Palúa | imputado (1).xlsx | 18.665 | 99,984 | 0,000 | 0,005 | 0 | 1,00 |
| Palúa | `dataset_combinado.csv` | 18.665 | 99,984 | 0,000 | 0,005 | 0 | 1,00 |
| Palúa | `dataset_hasta_2006.csv` | 12.040 | 100,000 | 0,000 | 0,005 | 0 | 1,00 |
| Palúa | imputado `.csv` | 18.665 | 99,984 | 0,000 | 0,005 | 0 | 1,00 |
| Palúa | imputado `.xlsx` | 18.665 | 99,984 | 0,000 | 0,005 | 0 | 1,00 |
| Palúa | `palua_raw.csv` | 18.665 | 99,984 | 0,000 | 0,005 | 0 | 1,00 |

## Anexo B. Scripts y salidas

| Script | Qué hace | Salidas principales |
|---|---|---|
| `scripts/common.py` | Carga y normalización de las 15 fuentes | — |
| `scripts/01_inventario.py` | Inventario de archivos, hojas y estaciones | `inventario_archivos.csv`, `inventario_estaciones.csv`, `inventario_libro.json` |
| `scripts/02_comparacion.py` | 116 comparaciones por pares, lags, offsets, ×100 | `comparacion_pares.csv`, `comparacion_vs_raw.csv` |
| `scripts/03_fuente_imputados.py` | Fuente raw, bloques imputados, duplicados, desfase Entrenar | `imputados_resumen.csv`, `imputados_bloques.csv`, `fuente_imputados.json`, `log_03.txt` |
| `scripts/04_qc.py` | Saltos >0,5, clasificación a/b/c, constantes, rangos | `saltos_mayores_05.csv`, `eventos_sospechosos.csv`, `dias_b_eliminar.csv`, `constantes.csv`, `qc_resumen.json` |
| `scripts/05_cobertura.py` | Cobertura anual, huecos, rangos y partición | `cobertura_anual.csv`, `cobertura_rangos.csv`, `particion_recomendada.csv`, `huecos.csv`, `cobertura_resumen.json` |
| `scripts/06_dataset_real.py` | Genera el dataset limpio y verifica puntos de control | `dataset_real.csv`, `dataset_real_resumen.json` |
| `scripts/07_residuos.py` | Residuos anuales estación-vecina | `residuo_anual.csv`, `log_07.txt` |

## Anexo C. Matriz de cobertura 1974-2025 (%)

| Año | Ayacucho | Caicara | Ciudad Bolívar | Palúa |
|---:|---:|---:|---:|---:|
| 1974 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1975 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1976 | 99,73 | 100,00 | 100,00 | 99,73 |
| 1977 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1978 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1979 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1980 | 99,73 | 100,00 | 100,00 | 99,73 |
| 1981 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1982 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1983 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1984 | 99,73 | 100,00 | 100,00 | 99,73 |
| 1985 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1986 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1987 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1988 | 99,73 | 100,00 | 100,00 | 99,73 |
| 1989 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1990 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1991 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1992 | 99,73 | 100,00 | 100,00 | 99,73 |
| 1993 | 100,00 | 100,00 | 100,00 | 98,63 |
| 1994 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1995 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1996 | 99,73 | 100,00 | 100,00 | 99,73 |
| 1997 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1998 | 100,00 | 100,00 | 100,00 | 100,00 |
| 1999 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2000 | 99,73 | 100,00 | 100,00 | 99,73 |
| 2001 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2002 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2003 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2004 | 99,73 | 100,00 | 100,00 | 99,73 |
| 2005 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2006 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2007 | **0,00** | 100,00 | 100,00 | 100,00 |
| 2008 | **8,47** | 100,00 | 100,00 | 99,73 |
| 2009 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2010 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2011 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2012 | 99,73 | 100,00 | 100,00 | 99,73 |
| 2013 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2014 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2015 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2016 | 99,73 | 100,00 | 100,00 | 99,73 |
| 2017 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2018 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2019 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2020 | 99,73 | 100,00 | 100,00 | 99,73 |
| 2021 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2022 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2023 | 100,00 | 100,00 | 100,00 | 100,00 |
| 2024 | 99,73 | 100,00 | 100,00 | 99,73 |
| 2025 | 15,07 | 15,07 | 15,07 | 15,07 |

---

## Anexo D. Revisión cruzada y decisiones aplicadas al proyecto (2026-10-28)

Se hicieron dos auditorías independientes (OpenCode/DeepSeek, este informe; Gemini 3.8 Flash, aparte). Coinciden en:
fuente real = hojas por estación del libro; `Entrenar` tiene Ciudad Bolívar corrida 1 día en 2020-01-01..2021-12-30;
el bloque de Ayacucho 2007-01-01..2008-11-30 de los imputados es sintético; los tres imputados son idénticos.

**Cambios de referencia de Ciudad Bolívar: no son permanentes.** Mediana anual de CB − Caicara(t−4):
1986 = −18,94, 1988 = −18,18 (un escalón real de −2,20 m la habría bajado a ≈ −21); 2018 = −18,33,
2020 = −18,42 (sin el +1,00 m). Día a día, CB − Palúa vuelve a su nivel previo en ~7 días (1987) y ~10 días (2019).
Por eso `CAMBIOS_DE_REFERENCIA` quedó vacío: la corrección anterior sumaba +1 m a todo 2009-2018.

**Dataset del proyecto:** `data/niveles_reales.xlsx` = `dataset_real_1935_2025.csv` desde 2009-01-01, con
Ciudad Bolívar 2019-01-01..2019-01-10 a NaN (pulso espurio: +1 m mientras las vecinas bajan). NaN restantes:
Palúa 2021-10-29 (b) y los 29-feb de Palúa 2012/2016/2020. Se interpolan para las variables y no se evalúan.

**Rango:** 2009-2025 (bloque continuo con las 4 estaciones al 100 %). Se mantiene la partición del proyecto
(train 2009-2016, validación 2017-2019, prueba 2020-feb 2025). Antes de 2009 solo sirve 1974-2006 como bloque aparte,
por el hueco de Ayacucho.
