# Perfiles financieros de las empresas medianas y grandes de Colombia
### ACP y K-Means sobre la base de Supersociedades (cierre 2024)

Taller individual · *Machine Learning aplicado a Economía y Finanzas*

**Aplicación desplegada:** https://perfiles-financieros-colombia.streamlit.app

---

## 1. Contexto y pregunta

Los rankings empresariales ordenan a las compañías por ingresos, y los análisis sectoriales las agrupan por actividad
económica. Ninguna de las dos miradas dice **cómo** gana dinero cada empresa ni **qué tan sana** es su estructura
financiera. Dentro de un mismo sector conviven comercializadoras de alto volumen y margen estrecho, empresas de
servicios muy rentables y sin deuda, compañías que crecieron apalancadas y otras que, pese a su tamaño, apenas
generan utilidad.

**Preguntas que guían el ejercicio:**

1. ¿Cuáles son las dimensiones de fondo (rentabilidad, apalancamiento, eficiencia, crecimiento, tamaño) que explican
   la mayor parte de las diferencias financieras entre empresas?
2. ¿Existen grupos de empresas con perfiles financieros parecidos? ¿Cómo se describen?
3. ¿Esos grupos coinciden con el macrosector o lo atraviesan?

**¿Por qué ACP y K-Means?** Los indicadores financieros están correlacionados (margen y ROA se mueven juntos, y el
endeudamiento va en contra de la rentabilidad). El ACP resume esa información compartida en pocos ejes no
correlacionados que se pueden interpretar. No existe una clasificación "verdadera" de perfiles financieros, así que
el problema es no supervisado: K-Means busca grupos de empresas cercanas en el espacio de los componentes, sin que
indicadores redundantes pesen doble en la distancia.

## 2. Datos

| | |
|---|---|
| **Fuente** | Superintendencia de Sociedades. *Informe de las 9.000 empresas siguientes más grandes del país* (publicado en agosto de 2025). |
| **Archivo** | `data/supersociedades_9000_empresas_2024.xlsx`, descargado del portal de Supersociedades ([enlace directo](https://www.supersociedades.gov.co/documents/20122/460525/Base%209000_empresas_2025%20Periodista.xlsx/7e8159b9-38a6-1f1b-24b1-7889cc25fcad?t=1755257649208)). |
| **Unidad de observación** | Una sociedad (NIT) en los puestos 1.001 a 10.000 del ranking nacional por ingresos operacionales. |
| **Cobertura** | 9.000 empresas de todo el país, vigiladas por Supersociedades, Supersalud, Supertransporte, Supervigilancia, Superservicios y Superfinanciera. Estados financieros NIIF de 2024 y 2023, en miles de pesos. |
| **Variables originales** | Ingresos operacionales, ganancia (pérdida), activos, pasivos y patrimonio (2024 y 2023); región, departamento, ciudad, CIIU, macrosector, supervisor y marco normativo. |
| **Licencia** | Información pública. Fuente: Superintendencia de Sociedades, www.supersociedades.gov.co |

> ¿Por qué esta base y no la de datos.gov.co? El conjunto `6cat-2gcs` de datos.gov.co publica las mismas cuentas,
> pero en **billones de pesos redondeados a dos decimales**. Para una empresa típica de la muestra, eso deja activos
> de "0,01", lo que hace inservibles las razones financieras. El Excel de Supersociedades trae las cifras completas.

### Indicadores construidos (variables del modelo)

| Indicador | Cálculo | Qué mide |
|---|---|---|
| Margen neto | Ganancia / Ingresos | Utilidad por cada peso vendido |
| ROA | Ganancia / Activos | Rentabilidad de los activos |
| Endeudamiento | Pasivos / Activos | Financiación con terceros |
| Rotación de activos | Ingresos / Activos | Eficiencia e intensidad de capital |
| Crecimiento de ingresos | Ingresos 2024 / Ingresos 2023 − 1 | Dinámica comercial |
| Crecimiento de activos | Activos 2024 / Activos 2023 − 1 | Expansión del balance |
| Tamaño | log10(Activos en pesos) | Escala |
| Cambio en ROA *(opcional)* | ROA 2024 − ROA 2023 | Mejora o deterioro de la rentabilidad |

Se trabaja con razones y no con montos, porque los montos están dominados por el tamaño: un ACP sobre ellos solo
encontraría la dimensión "empresa grande / empresa pequeña".

### Limitaciones de la fuente
- Solo hay cinco cuentas por año: no se pueden calcular liquidez, EBITDA ni cobertura de intereses.
- Hay sesgo de selección: las 1.000 más grandes y las pymes no están en la base.
- Solo hay dos años, así que el crecimiento puede reflejar eventos puntuales (fusiones, escisiones).
- Algunas sociedades reportan estados separados (matrices), lo que puede inflar activos o utilidades.

## 3. Metodología

1. **Calidad de datos.** No hay NIT duplicados y el balance cuadra en todas las empresas. Se excluyen 19 empresas
   sin información de 2023 y 19 con ingresos o activos ≤ 0, que dejan los cocientes indefinidos. Quedan **8.962 empresas**.
   Hay una opción en la app para excluir también las empresas con patrimonio negativo.
2. **Valores extremos.** Los indicadores tienen colas muy largas (márgenes > 100 %, crecimientos de miles por ciento).
   Se aplica una **winsorización** al 2,5 % en cada cola: los extremos se reemplazan por el percentil límite y no se
   elimina ninguna empresa. Sin este paso, K-Means forma grupos de unas pocas empresas atípicas.
3. **Estandarización** con `StandardScaler` (puntaje z), porque los indicadores están en escalas muy distintas.
4. **ACP** sobre los indicadores estandarizados. El número de componentes se elige con el criterio de Kaiser
   (valor propio > 1), contrastado con el gráfico de sedimentación y la varianza acumulada.
5. **K-Means** sobre los componentes retenidos (`n_init=20`, `random_state=42`). K se elige comparando el método del
   codo y el coeficiente de silueta para K = 2…10, y revisando que no haya grupos diminutos.
6. **Interpretación** con las medianas de los indicadores **originales** en cada segmento, su composición por sector y
   región, y la silueta de cada empresa.

## 4. Resultados principales (configuración por defecto)

**ACP.** Tres componentes tienen valor propio > 1 y conservan el **74,9 %** de la varianza:

| Componente | % varianza | Interpretación |
|---|---|---|
| CP1 | 31,7 % | **Rentabilidad**: margen y ROA altos, en sentido opuesto al endeudamiento |
| CP2 | 24,8 % | **Escala vs. rotación**: empresas grandes de balance pesado frente a comercializadoras de alta rotación |
| CP3 | 18,5 % | **Expansión apalancada**: crecimiento de ingresos y activos acompañado de más deuda |

**K-Means.** La silueta es máxima en **K = 4** (≈ 0,28). Esa estructura es moderada: los grupos existen, pero con
fronteras difusas.

| Segmento | % empresas | Perfil (medianas) |
|---|---|---|
| Baja rentabilidad y alto endeudamiento | 45 % | Margen 1,8 %, ROA 2,0 %, endeudamiento 63 %, ventas estancadas; 25 % con pérdidas |
| Rentables y conservadoras | 24 % | Margen 11,8 %, ROA 12,3 %, endeudamiento 30 %; prácticamente sin pérdidas |
| Alto volumen, margen estrecho | 23 % | Rotación 3,4 veces, margen 1,9 %, ROA 6,8 %; 54 % comercio; las más pequeñas |
| Expansión apalancada | 8 % | Ingresos +85 %, activos +68 %, endeudamiento 66 % |

**Conclusión central:** el sector económico no determina el perfil financiero. Comercio, servicios y manufactura
aparecen en los cuatro segmentos, y la región tampoco los diferencia. Los segmentos sirven como tipologías de
referencia (por ejemplo, para vigilar cartera comercial o priorizar la supervisión), no como categorías rígidas.

La app permite cambiar los indicadores, el porcentaje de winsorización, los sectores analizados, el número de
componentes y K, y todo se recalcula en vivo.

## 5. Estructura del repositorio

```
├── app.py                     # Aplicación Streamlit (6 secciones)
├── src/
│   └── procesamiento.py       # Limpieza, indicadores, winsorización, ACP, K-Means, perfiles
├── data/
│   └── supersociedades_9000_empresas_2024.xlsx   # Base original sin modificar
├── .streamlit/config.toml     # Tema visual
├── requirements.txt
└── README.md
```

Todas las tablas, métricas y gráficos se calculan cuando se ejecuta la app, a partir del Excel original; no hay
resultados precalculados ni imágenes fijas.

## 6. Cómo ejecutar

```bash
git clone https://github.com/sebasrolemmel/perfiles-financieros-empresas-colombia.git
cd perfiles-financieros-empresas-colombia
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10 o superior. La app abre en `http://localhost:8501`.

**Despliegue:** en Streamlit Community Cloud, crear una app a partir de este repositorio, con la rama `main` y el
archivo principal `app.py`.

## 7. Limitaciones del análisis
1. No se mide liquidez ni capacidad de pago, por falta de cuentas en la fuente.
2. Es una sola foto (2024 frente a 2023); con más años, el segmento de expansión podría resultar transitorio.
3. Los resultados no se generalizan a las 1.000 más grandes ni a las pymes.
4. Los segmentos son sensibles a la winsorización y a los indicadores elegidos (se puede comprobar en la app).
5. K-Means supone grupos esféricos de tamaño similar. Una extensión natural serían los modelos de mezcla gaussiana
   o el clustering jerárquico.
