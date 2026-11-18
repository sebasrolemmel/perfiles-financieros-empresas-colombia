"""
Perfiles financieros de las empresas medianas-grandes de Colombia
ACP + K-Means sobre la base de Supersociedades (9.000 empresas siguientes, cierre 2024)

Ejecutar:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import procesamiento as pr

st.set_page_config(
    page_title="Perfiles financieros · Empresas de Colombia",
    page_icon="📊",
    layout="wide",
)

PALETA = ["#0F766E", "#C2410C", "#4338CA", "#B45309", "#BE185D", "#15803D", "#0369A1", "#7C3AED", "#A16207", "#475569"]

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    h1, h2, h3 {color: #0F3D3E;}
    .nota {background:#F1F5F4; border-left:4px solid #0F766E; padding:0.8rem 1rem; border-radius:4px; margin:0.5rem 0 1rem 0;}
    .alerta {background:#FFF7ED; border-left:4px solid #C2410C; padding:0.8rem 1rem; border-radius:4px; margin:0.5rem 0 1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def nota(texto, clase="nota"):
    st.markdown(f'<div class="{clase}">{texto}</div>', unsafe_allow_html=True)


def fmt(valor, formato):
    if pd.isna(valor):
        return "—"
    return f"{valor*100:,.1f} %" if formato == "pct" else f"{valor:,.2f}"


# ---------------------------------------------------------------------------
# Datos (con caché) y controles globales
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Leyendo la base de Supersociedades…")
def datos_crudos():
    return pr.cargar_datos()


@st.cache_data(show_spinner="Preparando los datos…")
def preparar(excluir_neg, sectores, variables, q):
    crudo = datos_crudos()
    d, bitacora = pr.limpiar(crudo, excluir_patrimonio_negativo=excluir_neg)
    if sectores:
        n = len(d)
        d = d[d["macrosector"].isin(sectores)].reset_index(drop=True)
        bitacora = pd.concat(
            [bitacora, pd.DataFrame([("Fuera de los macrosectores seleccionados", n - len(d))], columns=bitacora.columns)],
            ignore_index=True,
        )
    v = pr.construir_indicadores(d)
    w, limites = pr.winsorizar(v[list(variables)], q)
    X = pr.estandarizar(w)
    return d, bitacora, v, w, limites, X


@st.cache_data(show_spinner="Calculando ACP…")
def acp(X, variables):
    r = pr.ajustar_acp(X, list(variables))
    return r["varianza"], r["cargas"], r["correlaciones"], r["puntajes"]


@st.cache_data(show_spinner="Probando distintos valores de K…")
def curva_k(Z, k_max):
    return pr.evaluar_k(Z, 2, k_max)


@st.cache_data(show_spinner="Segmentando empresas…")
def segmentar(Z, k):
    km = pr.ajustar_kmeans(Z, k)
    # Renumerar por tamaño (1 = el más grande) solo para que la numeración sea estable;
    # el número NO indica orden ni jerarquía.
    orden = pd.Series(km.labels_).value_counts().index.tolist()
    mapa = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
    etiquetas = np.array([mapa[e] for e in km.labels_])
    centros = km.cluster_centers_[orden]
    return etiquetas, centros


crudo = datos_crudos()

with st.sidebar:
    st.title("📊 Perfiles financieros")
    st.caption("ACP + K-Means · Supersociedades 2024")
    pagina = st.radio(
        "Navegación",
        [
            "1. Contexto",
            "2. Exploración",
            "3. Preparación",
            "4. ACP",
            "5. K-Means",
            "6. Conclusiones",
        ],
    )
    st.divider()
    st.subheader("Decisiones del análisis")
    variables = st.multiselect(
        "Indicadores que entran al modelo",
        options=list(pr.INDICADORES.keys()),
        default=pr.INDICADORES_POR_DEFECTO,
        format_func=pr.nombre,
        help="Se necesitan al menos 4. Por defecto se omite 'Cambio en ROA' porque repite información de ROA y crecimiento.",
    )
    q = st.slider(
        "Recorte de extremos (winsorización, % en cada cola)",
        0.0, 5.0, 2.5, 0.5,
        help="Los valores por debajo del percentil q y por encima del 100−q se reemplazan por esos percentiles.",
    ) / 100
    excluir_neg = st.checkbox("Excluir empresas con patrimonio negativo", value=False)
    sectores = st.multiselect(
        "Analizar solo estos macrosectores (vacío = todos)",
        sorted(crudo["macrosector"].unique()),
    )
    st.divider()
    st.caption(
        "Fuente: Superintendencia de Sociedades, informe de las 9.000 empresas siguientes "
        "más grandes del país (corte 31-dic-2024)."
    )

if len(variables) < 4:
    st.error("Selecciona al menos 4 indicadores en la barra lateral para continuar.")
    st.stop()

variables = tuple(variables)
d, bitacora, v, w, limites, X = preparar(excluir_neg, tuple(sectores), variables, q)

if len(d) < 50:
    st.error("Con los filtros elegidos quedan muy pocas empresas para el análisis. Amplía la selección.")
    st.stop()

varianza, cargas, correl, puntajes = acp(X, variables)
sug = pr.sugerir_componentes(varianza)

# Número de componentes y K se guardan en session_state para compartir entre páginas
if "n_comp" not in st.session_state or st.session_state.get("n_comp_vars") != variables:
    st.session_state.n_comp = sug["kaiser"]
    st.session_state.n_comp_vars = variables
n_comp = min(st.session_state.n_comp, len(variables))
Z = puntajes[:, :n_comp]
curva = curva_k(Z, 10)
k_sugerido = int(curva.loc[curva["Silueta"].idxmax(), "K"])
if "k" not in st.session_state:
    st.session_state.k = k_sugerido
etiquetas, centros = segmentar(Z, st.session_state.k)
K = st.session_state.k
textos_seg, medias_z = pr.describir_segmentos(w, etiquetas, list(variables))
nombres_seg = {s: f"Segmento {s+1}" for s in range(K)}
d_seg = d.assign(segmento=[nombres_seg[e] for e in etiquetas])
color_map = {nombres_seg[s]: PALETA[s % len(PALETA)] for s in range(K)}
pct_info = varianza["% acumulado"].iloc[n_comp - 1]


# ---------------------------------------------------------------------------
# 1. CONTEXTO
# ---------------------------------------------------------------------------
if pagina == "1. Contexto":
    st.title("¿Qué perfiles financieros tienen las empresas medianas y grandes de Colombia?")
    st.markdown(
        """
        Detrás de cada ranking de "las empresas más grandes" hay compañías muy distintas: comercializadoras que
        mueven mucho volumen con márgenes estrechos, empresas de servicios muy rentables con poca deuda,
        compañías que crecieron rápido apalancadas en terceros y otras que, pese a su tamaño, apenas generan utilidad.
        Clasificarlas solo por **sector** o por **ingresos** esconde esas diferencias.

        **Preguntas que guían el ejercicio**
        1. ¿Cuáles son las dimensiones de fondo que explican la mayor parte de las diferencias financieras entre
           empresas (rentabilidad, apalancamiento, eficiencia, crecimiento, tamaño)?
        2. ¿Existen grupos de empresas con un perfil financiero parecido? ¿Cómo se describen?
        3. ¿Esos grupos coinciden con el macrosector o lo atraviesan?
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Empresas en la base", f"{len(crudo):,}".replace(",", "."))
    c2.metric("Empresas analizadas", f"{len(d):,}".replace(",", "."))
    c3.metric("Indicadores en el modelo", len(variables))
    c4.metric("Corte contable", "31-dic-2024")

    st.subheader("Fuente y cobertura")
    st.markdown(
        """
        - **Origen:** Superintendencia de Sociedades, *Informe de las 9.000 empresas siguientes más grandes del país*
          (publicado en agosto de 2025). Archivo Excel público descargado del portal de Supersociedades e incluido en
          el repositorio (`data/`).
        - **Qué representa cada fila:** una sociedad (identificada por NIT) que ocupa los puestos **1.001 a 10.000** del
          ranking nacional por ingresos operacionales. Las 1.000 más grandes **no** están en esta base.
        - **Cobertura:** sociedades vigiladas por Supersociedades y por otras superintendencias (Salud, Transporte,
          Vigilancia, Servicios Públicos, Financiera), con estados financieros bajo NIIF. Cifras en **miles de pesos**,
          años **2024 y 2023**.
        - **Variables disponibles:** ingresos operacionales, ganancia (pérdida), activos, pasivos y patrimonio de cada año,
          más región, departamento, ciudad, código CIIU y macrosector.
        """
    )

    st.subheader("Indicadores construidos")
    tabla = pd.DataFrame(
        [(i["nombre"], i["formula"], i["lectura"], "Sí" if k in variables else "No") for k, i in pr.INDICADORES.items()],
        columns=["Indicador", "Cálculo", "Qué mide", "¿En el modelo?"],
    )
    st.dataframe(tabla, hide_index=True, width="stretch")
    nota(
        "Se trabaja con <b>razones financieras</b> y no con los montos en pesos: los montos están dominados por el "
        "tamaño (una empresa con más activos también tiene más pasivos, ingresos y utilidad), así que un ACP sobre "
        "ellos solo encontraría la dimensión 'tamaño'. Las razones permiten comparar empresas de escalas distintas; "
        "el tamaño entra de forma explícita y controlada como log10(activos)."
    )

    st.subheader("¿Por qué ACP y K-Means?")
    st.markdown(
        """
        - Los indicadores financieros están **correlacionados** entre sí (margen y ROA se mueven juntos; el endeudamiento
          suele ir en contra de la rentabilidad). El **ACP** convierte esas variables correlacionadas en pocos ejes no
          correlacionados y permite ver qué combinaciones de indicadores explican la mayor parte de la variación.
          Eso responde la pregunta 1.
        - No existe una clasificación "verdadera" de perfiles financieros con la cual entrenar un modelo supervisado.
          **K-Means** busca grupos de empresas cercanas entre sí en el espacio de los componentes, sin etiquetas previas.
          Eso responde las preguntas 2 y 3.
        - Hacer K-Means sobre los componentes (y no sobre las variables originales) evita que dos indicadores casi
          redundantes, como margen y ROA, pesen doble en la distancia, y elimina parte del ruido de los últimos componentes.
        """
    )

    st.subheader("Limitaciones de la fuente")
    st.markdown(
        """
        - **Solo cinco cuentas por año.** No hay activo corriente, inventarios, EBITDA ni gastos financieros, así que no
          se pueden calcular liquidez, cobertura de intereses ni márgenes operativos.
        - **Sesgo de selección.** Solo están las empresas que reportan estados financieros y que entran al ranking por
          ingresos. Las 1.000 más grandes y las pequeñas y medianas no aparecen, así que los resultados no se
          generalizan a todo el tejido empresarial.
        - **Dos años.** El crecimiento depende de un solo cambio anual y puede reflejar fusiones, escisiones o eventos
          puntuales.
        - **Estados individuales vs. separados.** Algunas sociedades reportan estados separados (matrices), lo que puede
          inflar activos o utilidades por inversiones en subordinadas.
        - **Información reportada, no auditada por el autor.** Hay valores extremos (márgenes mayores al 100 %,
          crecimientos de miles por ciento) que se tratan en la etapa de preparación.
        """
    )


# ---------------------------------------------------------------------------
# 2. EXPLORACIÓN
# ---------------------------------------------------------------------------
elif pagina == "2. Exploración":
    st.title("Exploración de la base")
    t1, t2, t3, t4 = st.tabs(["Estructura", "Calidad de datos", "Distribuciones", "Composición"])

    with t1:
        st.markdown(f"La base original tiene **{crudo.shape[0]:,} filas y {crudo.shape[1]} columnas**.".replace(",", "."))
        tipos = pd.DataFrame(
            {
                "Columna": crudo.columns,
                "Tipo": [str(t) for t in crudo.dtypes],
                "Faltantes": crudo.isna().sum().values,
                "Ejemplo": [str(crudo[c].iloc[0])[:40] for c in crudo.columns],
            }
        )
        st.dataframe(tipos, hide_index=True, width="stretch")
        nota(
            "Las columnas <code>ranking</code>, <code>nit</code> y <code>razon_social</code> son identificadores y "
            "<b>no entran al modelo</b>. Región, sector, supervisor y marco normativo son categóricas: se usan para "
            "<b>interpretar</b> los grupos, no para formarlos."
        )
        st.subheader("Estadísticas descriptivas de las cuentas (miles de millones de pesos)")
        desc = (crudo[pr.COLUMNAS_MONETARIAS] / 1e6).describe().T
        st.dataframe(desc.style.format("{:,.2f}"), width="stretch")
        st.subheader("Vista de los datos")
        st.dataframe(crudo.head(50), width="stretch")

    with t2:
        st.subheader("Diagnóstico de calidad")
        st.dataframe(pr.diagnostico_calidad(crudo), hide_index=True, width="stretch")
        nota(
            "No hay NIT duplicados y el balance cuadra (activo = pasivo + patrimonio) en todas las empresas. "
            "Los problemas reales son: empresas sin información de 2023 (no se puede calcular su crecimiento), "
            "unas pocas con ingresos 2023 en cero o negativos, y empresas con patrimonio negativo, que son técnicamente "
            "insolventes y producen razones extremas."
        )

    with t3:
        st.subheader("Indicadores antes de cualquier recorte")
        v_all = pr.construir_indicadores(pr.limpiar(crudo)[0])
        var_sel = st.selectbox("Indicador", list(pr.INDICADORES.keys()), format_func=pr.nombre)
        desc_i = v_all[var_sel].describe(percentiles=[0.01, 0.025, 0.25, 0.5, 0.75, 0.975, 0.99])
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.histogram(v_all, x=var_sel, nbins=120, color_discrete_sequence=["#0F766E"])
            fig.update_layout(
                title=f"Distribución de {pr.nombre(var_sel)} (sin recortar)",
                xaxis_title=pr.nombre(var_sel), yaxis_title="Número de empresas", height=420,
            )
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.dataframe(desc_i.to_frame("valor").style.format("{:,.3f}"), width="stretch")
        nota(
            "Casi todos los indicadores tienen <b>colas muy largas</b>: por ejemplo, unas pocas empresas multiplicaron "
            "sus ingresos por cientos o tienen márgenes superiores al 100 % por ingresos no operacionales. "
            "Como el ACP y K-Means se basan en varianzas y distancias, unos pocos extremos pueden dominar los resultados. "
            "Por eso, en la siguiente etapa se aplica una winsorización.",
            "alerta",
        )

    with t4:
        c1, c2 = st.columns(2)
        with c1:
            s = crudo["macrosector"].value_counts().reset_index()
            fig = px.bar(s, x="count", y="macrosector", orientation="h", color_discrete_sequence=["#0F766E"])
            fig.update_layout(title="Empresas por macrosector", xaxis_title="Número de empresas", yaxis_title="", height=380)
            st.plotly_chart(fig, width="stretch")
        with c2:
            s = crudo["region"].value_counts().reset_index()
            fig = px.bar(s, x="count", y="region", orientation="h", color_discrete_sequence=["#C2410C"])
            fig.update_layout(title="Empresas por región", xaxis_title="Número de empresas", yaxis_title="", height=380)
            st.plotly_chart(fig, width="stretch")
        st.caption("Comercio y servicios concentran cerca de dos tercios de las empresas; Bogotá-Cundinamarca, casi la mitad.")


# ---------------------------------------------------------------------------
# 3. PREPARACIÓN
# ---------------------------------------------------------------------------
elif pagina == "3. Preparación":
    st.title("Preparación de los datos")
    st.markdown("Las decisiones de esta página se controlan desde la **barra lateral** y se recalculan en vivo.")

    st.subheader("Paso 1 · Exclusión de registros")
    st.dataframe(bitacora, hide_index=True, width="stretch")
    st.markdown(f"Quedan **{len(d):,} empresas** para el análisis.".replace(",", "."))

    st.subheader("Paso 2 · Construcción de indicadores")
    st.markdown(
        "Con las cuentas de 2024 y 2023 se calculan las razones financieras. Las variables seleccionadas para el modelo son: "
        + ", ".join(f"**{pr.nombre(c)}**" for c in variables) + "."
    )

    st.subheader(f"Paso 3 · Winsorización al {q*100:.1f} % en cada cola")
    lim = limites.copy()
    lim["variable"] = lim["variable"].map(pr.nombre)
    st.dataframe(
        lim.style.format({"limite_inferior": "{:,.3f}", "limite_superior": "{:,.3f}"}),
        hide_index=True, width="stretch",
    )
    nota(
        "<b>¿Por qué winsorizar y no eliminar?</b> Eliminar a toda empresa con algún indicador extremo dejaría por fuera "
        "a muchas compañías que son normales en los demás indicadores. La winsorización conserva a todas las empresas, "
        "pero reemplaza los valores más extremos de cada indicador por el percentil límite. Así, una empresa que creció "
        "un 3.000 % queda como 'de las que más crecieron', sin arrastrar por sí sola la media y la varianza. "
        "Con 0 % en la barra lateral puedes ver qué pasa sin este tratamiento: aparecen grupos de unas pocas empresas atípicas."
    )
    var_box = st.selectbox("Comparar antes / después", list(variables), format_func=pr.nombre)
    comp = pd.concat(
        [
            pd.DataFrame({"valor": v[var_box], "versión": "Original"}),
            pd.DataFrame({"valor": w[var_box], "versión": "Winsorizada"}),
        ]
    )
    fig = px.box(comp, x="versión", y="valor", color="versión", color_discrete_sequence=["#94A3B8", "#0F766E"], points=False)
    fig.update_layout(title=f"{pr.nombre(var_box)}: efecto del recorte de extremos", yaxis_title=pr.nombre(var_box), xaxis_title="", showlegend=False, height=420)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Paso 4 · Relaciones entre indicadores")
    corr = w.corr()
    corr.index = corr.columns = [pr.nombre(c) for c in corr.columns]
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(title="Matriz de correlaciones de Pearson (datos winsorizados)", height=560)
    st.plotly_chart(fig, width="stretch")
    pares = (
        corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().rename("r").reset_index()
        .assign(abs_r=lambda x: x["r"].abs()).sort_values("abs_r", ascending=False).head(4)
    )
    st.markdown(
        "Correlaciones más fuertes: "
        + "; ".join(f"**{a}–{b}** (r = {r:.2f})" for a, b, r, _ in pares.itertuples(index=False))
        + ". La existencia de correlaciones moderadas justifica el ACP: hay información compartida que puede resumirse."
    )

    st.subheader("Paso 5 · Estandarización")
    st.markdown(
        "Los indicadores están en escalas muy distintas (el endeudamiento varía entre 0 y 1; la rotación, entre 0 y 10). "
        "Sin estandarizar, el ACP y las distancias de K-Means quedarían dominados por las variables de mayor varianza. "
        "Por eso cada indicador se transforma a puntaje z (media 0, desviación 1) con `StandardScaler`."
    )
    st.dataframe(
        pd.DataFrame(X, columns=[pr.nombre(c) for c in variables]).describe().loc[["mean", "std", "min", "max"]].style.format("{:.2f}"),
        width="stretch",
    )


# ---------------------------------------------------------------------------
# 4. ACP
# ---------------------------------------------------------------------------
elif pagina == "4. ACP":
    st.title("Análisis de Componentes Principales")
    t1, t2, t3 = st.tabs(["Varianza explicada", "Cargas e interpretación", "Empresas en el plano factorial"])

    with t1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(
                varianza.style.format({"Valor propio": "{:.3f}", "% varianza": "{:.1f}", "% acumulado": "{:.1f}"}),
                hide_index=True, width="stretch",
            )
        with c2:
            fig = go.Figure()
            fig.add_bar(x=varianza["Componente"], y=varianza["% varianza"], name="% de varianza", marker_color="#0F766E")
            fig.add_scatter(x=varianza["Componente"], y=varianza["% acumulado"], name="% acumulado", mode="lines+markers", line_color="#C2410C")
            fig.update_layout(title="Varianza explicada por componente", yaxis_title="% de la varianza total", height=380, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
        fig = px.line(varianza, x="Componente", y="Valor propio", markers=True, color_discrete_sequence=["#4338CA"])
        fig.add_hline(y=1, line_dash="dash", line_color="#C2410C", annotation_text="Criterio de Kaiser (valor propio = 1)")
        fig.update_layout(title="Gráfico de sedimentación (scree plot)", yaxis_title="Valor propio", height=380)
        st.plotly_chart(fig, width="stretch")

        st.subheader("¿Cuántos componentes conservar?")
        st.markdown(
            f"- **Criterio de Kaiser** (valor propio > 1, es decir, el componente resume más que una variable original): "
            f"**{sug['kaiser']}** componentes.\n"
            f"- **Varianza acumulada ≥ 70 %:** **{sug['umbral']}** componentes.\n"
            f"- **Codo del gráfico de sedimentación:** obsérvalo arriba; la caída se aplana después de los primeros componentes."
        )
        nuevo = st.slider(
            "Componentes que se usarán para K-Means", 2, len(variables), n_comp,
            help="Por defecto se usa el criterio de Kaiser.",
        )
        if nuevo != st.session_state.n_comp:
            st.session_state.n_comp = nuevo
            st.rerun()
        nota(
            f"Con <b>{n_comp} componentes</b> se conserva el <b>{pct_info:.1f} %</b> de la variación de los "
            f"{len(variables)} indicadores estandarizados. El resto ({100-pct_info:.1f} %) se reparte en componentes "
            "que individualmente aportan menos que una sola variable original y que, en buena medida, recogen ruido "
            "o rasgos particulares de cada indicador."
        )

    with t2:
        n_show = min(max(n_comp, 3), len(variables))
        cg = cargas.iloc[:, :n_show].copy()
        cg.index = [pr.nombre(c) for c in cg.index]
        fig = px.imshow(cg, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(title="Cargas (pesos de cada indicador en cada componente)", height=480)
        st.plotly_chart(fig, width="stretch")
        st.subheader("Lectura de los componentes")
        for j in range(n_show):
            col = cargas.iloc[:, j]
            pos = col[col > 0.3].sort_values(ascending=False)
            neg = col[col < -0.3].sort_values()
            txt_pos = ", ".join(f"{pr.nombre(i)} ({x:+.2f})" for i, x in pos.items()) or "—"
            txt_neg = ", ".join(f"{pr.nombre(i)} ({x:+.2f})" for i, x in neg.items()) or "—"
            st.markdown(
                f"**CP{j+1}** ({varianza['% varianza'].iloc[j]:.1f} % de la varianza): valores altos asociados a "
                f"{txt_pos}; valores bajos asociados a {txt_neg}."
            )
        nota(
            "Se resaltan las cargas con valor absoluto mayor a 0,3. El signo de un componente es arbitrario: "
            "lo que importa es qué variables pesan juntas y cuáles en sentido contrario. "
            "Con la configuración por defecto, la lectura es: <b>CP1 = rentabilidad</b> (margen y ROA altos, poca "
            "deuda); <b>CP2 = intensidad de activos y escala</b> (empresas grandes con balance pesado frente a "
            "comercializadoras de alta rotación); <b>CP3 = expansión apalancada</b> (crecimiento de ingresos y activos "
            "acompañado de mayor endeudamiento)."
        )

    with t3:
        opciones = [f"CP{i+1}" for i in range(len(variables))]
        c1, c2, c3 = st.columns(3)
        ex = c1.selectbox("Eje X", opciones, index=0)
        ey = c2.selectbox("Eje Y", opciones, index=1)
        color_por = c3.selectbox("Colorear por", ["macrosector", "region", "supervisor"])
        ix, iy = opciones.index(ex), opciones.index(ey)
        pdf = pd.DataFrame({ex: puntajes[:, ix], ey: puntajes[:, iy], "Empresa": d["razon_social"], color_por: d[color_por]})
        fig = px.scatter(pdf, x=ex, y=ey, color=color_por, hover_name="Empresa", opacity=0.55, color_discrete_sequence=PALETA)
        esc = np.abs(puntajes[:, [ix, iy]]).max() * 0.8
        for var in variables:
            lx, ly = cargas.loc[var, ex] * esc, cargas.loc[var, ey] * esc
            fig.add_annotation(x=lx, y=ly, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#111827")
            fig.add_annotation(x=lx * 1.1, y=ly * 1.1, text=pr.nombre(var), showarrow=False, font=dict(size=11, color="#111827"))
        info = varianza["% varianza"].iloc[ix] + varianza["% varianza"].iloc[iy]
        fig.update_layout(
            title=f"Biplot: empresas y variables en {ex} vs {ey} (resume el {info:.1f} % de la varianza)",
            xaxis_title=f"{ex} ({varianza['% varianza'].iloc[ix]:.1f} %)",
            yaxis_title=f"{ey} ({varianza['% varianza'].iloc[iy]:.1f} %)", height=620,
        )
        st.plotly_chart(fig, width="stretch")
        nota(
            f"Este plano muestra solo el <b>{info:.1f} %</b> de la información total. Empresas que se ven cercanas "
            "aquí pueden estar lejos en los componentes que no se dibujan. Las flechas indican hacia dónde crece cada "
            "indicador. Que los colores por sector se mezclen sugiere que <b>el sector por sí solo no determina el "
            "perfil financiero</b>."
        )


# ---------------------------------------------------------------------------
# 5. K-MEANS
# ---------------------------------------------------------------------------
elif pagina == "5. K-Means":
    st.title("Segmentación con K-Means")
    st.markdown(
        f"K-Means se aplica sobre los **{n_comp} primeros componentes principales** ({pct_info:.1f} % de la varianza), "
        "definidos en la página de ACP."
    )
    t1, t2, t3, t4 = st.tabs(["Elección de K", "Grupos en el plano", "Perfil de los segmentos", "Buscar una empresa"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(curva, x="K", y="Inercia", markers=True, color_discrete_sequence=["#0F766E"])
            fig.update_layout(title="Método del codo", yaxis_title="Inercia (suma de distancias² intra-grupo)", height=380)
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.line(curva, x="K", y="Silueta", markers=True, color_discrete_sequence=["#C2410C"])
            fig.add_vline(x=k_sugerido, line_dash="dash", line_color="#475569", annotation_text=f"Máx. silueta: K={k_sugerido}")
            fig.update_layout(title="Coeficiente de silueta promedio", yaxis_title="Silueta", height=380)
            st.plotly_chart(fig, width="stretch")
        st.dataframe(curva.style.format({"Inercia": "{:,.0f}", "Silueta": "{:.3f}"}), hide_index=True, width="stretch")
        nuevo_k = st.slider("Número de segmentos (K)", 2, 10, K)
        if nuevo_k != st.session_state.k:
            st.session_state.k = nuevo_k
            st.rerun()
        nota(
            f"El codo no es nítido: la inercia cae de forma gradual, algo típico de datos financieros que forman un continuo "
            f"y no nubes separadas. Por eso el criterio principal es la <b>silueta</b>, que alcanza su máximo en "
            f"<b>K = {k_sugerido}</b> ({curva['Silueta'].max():.3f}). También se revisa que ningún grupo sea "
            "diminuto (columna 'Tamaño del grupo más pequeño'), porque un segmento de pocas empresas suele ser un "
            "conjunto de atípicos y no un perfil. "
            "Una silueta entre 0,25 y 0,30 indica una estructura <b>débil o moderada</b>: los grupos existen, pero con "
            "fronteras difusas."
        )
        st.caption("La silueta se calcula sobre una muestra aleatoria fija de 4.000 empresas para que la app responda rápido.")

        sil = pr.silueta_por_observacion(Z, etiquetas)
        sil["Segmento"] = sil["segmento"].map(nombres_seg)
        sil = sil.sort_values(["segmento", "silueta"], ascending=[True, False]).reset_index(drop=True)
        sil["pos"] = range(len(sil))
        fig = px.bar(sil, x="silueta", y="pos", color="Segmento", orientation="h", color_discrete_map=color_map)
        fig.update_traces(marker_line_width=0)
        fig.add_vline(x=sil["silueta"].mean(), line_dash="dash", annotation_text=f"Promedio {sil['silueta'].mean():.2f}")
        fig.update_layout(title=f"Silueta de cada empresa con K = {K} (muestra de 3.000)", yaxis=dict(showticklabels=False, title=""),
                          xaxis_title="Coeficiente de silueta", height=520, bargap=0)
        st.plotly_chart(fig, width="stretch")
        neg = (sil["silueta"] < 0).mean() * 100
        baja = (sil["silueta"] < 0.1).mean() * 100
        st.markdown(
            f"El **{baja:.0f} %** de las empresas tiene una silueta menor a 0,1 (están en la frontera entre dos segmentos) y "
            f"el **{neg:.0f} %** tiene una silueta negativa (quedarían casi igual de bien en otro grupo)."
        )

    with t2:
        opciones = [f"CP{i+1}" for i in range(n_comp)]
        c1, c2 = st.columns(2)
        ex = c1.selectbox("Eje X", opciones, index=0, key="kx")
        ey = c2.selectbox("Eje Y", opciones, index=1, key="ky")
        ix, iy = opciones.index(ex), opciones.index(ey)
        pdf = pd.DataFrame({ex: Z[:, ix], ey: Z[:, iy], "Segmento": d_seg["segmento"], "Empresa": d["razon_social"], "Sector": d["macrosector"]})
        fig = px.scatter(pdf, x=ex, y=ey, color="Segmento", hover_name="Empresa", hover_data=["Sector"], opacity=0.55,
                         color_discrete_map=color_map, category_orders={"Segmento": list(nombres_seg.values())})
        fig.add_scatter(x=centros[:, ix], y=centros[:, iy], mode="markers+text", text=list(nombres_seg.values()),
                        textposition="top center", marker=dict(symbol="x", size=16, color="black"), name="Centroides")
        info = varianza["% varianza"].iloc[ix] + varianza["% varianza"].iloc[iy]
        fig.update_layout(title=f"Segmentos en {ex} vs {ey} ({info:.1f} % de la varianza)", height=600,
                          xaxis_title=f"{ex} ({varianza['% varianza'].iloc[ix]:.1f} %)", yaxis_title=f"{ey} ({varianza['% varianza'].iloc[iy]:.1f} %)")
        st.plotly_chart(fig, width="stretch")
        if n_comp >= 3:
            with st.expander("Ver en 3D (CP1, CP2, CP3)"):
                p3 = pd.DataFrame({"CP1": Z[:, 0], "CP2": Z[:, 1], "CP3": Z[:, 2], "Segmento": d_seg["segmento"], "Empresa": d["razon_social"]})
                fig = px.scatter_3d(p3.sample(min(3000, len(p3)), random_state=1), x="CP1", y="CP2", z="CP3", color="Segmento",
                                    hover_name="Empresa", opacity=0.6, color_discrete_map=color_map)
                fig.update_traces(marker_size=3)
                fig.update_layout(height=600, title="Segmentos en los tres primeros componentes (muestra de 3.000)")
                st.plotly_chart(fig, width="stretch")
        nota("Los números de segmento son solo nombres: el Segmento 1 es el más numeroso, pero no es 'mejor' ni 'primero' en ningún sentido financiero.")

    with t3:
        st.subheader("Perfil con los indicadores originales")
        perfil = pr.perfil_segmentos(v, etiquetas)
        perfil.index = [nombres_seg[i] for i in perfil.index]
        mostrar = pd.DataFrame(index=perfil.index)
        mostrar["Empresas"] = perfil["n_empresas"].astype(int)
        mostrar["% del total"] = perfil["% del total"].map(lambda x: f"{x:.1f} %")
        for c in pr.INDICADORES:
            mostrar[pr.nombre(c)] = perfil[c].map(lambda x, f=pr.INDICADORES[c]["formato"]: fmt(x, f))
        mostrar["% con pérdidas"] = (d["ganancia_2024"] < 0).groupby(etiquetas).mean().values * 100
        mostrar["% con pérdidas"] = mostrar["% con pérdidas"].map(lambda x: f"{x:.0f} %")
        mostrar["Ingresos medianos ($ mil millones)"] = (d["ingresos_2024"] / 1e6).groupby(etiquetas).median().values.round(1)
        mostrar["Rasgo distintivo"] = [textos_seg[i] for i in range(K)]
        st.dataframe(mostrar.T.astype(str), width="stretch")
        med_total = v.median()
        st.caption(
            "Medianas de cada indicador sin winsorizar (la mediana es robusta a extremos). Referencia de toda la muestra: "
            + "; ".join(f"{pr.nombre(c)} {fmt(med_total[c], pr.INDICADORES[c]['formato'])}" for c in variables)
        )

        mz = medias_z.copy()
        mz.index = [nombres_seg[i] for i in mz.index]
        mz.columns = [pr.nombre(c) for c in mz.columns]
        fig = px.imshow(mz, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1.5, zmax=1.5, aspect="auto")
        fig.update_layout(title="¿En qué se distingue cada segmento? (promedio estandarizado: 0 = promedio general)", height=380)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Composición por sector y región")
        c1, c2 = st.columns(2)
        with c1:
            ct = pd.crosstab(d_seg["segmento"], d_seg["macrosector"], normalize="index").mul(100).reset_index().melt(id_vars="segmento", var_name="Macrosector", value_name="%")
            fig = px.bar(ct, x="%", y="segmento", color="Macrosector", orientation="h", color_discrete_sequence=PALETA)
            fig.update_layout(title="Macrosector dentro de cada segmento", xaxis_title="% de las empresas del segmento", yaxis_title="", height=380, legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, width="stretch")
        with c2:
            ct = pd.crosstab(d_seg["segmento"], d_seg["region"], normalize="index").mul(100).reset_index().melt(id_vars="segmento", var_name="Región", value_name="%")
            fig = px.bar(ct, x="%", y="segmento", color="Región", orientation="h", color_discrete_sequence=PALETA)
            fig.update_layout(title="Región dentro de cada segmento", xaxis_title="% de las empresas del segmento", yaxis_title="", height=380, legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, width="stretch")

        var_dist = st.selectbox("Distribución de un indicador por segmento", list(variables), format_func=pr.nombre)
        fig = px.box(pd.DataFrame({"valor": w[var_dist], "Segmento": d_seg["segmento"]}), x="Segmento", y="valor", color="Segmento",
                     color_discrete_map=color_map, category_orders={"Segmento": list(nombres_seg.values())}, points=False)
        fig.update_layout(title=f"{pr.nombre(var_dist)} por segmento (valores winsorizados)", yaxis_title=pr.nombre(var_dist), showlegend=False, height=420)
        st.plotly_chart(fig, width="stretch")

    with t4:
        texto = st.text_input("Escribe parte del nombre de una empresa", "")
        res = d_seg
        if texto:
            res = d_seg[d_seg["razon_social"].str.contains(texto, case=False, na=False)]
        tabla = res[["razon_social", "nit", "macrosector", "region", "ciudad", "segmento"]].join(v[list(variables)].round(3))
        tabla.columns = [pr.nombre(c) if c in pr.INDICADORES else c for c in tabla.columns]
        st.dataframe(tabla.head(300), hide_index=True, width="stretch")
        st.download_button(
            "Descargar resultados (CSV)",
            d_seg.join(v).join(pd.DataFrame(Z, columns=[f"CP{i+1}" for i in range(n_comp)])).to_csv(index=False).encode("utf-8"),
            file_name="segmentacion_empresas.csv", mime="text/csv",
        )


# ---------------------------------------------------------------------------
# 6. CONCLUSIONES
# ---------------------------------------------------------------------------
elif pagina == "6. Conclusiones":
    st.title("Discusión de resultados y conclusiones")
    sil_k = float(curva.loc[curva["K"] == K, "Silueta"].iloc[0])
    perfil = pr.perfil_segmentos(v, etiquetas)
    perd = (d["ganancia_2024"] < 0).groupby(etiquetas).mean() * 100

    st.subheader("Resultados con la configuración actual")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Empresas", f"{len(d):,}".replace(",", "."))
    c2.metric("Componentes retenidos", n_comp, f"{pct_info:.1f} % de la varianza", delta_color="off")
    c3.metric("Segmentos (K)", K)
    c4.metric("Silueta promedio", f"{sil_k:.3f}")

    st.markdown("**Segmentos encontrados** (medianas de cada grupo):")
    for s in range(K):
        p = perfil.loc[s]
        sector_top = d.loc[etiquetas == s, "macrosector"].value_counts(normalize=True)
        st.markdown(
            f"- **{nombres_seg[s]}** · {int(p['n_empresas']):,} empresas ({p['% del total']:.0f} %) · *{textos_seg[s]}*. ".replace(",", ".")
            + f"Margen {fmt(p['margen_neto'],'pct')}, ROA {fmt(p['roa'],'pct')}, endeudamiento {fmt(p['endeudamiento'],'pct')}, "
            + f"rotación {fmt(p['rotacion_activos'],'num')}, crecimiento de ingresos {fmt(p['crec_ingresos'],'pct')}, "
            + f"crecimiento de activos {fmt(p['crec_activos'],'pct')}; {perd.loc[s]:.0f} % con pérdidas; "
            + f"sector más frecuente: {sector_top.index[0].title()} ({sector_top.iloc[0]*100:.0f} %)."
        )

    st.subheader("¿Qué aprendimos de los datos?")
    st.markdown(
        """
        *Esta lectura corresponde a la configuración por defecto (7 indicadores, winsorización del 2,5 %, todos los
        sectores, 3 componentes y K = 4). Si cambias las decisiones en la barra lateral, los números de arriba se
        actualizan, pero este texto no.*

        **1. Las diferencias financieras se resumen en tres dimensiones.** Con tres componentes se conserva cerca del
        75 % de la variación de los siete indicadores.
        - El **primer componente es la rentabilidad**: margen neto y ROA cargan juntos y en sentido opuesto al
          endeudamiento. Las empresas más rentables tienden a ser las menos apalancadas, lo que es coherente con la
          teoría del *pecking order*: quien genera utilidades se financia con recursos propios.
        - El **segundo componente opone escala a rotación**: empresas grandes con balances pesados (infraestructura,
          inmuebles, inversiones en otras sociedades) frente a empresas más pequeñas que venden varias veces su activo
          en el año, típicamente comercializadoras.
        - El **tercer componente es la expansión apalancada**: crecer en ingresos y activos va de la mano con un mayor
          endeudamiento.

        **2. Aparecen cuatro perfiles con sentido económico, aunque con fronteras difusas.**
        - *Baja rentabilidad y alto endeudamiento* (≈45 % de las empresas, el grupo más numeroso): margen neto
          mediano de 1,8 %, ROA de 2 %, pasivos equivalentes al 63 % de los activos y ventas prácticamente estancadas
          (+0,9 %). Una de cada cuatro empresas tuvo pérdidas en 2024. Es el grupo que merece más atención desde el
          punto de vista de riesgo crediticio.
        - *Rentables y conservadoras* (≈24 %): margen de 12 %, ROA de 12 %, endeudamiento de apenas 30 % y
          prácticamente ninguna empresa con pérdidas. Predominan los servicios (42 %), pero hay también comercio y
          manufactura.
        - *Alto volumen, margen estrecho* (≈23 %): las empresas más pequeñas de la muestra, que venden más de tres
          veces su activo en el año, con un margen de 1,9 % pero un ROA aceptable (6,8 %) gracias a la rotación. Son
          sobre todo comercializadoras (54 %): ganan por volumen, no por margen.
        - *Expansión apalancada* (≈8 %): ingresos que crecieron 85 % y activos que crecieron 68 % en un solo año, con
          un endeudamiento de 66 %. Puede reunir proyectos en expansión, adquisiciones o reorganizaciones
          societarias.

        **3. El sector no define el perfil.** Comercio, servicios y manufactura aparecen en los cuatro segmentos, y
        ninguno está dominado por un solo sector: el más homogéneo, el de alto volumen, tiene 54 % de comercio. La
        región tampoco marca diferencias claras: Bogotá-Cundinamarca y Antioquia pesan casi lo mismo en todos los
        grupos. En el biplot coloreado por sector, los colores se superponen. Dos empresas del mismo sector pueden
        tener perfiles financieros opuestos, así que segmentar solo por actividad económica (como hacen muchos
        análisis sectoriales) deja información por fuera.

        **4. Los grupos no son nubes separadas.** La silueta máxima (≈0,29) indica una estructura moderada: los
        indicadores financieros forman un continuo y una parte importante de las empresas queda en la frontera entre
        dos segmentos. Los segmentos sirven como **tipologías de referencia**, no como categorías rígidas. Por eso, una
        empresa con silueta baja debería analizarse caso a caso.
        """
    )

    st.subheader("Utilidad práctica")
    st.markdown(
        """
        - **Entidades financieras:** el segmento de baja rentabilidad y alto endeudamiento sirve como primer filtro de
          vigilancia de cartera comercial. El de expansión apalancada señala clientes con necesidades de financiación
          crecientes.
        - **Supervisores (Supersociedades):** complementar los rankings por ingresos con tipologías financieras permite
          priorizar la supervisión de empresas con señales de deterioro, sin depender del sector.
        - **Inversionistas y analistas:** comparar una empresa con la mediana de su segmento es más informativo que
          compararla con la de su sector.
        """
    )

    st.subheader("Limitaciones que deben tenerse en cuenta")
    st.markdown(
        """
        1. **Indicadores limitados por la fuente.** Sin activo corriente, EBITDA ni gastos financieros, no se mide
           liquidez ni capacidad de pago, que son centrales en un análisis de riesgo. Los segmentos describen
           rentabilidad, estructura y dinámica, pero no solvencia de corto plazo.
        2. **Una sola foto (2024 frente a 2023).** El crecimiento de un año puede reflejar eventos puntuales (fusiones,
           revaluaciones, venta de activos). Con más años, el segmento de expansión podría resultar transitorio.
        3. **Población recortada.** Solo se analizan las empresas en los puestos 1.001 a 10.000 por ingresos. Los
           resultados no aplican a las 1.000 más grandes ni a las pymes.
        4. **Sensibilidad a las decisiones de preparación.** Sin winsorización, K-Means aísla pequeños grupos de
           atípicos. Con otros indicadores o con otro porcentaje de recorte, los segmentos cambian. En la barra lateral
           se puede comprobar qué tan robusta es cada conclusión.
        5. **K-Means supone grupos esféricos de tamaño similar,** algo que los datos financieros no cumplen
           estrictamente. Métodos como modelos de mezcla gaussiana o clustering jerárquico serían una extensión natural.
        """
    )
