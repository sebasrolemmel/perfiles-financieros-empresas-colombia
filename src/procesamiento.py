"""
Funciones de procesamiento y modelado del proyecto.

Todo el cálculo (lectura, limpieza, construcción de indicadores, winsorización,
estandarización, ACP y K-Means) vive aquí para que la aplicación de Streamlit
solo se encargue de mostrar resultados y recoger las decisiones del usuario.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

RUTA_DATOS = Path(__file__).resolve().parents[1] / "data" / "supersociedades_9000_empresas_2024.xlsx"

# Enlace oficial del archivo en el portal de Supersociedades (respaldo si no está en data/)
URL_DATOS = (
    "https://www.supersociedades.gov.co/documents/20122/460525/"
    "Base%209000_empresas_2025%20Periodista.xlsx/7e8159b9-38a6-1f1b-24b1-7889cc25fcad?t=1755257649208"
)

SEMILLA = 42

# Nombres cortos para las columnas originales del archivo de Supersociedades
RENOMBRAR = {
    "RANKING": "ranking",
    "NIT": "nit",
    "RAZON SOCIAL": "razon_social",
    "SUPERVISOR": "supervisor",
    "REGION": "region",
    "DEPARTAMENTO DOMICILIO": "departamento",
    "CIUDAD DOMICILIO": "ciudad",
    "CIIU": "ciiu",
    "MACROSECTOR": "macrosector",
    "INGRESOS OPERACIONALES 2024*": "ingresos_2024",
    "GANANCIA (PERDIDA) 2024": "ganancia_2024",
    "TOTAL ACTIVOS 2024": "activos_2024",
    "TOTAL PASIVOS 2024": "pasivos_2024",
    "TOTAL PATRIMONIO 2024": "patrimonio_2024",
    "INGRESOS OPERACIONALES 2023*": "ingresos_2023",
    "GANANCIA (PERDIDA) 2023": "ganancia_2023",
    "TOTAL ACTIVOS 2023": "activos_2023",
    "TOTAL PASIVOS 2023": "pasivos_2023",
    "TOTAL PATRIMONIO 2023": "patrimonio_2023",
    "Punto de entrada": "marco_normativo",
}

COLUMNAS_MONETARIAS = [
    "ingresos_2024", "ganancia_2024", "activos_2024", "pasivos_2024", "patrimonio_2024",
    "ingresos_2023", "ganancia_2023", "activos_2023", "pasivos_2023", "patrimonio_2023",
]

# Indicadores financieros que se construyen a partir de las cuentas originales
INDICADORES = {
    "margen_neto": {
        "nombre": "Margen neto",
        "formula": "Ganancia 2024 / Ingresos operacionales 2024",
        "lectura": "Cuántos pesos de utilidad quedan por cada peso vendido.",
        "formato": "pct",
    },
    "roa": {
        "nombre": "ROA",
        "formula": "Ganancia 2024 / Activos totales 2024",
        "lectura": "Rentabilidad que generan los activos de la empresa.",
        "formato": "pct",
    },
    "endeudamiento": {
        "nombre": "Endeudamiento",
        "formula": "Pasivos totales 2024 / Activos totales 2024",
        "lectura": "Proporción de los activos financiada con terceros.",
        "formato": "pct",
    },
    "rotacion_activos": {
        "nombre": "Rotación de activos",
        "formula": "Ingresos operacionales 2024 / Activos totales 2024",
        "lectura": "Veces que la empresa 'vende' sus activos en el año (eficiencia / intensidad de capital).",
        "formato": "num",
    },
    "crec_ingresos": {
        "nombre": "Crecimiento de ingresos",
        "formula": "Ingresos 2024 / Ingresos 2023 − 1",
        "lectura": "Variación anual de las ventas.",
        "formato": "pct",
    },
    "crec_activos": {
        "nombre": "Crecimiento de activos",
        "formula": "Activos 2024 / Activos 2023 − 1",
        "lectura": "Expansión (o contracción) del balance.",
        "formato": "pct",
    },
    "tamano": {
        "nombre": "Tamaño (log10 activos)",
        "formula": "log10(Activos totales 2024 en pesos)",
        "lectura": "Escala de la empresa; 10 ≈ $10.000 millones, 11 ≈ $100.000 millones.",
        "formato": "num",
    },
    "var_roa": {
        "nombre": "Cambio en ROA",
        "formula": "ROA 2024 − ROA 2023",
        "lectura": "Si la rentabilidad mejoró o empeoró frente al año anterior.",
        "formato": "pct",
    },
}

INDICADORES_POR_DEFECTO = [
    "margen_neto", "roa", "endeudamiento", "rotacion_activos",
    "crec_ingresos", "crec_activos", "tamano",
]


def nombre(var: str) -> str:
    return INDICADORES.get(var, {}).get("nombre", var)


# ---------------------------------------------------------------------------
# 1. Carga y limpieza
# ---------------------------------------------------------------------------
def cargar_datos(ruta: Path | str = RUTA_DATOS) -> pd.DataFrame:
    """Lee el Excel oficial. Las 4 primeras filas son títulos y notas.
    Si el archivo no está en data/, se descarga del portal de Supersociedades."""
    ruta = Path(ruta)
    if not ruta.exists():
        import urllib.request

        ruta.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL_DATOS, ruta)
    df = pd.read_excel(ruta, header=4)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=RENOMBRAR)
    for c in COLUMNAS_MONETARIAS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["razon_social", "region", "macrosector", "supervisor", "ciudad", "departamento", "ciiu"]:
        df[c] = df[c].astype(str).str.strip()
    df["nit"] = df["nit"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df


def diagnostico_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de problemas de calidad que motivan las decisiones de limpieza."""
    filas = [
        ("Registros en el archivo", len(df)),
        ("NIT duplicados", int(df["nit"].duplicated().sum())),
        ("Empresas sin información 2023 (faltantes)", int(df[COLUMNAS_MONETARIAS[5:]].isna().any(axis=1).sum())),
        ("Ingresos 2023 ≤ 0 (crecimiento no calculable)", int((df["ingresos_2023"] <= 0).sum())),
        ("Activos 2024 ≤ 0", int((df["activos_2024"] <= 0).sum())),
        ("Patrimonio 2024 negativo (pasivos > activos)", int((df["patrimonio_2024"] < 0).sum())),
        ("Empresas con pérdidas en 2024", int((df["ganancia_2024"] < 0).sum())),
        (
            "Descuadres Activo ≠ Pasivo + Patrimonio (> $1 millón)",
            int((df["activos_2024"] - df["pasivos_2024"] - df["patrimonio_2024"]).abs().gt(1000).sum()),
        ),
    ]
    return pd.DataFrame(filas, columns=["Revisión", "Empresas"])


def limpiar(df: pd.DataFrame, excluir_patrimonio_negativo: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Elimina registros que impiden calcular los indicadores. Devuelve (datos, bitácora)."""
    bitacora = []
    n0 = len(df)
    d = df.drop_duplicates(subset="nit").copy()
    bitacora.append(("Duplicados por NIT", n0 - len(d)))

    n = len(d)
    d = d.dropna(subset=COLUMNAS_MONETARIAS)
    bitacora.append(("Sin información 2023 (no se puede calcular crecimiento)", n - len(d)))

    n = len(d)
    d = d[(d["ingresos_2023"] > 0) & (d["activos_2023"] > 0) & (d["activos_2024"] > 0) & (d["ingresos_2024"] > 0)]
    bitacora.append(("Ingresos o activos ≤ 0 (cocientes indefinidos)", n - len(d)))

    if excluir_patrimonio_negativo:
        n = len(d)
        d = d[d["patrimonio_2024"] > 0]
        bitacora.append(("Patrimonio negativo (opción activada)", n - len(d)))

    bit = pd.DataFrame(bitacora, columns=["Regla de exclusión", "Empresas excluidas"])
    return d.reset_index(drop=True), bit


# ---------------------------------------------------------------------------
# 2. Construcción de indicadores
# ---------------------------------------------------------------------------
def construir_indicadores(d: pd.DataFrame) -> pd.DataFrame:
    v = pd.DataFrame(index=d.index)
    v["margen_neto"] = d["ganancia_2024"] / d["ingresos_2024"]
    v["roa"] = d["ganancia_2024"] / d["activos_2024"]
    v["endeudamiento"] = d["pasivos_2024"] / d["activos_2024"]
    v["rotacion_activos"] = d["ingresos_2024"] / d["activos_2024"]
    v["crec_ingresos"] = d["ingresos_2024"] / d["ingresos_2023"] - 1
    v["crec_activos"] = d["activos_2024"] / d["activos_2023"] - 1
    v["tamano"] = np.log10(d["activos_2024"] * 1000)  # el archivo viene en miles de pesos
    v["var_roa"] = v["roa"] - d["ganancia_2023"] / d["activos_2023"]
    return v


def winsorizar(v: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recorta cada variable a los percentiles [q, 1-q]. Devuelve datos y límites usados."""
    w = v.copy()
    limites = []
    for c in w.columns:
        if q > 0:
            lo, hi = w[c].quantile([q, 1 - q])
        else:
            lo, hi = w[c].min(), w[c].max()
        afectadas = int(((w[c] < lo) | (w[c] > hi)).sum())
        w[c] = w[c].clip(lo, hi)
        limites.append((c, lo, hi, afectadas))
    lim = pd.DataFrame(limites, columns=["variable", "limite_inferior", "limite_superior", "valores_recortados"])
    return w, lim


def estandarizar(w: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(w)


# ---------------------------------------------------------------------------
# 3. ACP
# ---------------------------------------------------------------------------
def ajustar_acp(X: np.ndarray, columnas: list[str]) -> dict:
    pca = PCA(random_state=SEMILLA).fit(X)
    k = len(columnas)
    nombres = [f"CP{i+1}" for i in range(k)]
    varianza = pd.DataFrame(
        {
            "Componente": nombres,
            "Valor propio": pca.explained_variance_,
            "% varianza": pca.explained_variance_ratio_ * 100,
            "% acumulado": np.cumsum(pca.explained_variance_ratio_) * 100,
        }
    )
    cargas = pd.DataFrame(pca.components_.T, index=columnas, columns=nombres)
    # Correlación variable-componente (cargas escaladas por la raíz del valor propio)
    correl = cargas * np.sqrt(pca.explained_variance_)
    puntajes = pca.transform(X)
    return {"modelo": pca, "varianza": varianza, "cargas": cargas, "correlaciones": correl, "puntajes": puntajes}


def sugerir_componentes(varianza: pd.DataFrame, umbral: float = 70.0) -> dict:
    kaiser = int((varianza["Valor propio"] > 1).sum())
    acumulado = int((varianza["% acumulado"] < umbral).sum() + 1)
    return {"kaiser": max(kaiser, 1), "umbral": min(acumulado, len(varianza))}


# ---------------------------------------------------------------------------
# 4. K-Means
# ---------------------------------------------------------------------------
def evaluar_k(Z: np.ndarray, k_min: int = 2, k_max: int = 10, muestra_silueta: int = 4000) -> pd.DataFrame:
    """Inercia (codo) y silueta para varios K. La silueta se calcula sobre una muestra
    aleatoria fija para que la app responda rápido (el cálculo exacto es O(n²))."""
    filas = []
    n_muestra = min(muestra_silueta, len(Z))
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(Z)
        try:
            sil = silhouette_score(Z, km.labels_, sample_size=n_muestra, random_state=SEMILLA)
        except ValueError:
            # Ocurre si un grupo es tan pequeño (p. ej. 1 empresa atípica) que la muestra
            # aleatoria solo contiene un grupo: se calcula sobre todos los datos.
            sil = silhouette_score(Z, km.labels_) if len(Z) <= 12000 else np.nan
        filas.append((k, km.inertia_, sil, int(np.bincount(km.labels_).min())))
    return pd.DataFrame(filas, columns=["K", "Inercia", "Silueta", "Tamaño del grupo más pequeño"])


def ajustar_kmeans(Z: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(Z)


def silueta_por_observacion(Z: np.ndarray, etiquetas: np.ndarray, n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(SEMILLA)
    idx = rng.choice(len(Z), size=min(n, len(Z)), replace=False)
    s = silhouette_samples(Z[idx], etiquetas[idx])
    return pd.DataFrame({"idx": idx, "segmento": etiquetas[idx], "silueta": s})


def perfil_segmentos(v: pd.DataFrame, etiquetas: np.ndarray) -> pd.DataFrame:
    """Mediana de cada indicador (sin winsorizar) por segmento: robusta a valores extremos."""
    p = v.groupby(etiquetas).median()
    p["n_empresas"] = np.bincount(etiquetas)
    p["% del total"] = p["n_empresas"] / len(etiquetas) * 100
    return p


def describir_segmentos(w: pd.DataFrame, etiquetas: np.ndarray, columnas: list[str], top: int = 2) -> dict:
    """Etiqueta descriptiva para cada segmento: las variables en que más se aparta
    del promedio general (en desviaciones estándar). No implica ningún orden."""
    z = (w[columnas] - w[columnas].mean()) / w[columnas].std()
    medias = z.groupby(etiquetas).mean()
    textos = {}
    for seg, fila in medias.iterrows():
        orden = fila.abs().sort_values(ascending=False).index[:top]
        partes = [f"{'↑' if fila[c] > 0 else '↓'} {nombre(c).lower()}" for c in orden]
        textos[seg] = " · ".join(partes)
    return textos, medias
