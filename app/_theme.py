"""Tema visual PathSun compartido por todas las paginas.

Centraliza:
- Logo PathSun en la franja blanca superior (header nativo de Streamlit)
- Alertas info recoloreadas a lila #6B5BF6 (paleta PathSun) en vez del celeste
  por defecto
- Padding superior del body para no quedar tapado por el header
- Oculta el logo del sidebar (ya esta en el header)

Cada pagina llama `apply_pathsun_theme()` al inicio.
"""

import base64
from pathlib import Path

import streamlit as st

_ASSETS = Path(__file__).resolve().parent / "assets"
_LOGO_HORIZONTAL = _ASSETS / "logo_horizontal_trim.png"
_FAVICON = _ASSETS / "favicon.png"

# Encode una sola vez al importar el modulo
_LOGO_B64 = base64.b64encode(_LOGO_HORIZONTAL.read_bytes()).decode()

_THEME_CSS = f"""
<style>
/* === Header de Streamlit oculto (recupera espacio) === */
header[data-testid="stHeader"] {{
    display: none !important;
}}
/* Sidebar mas estrecha (recupera espacio) */
[data-testid="stSidebar"] {{
    min-width: 220px !important;
    max-width: 240px !important;
}}
/* Logo del sidebar (st.logo) grande y CENTRADO horizontalmente */
[data-testid="stSidebar"] [data-testid="stLogo"],
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{
    height: auto !important;
    padding: 0.3rem 0.5rem 0.3rem 0.5rem !important;
    margin-top: 0 !important;
    text-align: center !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0 !important;
}}
[data-testid="stSidebar"] [data-testid="stLogo"] img,
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] img,
[data-testid="stSidebar"] header img {{
    width: 230px !important;
    height: auto !important;
    max-width: 230px !important;
    max-height: none !important;
    object-fit: contain !important;
    margin: 0 auto !important;
    display: block !important;
}}

/* === Alertas info en lila PathSun === */
div[data-testid="stAlert"] {{
    background-color: #F0EBFC !important;
    border-left: 4px solid #6B5BF6 !important;
}}

/* === Cards bordered (st.container(border=True)) con estilo PathSun === */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border: 1px solid #E0D6F8 !important;
    border-radius: 0.7rem !important;
    background-color: #FCFBFE !important;
    padding: 0.8rem !important;
    box-shadow: 0 1px 3px rgba(13, 27, 76, 0.04) !important;
    height: 100% !important;
}}
/* Las columnas se estiran a la misma altura (cards igualadas) */
[data-testid="stHorizontalBlock"] {{
    align-items: stretch !important;
}}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    display: flex !important;
    flex-direction: column !important;
    height: auto !important;
}}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
}}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] {{
    flex: 1 !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
}}

/* === Typography refinements (sin linea divisoria) === */
h1, h2, h3, h4, h5, h6 {{
    color: #0D1B4C;
}}

/* === Padding superior del contenido (minimo, sin header) === */
.block-container {{
    padding-top: 0.5rem !important;
    padding-bottom: 1.2rem !important;
    max-width: 1400px;
}}
/* Eliminar margen extra del primer elemento del body */
.block-container > div:first-child,
.block-container [data-testid="stVerticalBlock"]:first-child > div:first-child {{
    margin-top: 0 !important;
}}

/* === Reducir spacing default entre elementos === */
[data-testid="stVerticalBlock"] {{
    gap: 0.35rem;
}}
/* Titulos sin margen exagerado */
h1, h2, h3, h4, h5, h6 {{
    margin-top: 0.2rem !important;
    margin-bottom: 0.4rem !important;
}}
/* Markdown paragraphs mas pegados */
[data-testid="stMarkdownContainer"] p {{
    margin-bottom: 0.4rem !important;
}}
/* Imagenes sin margen extra */
[data-testid="stImage"] {{
    margin-bottom: 0.2rem !important;
}}

/* === Mejorar st.metric === */
[data-testid="stMetricValue"] {{
    color: #6B5BF6;
    font-weight: 700;
}}
[data-testid="stMetricLabel"] {{
    color: #0D1B4C;
    font-weight: 500;
}}

/* === Botones primarios en morado PathSun === */
.stButton > button[kind="primary"] {{
    background-color: #6B5BF6;
    border-color: #6B5BF6;
}}
.stButton > button[kind="primary"]:hover {{
    background-color: #5848D8;
    border-color: #5848D8;
}}

/* === st.success en verde más suave === */
div[data-testid="stAlert"][data-baseweb] svg[color="success"] ~ div,
div[data-testid="stNotification"]:has(svg[color="success"]) {{
    background-color: #E8F5E9 !important;
    border-left-color: #4CAF50 !important;
}}
</style>
"""


def apply_pathsun_theme():
    """Inyecta el CSS de marca PathSun + logo en sidebar.

    Llamar al inicio de cada pagina (tras st.set_page_config).
    """
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
    st.logo(
        str(_LOGO_HORIZONTAL),
        icon_image=str(_FAVICON),
        size="large",
    )
