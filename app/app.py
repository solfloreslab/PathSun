"""PathSun - aplicacion Streamlit (landing).

Ejecutar desde el directorio raiz del proyecto:
    streamlit run app/app.py
"""

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
META_PATH = MODELS_DIR / "pathsun_meta.json"
ASSETS_DIR = PROJECT_ROOT / "app" / "assets"
LOGO_HORIZONTAL = ASSETS_DIR / "logo_horizontal_trim.png"
FAVICON = ASSETS_DIR / "favicon.png"


st.set_page_config(
    page_title="PathSun",
    page_icon=str(FAVICON),
    layout="wide",
    initial_sidebar_state="expanded",
)

from _theme import apply_pathsun_theme
apply_pathsun_theme()


# Bloque educativo - las 5 clases (arranca directo, el logo vive en el header arriba)
st.markdown(
    "PathSun clasifica células escamosas cervicales de citología **Papanicolaou** "
    "en **cinco categorías morfológicas** del sistema Bethesda. La app combina "
    "el modelo (red **VGG16** con transfer learning sobre el dataset **SIPaKMeD**, "
    "Plissiti et al. 2018) con un modo docente para aprender citología "
    "interpretando lo que el modelo ve."
)

# Galeria de las 5 clases: imagen ejemplo + nombre + grupo + rasgo clave
from PIL import Image as _PILImage
CASOS_DIR = PROJECT_ROOT / "app" / "casos_docente"


def _center_crop_square(img):
    """Recorta al cuadrado (descarta bordes)."""
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _pad_to_square(img, fill=(252, 251, 254)):
    """Anade padding para hacer la imagen cuadrada SIN recortar la celula.
    El relleno coincide con el fondo de la card (#FCFBFE)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = _PILImage.new("RGB", (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


EJEMPLOS_5_CLASES = [
    {
        "img": "im_Superficial-Intermediate_025_01.bmp",
        "nombre": "Superficial-Intermedia",
        "grupo": "Normal",
        "grupo_color": "#2e7d32",
        "rasgo": "Eosinófila · núcleo picnótico",
    },
    {
        "img": "im_Parabasal_054_03.bmp",
        "nombre": "Parabasal",
        "grupo": "Normal",
        "grupo_color": "#2e7d32",
        "rasgo": "Basófila · núcleo regular",
    },
    {
        "img": "im_Metaplastic_261_02.bmp",
        "nombre": "Metaplásica",
        "grupo": "Benigna",
        "grupo_color": "#f9a825",
        "rasgo": "Zona de transformación",
    },
    {
        "img": "im_Koilocytotic_086_04.bmp",
        "nombre": "Koilocítica",
        "grupo": "Anormal (LSIL)",
        "grupo_color": "#c62828",
        "rasgo": "Halo + atipia · HPV",
    },
    {
        "img": "im_Dyskeratotic_138_05.bmp",
        "nombre": "Disqueratósica",
        "grupo": "Anormal (HSIL)",
        "grupo_color": "#c62828",
        "rasgo": "Queratinización anómala",
    },
]

def _render_ejemplo_card(ex):
    """Card de ejemplo vertical: imagen arriba (cuadrada padded) + texto debajo.
    Pad-to-square garantiza que todas las imagenes tengan misma dimension
    SIN recortar las celulas."""
    with st.container(border=True):
        img_path = CASOS_DIR / ex["img"]
        if img_path.exists():
            pil = _PILImage.open(img_path)
            pil_padded = _pad_to_square(pil)
            st.image(pil_padded, use_container_width=True)
        st.markdown(
            f"**{ex['nombre']}**  \n"
            f"<span style='color:{ex['grupo_color']}; font-weight:600; font-size:0.9rem;'>"
            f"{ex['grupo']}</span>",
            unsafe_allow_html=True,
        )
        st.caption(ex["rasgo"])


st.markdown("##### Las 5 clases morfológicas")

# 5 cards en una sola fila horizontal (imagen arriba + texto debajo)
cols = st.columns(5, gap="small")
for col, ex in zip(cols, EJEMPLOS_5_CLASES):
    with col:
        _render_ejemplo_card(ex)

st.divider()

# Tres tarjetas de navegacion (orden: Modo Docente primero)
col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    with st.container(border=True):
        st.markdown("#### 🎓 Modo Docente")
        st.markdown("30 casos para practicar. Clasificas tú primero, luego ves la solución.")
with col2:
    with st.container(border=True):
        st.markdown("#### 🔬 Predicción")
        st.markdown("Sube una imagen → clase + probabilidades + **Grad-CAM**.")
with col3:
    with st.container(border=True):
        st.markdown("#### ℹ️ Acerca de")
        st.markdown("Dataset, modelo, métricas y referencias.")

st.caption(
    "Herramienta educativa, no de diagnóstico clínico. Las predicciones son "
    "orientativas y deben interpretarse en el contexto morfológico completo. "
    "SolFloresLab · 2026 · MIT License"
)
