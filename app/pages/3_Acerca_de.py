"""Pagina 3 - Acerca de PathSun."""

import json
import sys as _sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
META_PATH = PROJECT_ROOT / "models" / "pathsun_meta.json"

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _theme import apply_pathsun_theme
apply_pathsun_theme()

st.title("Acerca de PathSun")

# === Objetivo ===
with st.container(border=True):
    st.markdown("#### 🎯 Objetivo")
    st.markdown(
        "PathSun aplica Deep Learning a la clasificación de células escamosas "
        "cervicales en muestras de citología Papanicolaou. Su finalidad es ser "
        "un recurso pedagógico para formación en anatomía patológica y "
        "citodiagnóstico, **no una herramienta de diagnóstico clínico**."
    )

# === Dataset ===
with st.container(border=True):
    st.markdown("#### 📊 Dataset — SIPaKMeD")
    st.markdown(
        "**SIPaKMeD** es un dataset público de **4 049 células cervicales "
        "aisladas** recortadas de preparaciones Papanicolaou y anotadas por "
        "citopatólogos. Distribuidas en cinco clases morfológicas:\n"
        "- Superficial-Intermedia, Parabasal *(normales)*\n"
        "- Metaplásica *(benigna)*\n"
        "- Koilocítica, Disqueratósica *(anormales)*\n\n"
        "**Cita:** Plissiti, M. E. et al. (2018). SIPaKMeD: a new dataset for "
        "feature and image based classification of normal and pathological "
        "cervical cells in Pap smear images. *IEEE ICIP*, 3144-3148. "
        "DOI: 10.1109/ICIP.2018.8451588"
    )

# === Arquitectura ===
with st.container(border=True):
    st.markdown("#### 🧠 Arquitectura del modelo")
    st.markdown(
        "Protocolo **Kaur et al. (2025)**: VGG16 pre-entrenada en ImageNet como "
        "extractor de características congelado, con cabeza de clasificación "
        "propia (GlobalAveragePooling → Dropout 0.4 → BatchNormalization → "
        "Dense(128, ReLU) → Dropout 0.2 → BatchNormalization → Dense(5, "
        "softmax)).\n\n"
        "Optimizador Adam (lr=1e-3), pérdida `categorical_crossentropy`, "
        "batch 32, hasta 50 épocas con EarlyStopping (paciencia 8) y "
        "ReduceLROnPlateau. Augmentación de datos **geométrica solamente** "
        "(flip, rotación, zoom, traslación). **No se usa augmentación "
        "cromática** porque en citología Papanicolaou la intensidad de tinción "
        "hematoxilina/eosina es una señal diagnóstica.\n\n"
        "**Cita:** Kaur, A. et al. (2025). Comparison of deep transfer learning "
        "models for classification of cervical cancer from pap smear images. "
        "*Scientific Reports*. DOI: 10.1038/s41598-024-74531-0"
    )

# === Métricas ===
with st.container(border=True):
    st.markdown("#### 📈 Métricas del modelo")
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy en test", f"{meta['test_accuracy']*100:.2f}%")
        col2.metric("Pérdida en test", f"{meta['test_loss']:.4f}")
        col3.metric("F1 macro", "0.932")
        st.caption(
            f"Backend Keras: {meta.get('keras_backend', 'n/d')} · "
            f"Keras {meta.get('keras_version', 'n/d')} · "
            f"Sobre 810 imágenes del split test"
        )
    else:
        st.info("Las métricas se mostrarán cuando el modelo entrenado esté disponible.")

# === Marco pedagógico ===
with st.container(border=True):
    st.markdown("#### 🎓 Marco pedagógico del Modo Docente")
    st.markdown(
        "El Modo Docente está inspirado en **CellaVision Proficiency Software**, "
        "herramienta usada en formación de técnicos de laboratorio en hematología "
        "(Krupinski et al. 2016). PathSun replica esa arquitectura para citología "
        "cervical bajo **tres reglas innegociables**:\n\n"
        "1. **Lock-then-reveal:** el alumno clasifica primero, la IA habla después.\n"
        "2. **La IA da material para pensar, no respuestas:** distribución de "
        "probabilidades + Grad-CAM con disclaimer (Selvaraju et al. 2017).\n"
        "3. **El ground truth siempre es humano:** la autoridad de la verdad es el "
        "citopatólogo que anotó SIPaKMeD, no la predicción del modelo.\n\n"
        "Esto distingue *segundo par de ojos* (lo deseable) de *oráculo* (lo que la "
        "evidencia anti-automatización rechaza: Rudolph et al. 2023, Budzyn et al. "
        "2025, Dratsch et al. 2023)."
    )

# === Limitaciones ===
with st.container(border=True):
    st.markdown("#### ⚠️ Limitaciones")
    st.markdown(
        "- El modelo se ha entrenado y evaluado **solo** sobre SIPaKMeD, "
        "recogido en un único laboratorio. Su comportamiento sobre muestras de "
        "otro centro (con distinta tinción, microscopio o protocolo) no está "
        "validado.\n"
        "- La clasificación se realiza sobre células **ya recortadas** por un "
        "experto. No incluye detección automática de células en una preparación "
        "completa.\n"
        "- Las cinco clases representan morfologías canónicas. Casos limítrofes, "
        "artefactos o células en mitosis pueden quedar fuera del rango aprendido.\n"
        "- **PathSun no sustituye al citopatólogo.** Cualquier decisión clínica "
        "debe basarse en la valoración humana de la muestra completa."
    )

# === Referencias ===
with st.container(border=True):
    st.markdown("#### 📚 Referencias")
    st.markdown(
        "- **Plissiti, M. E.** et al. (2018). SIPaKMeD: a new dataset for "
        "feature and image based classification of normal and pathological "
        "cervical cells in Pap smear images. *IEEE ICIP*, 3144-3148. "
        "DOI: 10.1109/ICIP.2018.8451588\n"
        "- **Kaur, A.** et al. (2025). Comparison of deep transfer learning "
        "models for classification of cervical cancer from pap smear images. "
        "*Scientific Reports*. DOI: 10.1038/s41598-024-74531-0\n"
        "- **Simonyan, K. & Zisserman, A.** (2014). Very deep convolutional "
        "networks for large-scale image recognition. arXiv:1409.1556\n"
        "- **Selvaraju, R. R.** et al. (2017). Grad-CAM: visual explanations "
        "from deep networks via gradient-based localization. *ICCV 2017*.\n"
        "- **Krupinski, E. A.** et al. (2016). Using CellaVision Proficiency "
        "Software to train competent hematology students. *BMC Medical "
        "Education*, 16, 212. DOI: 10.1186/s12909-016-0816-9\n"
        "- **Rudolph, J.** et al. (2023). Automation bias in radiology with CADe.\n"
        "- **Budzyn, K.** et al. (2025). Deskilling tras retirada de IA.\n"
        "- **Dratsch, T.** et al. (2023). Efecto de la IA en el razonamiento médico."
    )

st.caption("SolFloresLab · 2026 · MIT License")
