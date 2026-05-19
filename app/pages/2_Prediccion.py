"""Pagina 1 - Prediccion sobre imagen subida por el usuario.

Pipeline (replica el protocolo del modelo entrenado):
  1. Carga del modelo y metadatos desde models/
  2. Lectura de la imagen subida -> RGB 224x224
  3. Preprocesado VGG16 (BGR + resta media ImageNet) - Simonyan & Zisserman 2014
  4. Inferencia -> softmax de 5 clases
  5. Grad-CAM sobre block5_conv3 - Selvaraju et al. 2017 ICCV

Las funciones _build_flat_grad_model, make_gradcam_heatmap y overlay_heatmap
estan reusadas tal cual del notebook canonico PathSun_Final_v2.ipynb (seccion
8.1), donde fueron validadas con la corrida que dio 93.21% de test accuracy.
La version aplanada es necesaria porque Keras 3 no permite construir un Model
cuyas outputs crucen la frontera de un sub-modelo anidado (VGG16 vive dentro
de best_model).
"""

import json
from pathlib import Path

import cv2
import keras
import numpy as np
import pandas as pd
import streamlit as st
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
META_PATH = MODELS_DIR / "pathsun_meta.json"
MODEL_PATH = MODELS_DIR / "pathsun_vgg16_best.keras"

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _theme import apply_pathsun_theme
apply_pathsun_theme()

st.markdown("### 🔬 Predicción sobre imagen")
st.caption(
    "Laboratorio para evaluar el modelo (test acc 93.21%). Para aprender citología, usa el **Modo Docente**."
)

if not META_PATH.exists() or not MODEL_PATH.exists():
    st.error(
        "Modelo no disponible. Coloca `pathsun_vgg16_best.keras` y "
        "`pathsun_meta.json` en `models/`."
    )
    st.stop()


@st.cache_resource(show_spinner="Cargando modelo VGG16...")
def load_model_and_meta():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    model = keras.models.load_model(MODEL_PATH)
    return model, meta


model, meta = load_model_and_meta()

CLASS_NAMES = meta["class_names"]
CLASS_LABELS_ES = meta["class_labels_es"]
CLASS_GROUP = meta["class_group"]

GROUP_COLOR = {"Normal": "#2e7d32", "Benigna": "#f9a825", "Anormal": "#c62828"}


def preprocess_image_for_vgg16(pil_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (img_rgb_uint8, img_preprocessed_batch).

    Replica el preprocesado del notebook (cell 4.2): RGB -> resize 224x224
    -> float32 -> vgg16_preprocess (BGR + resta media ImageNet).
    """
    img_rgb = pil_image.convert("RGB").resize((224, 224))
    img_rgb_uint8 = np.asarray(img_rgb, dtype=np.uint8)
    img_float = np.asarray(img_rgb, dtype=np.float32)
    img_batch = np.expand_dims(img_float, axis=0)
    img_preprocessed = vgg16_preprocess(img_batch)
    return img_rgb_uint8, img_preprocessed


def _build_flat_grad_model(model, last_conv_layer_name="block5_conv3"):
    vgg = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "vgg" in layer.name.lower():
            vgg = layer
            break
    if vgg is None:
        raise RuntimeError("No se encontro el sub-modelo VGG16 dentro de model.")

    inputs = keras.Input(shape=model.input_shape[1:])
    x = inputs
    conv_output = None
    for layer in vgg.layers:
        if layer.__class__.__name__ == "InputLayer":
            continue
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x
    if conv_output is None:
        raise RuntimeError(f"No se encontro la capa {last_conv_layer_name}.")

    vgg_idx = model.layers.index(vgg)
    for layer in model.layers[vgg_idx + 1:]:
        x = layer(x)

    return keras.Model(inputs, [conv_output, x])


def make_gradcam_heatmap(img_preprocessed, model, pred_index=None,
                         last_conv_layer_name="block5_conv3"):
    grad_model = _build_flat_grad_model(model, last_conv_layer_name)

    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_preprocessed, training=False)
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., None]
        heatmap = tf.squeeze(heatmap).numpy()
    else:
        import torch
        x = torch.tensor(img_preprocessed, requires_grad=True)
        conv_outputs, predictions = grad_model(x)
        if pred_index is None:
            pred_index = int(predictions[0].argmax())
        class_channel = predictions[0, pred_index]
        grads = torch.autograd.grad(class_channel, conv_outputs)[0]
        pooled_grads = grads.mean(dim=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = (conv_outputs @ pooled_grads[..., None]).squeeze().detach().cpu().numpy()

    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def overlay_heatmap(original_rgb_uint8, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_rgb_uint8.shape[1],
                                            original_rgb_uint8.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(original_rgb_uint8 * (1 - alpha) + heatmap_color * alpha)
    return overlay


# --- UI ---

col_up, col_tog = st.columns([3, 1], gap="medium")
with col_up:
    uploaded = st.file_uploader(
        "Sube una imagen (.bmp, .jpg, .png)",
        type=["bmp", "jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
with col_tog:
    show_gradcam = st.toggle(
        "Ver Grad-CAM",
        value=True,
        help="Mapa de calor de las regiones que más pesan en la decisión.",
    )

if uploaded is None:
    st.info("Esperando una imagen para analizar.")
    st.stop()


pil_img = Image.open(uploaded)
img_rgb_uint8, img_preprocessed = preprocess_image_for_vgg16(pil_img)

probs = model.predict(img_preprocessed, verbose=0)[0]
pred_idx = int(np.argmax(probs))
pred_class_internal = CLASS_NAMES[pred_idx]
pred_class_es = CLASS_LABELS_ES[pred_class_internal]
pred_group = CLASS_GROUP[pred_class_internal]
pred_color = GROUP_COLOR[pred_group]
pred_confidence = float(probs[pred_idx])


if show_gradcam:
    with st.spinner("Calculando mapa de calor..."):
        heatmap = make_gradcam_heatmap(img_preprocessed, model, pred_index=pred_idx)
        overlay = overlay_heatmap(img_rgb_uint8, heatmap, alpha=0.4)

with st.container(border=True):
    if show_gradcam:
        col_img, col_cam, col_pred = st.columns([1, 1, 1.4], gap="medium")
    else:
        col_img, col_pred = st.columns([1, 1.4], gap="medium")
        col_cam = None

    with col_img:
        st.image(img_rgb_uint8, caption="Original (224×224)", width=230)

    if col_cam is not None:
        with col_cam:
            st.image(overlay, caption=f"Grad-CAM · {pred_class_es}", width=230)

    with col_pred:
        st.markdown(
            f"##### El modelo sugiere: <span style='color:{pred_color}'>{pred_class_es}</span>"
            f"  ·  <span style='color:{pred_color};font-size:0.9em'>{pred_group}</span>"
            f"  ·  **{pred_confidence:.0%}**",
            unsafe_allow_html=True,
        )
        probs_df = pd.DataFrame({
            "Clase": [CLASS_LABELS_ES[c] for c in CLASS_NAMES],
            "Probabilidad": probs,
        }).sort_values("Probabilidad", ascending=False)
        st.bar_chart(
            probs_df, x="Clase", y="Probabilidad",
            horizontal=True, height=200, color="#7C3AED",
        )
        st.caption(
            "Sugerencia del modelo, no diagnóstico. Grad-CAM: hipótesis visual de "
            "dónde mira la red (Selvaraju et al. 2017), idealmente sobre núcleo/citoplasma."
        )

with st.expander("Sobre el metodo y referencias"):
    st.markdown(
        """
**Arquitectura.** VGG16 preentrenada en ImageNet (Simonyan & Zisserman, 2014)
con feature extractor congelado y cabeza personalizada: GlobalAveragePooling2D
-> Dropout(0.4) -> BatchNorm -> Dense(128, ReLU) -> Dropout(0.2) -> BatchNorm
-> Dense(5, softmax). Protocolo replicado de Kaur et al. (2025).

**Preprocesado.** Las imagenes se redimensionan a 224x224 RGB y se aplica
`keras.applications.vgg16.preprocess_input`, que convierte a BGR y resta las
medias ImageNet [103.939, 116.779, 123.68] por canal. Es el preprocesado
oficial de VGG16 (Simonyan & Zisserman, 2014).

**Grad-CAM.** Selvaraju et al. (2017) calcula la importancia de cada region
como la combinacion lineal de los mapas de activacion de la ultima capa
convolucional (`block5_conv3` en VGG16) ponderada por el gradiente de la
clase predicha respecto a esos mapas. Aqui usamos α=0.4 para la superposicion,
dentro del rango tipico 0.3-0.5 del paper original.

**Dataset.** SIPaKMeD (Plissiti et al., 2018): 4049 celulas cervicales
recortadas, 5 clases morfologicas. Splits 60/20/20 estratificados con
`random_state=42` (test: 810 imagenes).

**Referencias:**

- Plissiti, M. E., Dimitrakopoulos, P., Sfikas, G., Nikou, C., Krikoni, O.,
  & Charchanti, A. (2018). *SIPaKMeD: A new dataset for feature and image
  based classification of normal and pathological cervical cells in Pap
  smear images*. IEEE ICIP 2018.
- Kaur, A. et al. (2025). *Comparison of deep transfer learning models for
  classification of cervical cancer from Pap smear images*. Scientific
  Reports. DOI: 10.1038/s41598-024-74531-0.
- Simonyan, K., & Zisserman, A. (2014). *Very deep convolutional networks
  for large-scale image recognition*. arXiv:1409.1556.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &
  Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via
  gradient-based localization*. ICCV 2017.
        """
    )

st.warning(
    "Herramienta educativa. No sustituye la evaluacion citologica por un "
    "profesional sanitario.",
    icon="⚠️",
)
