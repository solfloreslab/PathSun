"""Pagina 2 - Modo Docente.

Arquitectura pedagogica inspirada en CellaVision Proficiency Software,
herramienta comercial usada en formacion de tecnicos de hematologia.

Tres reglas innegociables (ver `project_pathsun_pedagogy.md` en memoria):

  1. Lock-then-reveal: el alumno clasifica primero, la IA habla despues.
  2. La IA da material para pensar, no respuestas: distribucion de
     probabilidades + Grad-CAM con disclaimer.
  3. Ground truth siempre humano: la verdad la pone el citopatologo que
     anoto SIPaKMeD, NO la prediccion del modelo.

Las funciones de carga del modelo, preprocesado y Grad-CAM estan reusadas
del notebook canonico PathSun_Final_v2.ipynb (seccion 8.1).
"""

import json
import random
from collections import Counter
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
CASOS_DIR = PROJECT_ROOT / "app" / "casos_docente"
CASOS_JSON = CASOS_DIR / "casos_modo_docente.json"

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _theme import apply_pathsun_theme
apply_pathsun_theme()

st.markdown("### 🎓 Modo Docente")

if not META_PATH.exists() or not MODEL_PATH.exists() or not CASOS_JSON.exists():
    st.error(
        "Faltan artefactos. Requeridos: `models/pathsun_vgg16_best.keras`, "
        "`models/pathsun_meta.json`, `app/casos_docente/casos_modo_docente.json`."
    )
    st.stop()


@st.cache_resource(show_spinner="Cargando modelo VGG16...")
def load_model_and_meta():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    model = keras.models.load_model(MODEL_PATH)
    return model, meta


@st.cache_data
def load_cases():
    return json.loads(CASOS_JSON.read_text(encoding="utf-8"))


model, meta = load_model_and_meta()
CASES = load_cases()
N_TOTAL = len(CASES)

CLASS_NAMES = meta["class_names"]
CLASS_LABELS_ES = meta["class_labels_es"]
CLASS_GROUP = meta["class_group"]
LABELS_ES_LIST = [CLASS_LABELS_ES[c] for c in CLASS_NAMES]

GROUP_COLOR = {"Normal": "#2e7d32", "Benigna": "#f9a825", "Anormal": "#c62828"}

# Fallback morfologico generico por clase, en vocabulario Bethesda.
# Se muestra cuando el caso no tiene `interpretacion_morfologica` propia.
GENERIC_MORPHOLOGY = {
    "Superficial-Intermedia": {
        "diagnostico": "Normal — epitelio escamoso maduro",
        "nucleo": "Picnótico, pequeño, central, cromatina densa y homogénea",
        "citoplasma": "Poligonal amplio, eosinófilo (queratinización fisiológica)",
        "rel_nc": "Muy baja",
        "clave": "Patrón típico del epitelio escamoso maduro normal",
        "diferencial": "Disqueratósica (núcleo atípico en HSIL)",
    },
    "Parabasal": {
        "diagnostico": "Normal — epitelio basal-parabasal",
        "nucleo": "Grande respecto al citoplasma pero regular, contorno liso, cromatina fina y uniforme",
        "citoplasma": "Redondeado, basófilo (azul-verdoso), denso",
        "rel_nc": "Alta (pero sin atipia)",
        "clave": "Sin halo perinuclear ni atipia",
        "diferencial": "Metaplásica (con pseudópodos citoplásmicos)",
    },
    "Metaplasica": {
        "diagnostico": "Benigna — metaplasia escamosa de zona de transformación",
        "nucleo": "Central, regular, cromatina fina",
        "citoplasma": "Denso, a veces con prolongaciones tipo pseudópodos",
        "rel_nc": "Moderada",
        "clave": "**Sin halo perinuclear** (la confusión con koilocito se resuelve mirando el núcleo)",
        "diferencial": "Parabasal (sin pseudópodos) · Koilocítica (con atipia nuclear)",
    },
    "Koilocitica": {
        "diagnostico": "**Anormal — LSIL** · patognomónica de infección por HPV",
        "nucleo": "Agrandado, hipercromático, contorno irregular, cromatina gruesa (puede haber binucleación)",
        "citoplasma": "Halo perinuclear amplio, bien delimitado, bordes irregulares",
        "rel_nc": "Aumentada",
        "clave": "Combinación **halo + atipia nuclear** (no solo el halo)",
        "diferencial": "Metaplásica (sin atipia nuclear)",
    },
    "Disqueratosica": {
        "diagnostico": "**Anormal — HSIL** (lesión escamosa intraepitelial de alto grado) o paraqueratosis atípica",
        "nucleo": "Pequeño, picnótico, hipercromático (posible anisocariosis)",
        "citoplasma": "Eosinófilo intenso, queratinización anómala",
        "rel_nc": "Variable, forma celular irregular",
        "clave": "Queratinización anómala con atipia nuclear",
        "diferencial": "Superficial-Intermedia (sin atipia nuclear)",
    },
}


def _render_morphology_card(class_es: str):
    """Renderiza la morfologia como tabla estructurada Bethesda (compacta)."""
    m = GENERIC_MORPHOLOGY[class_es]
    st.markdown(f"**Diagnóstico:** {m['diagnostico']}")
    st.markdown(
        f"""
| | |
|---|---|
| **Núcleo** | {m['nucleo']} |
| **Citoplasma** | {m['citoplasma']} |
| **Clave** | {m['clave']} |
| **Confunde con** | {m['diferencial']} |
        """
    )

COMMON_CONFUSIONS = {
    ("Metaplasica", "Parabasal"): (
        "Metaplasica y Parabasal se confunden por su tamano celular "
        "similar. La clave es la **relacion N/C** (mayor en parabasal) y "
        "los pseudopodos citoplasmicos (caracteristicos de metaplasica)."
    ),
    ("Koilocitica", "Metaplasica"): (
        "El halo perinuclear de la koilocitica puede confundirse con el "
        "citoplasma vacuolar de algunas metaplasicas. La clave es la "
        "**atipia nuclear**: en koilocito el nucleo esta agrandado e "
        "hipercromatico; en metaplasica es regular."
    ),
    ("Disqueratosica", "Superficial-Intermedia"): (
        "Ambas tienen citoplasma eosinofilo. En disqueratosica la "
        "queratinizacion es anomala y el nucleo es atipico; en superficial-"
        "intermedia el nucleo es picnotico pero regular."
    ),
}


def preprocess_image_for_vgg16(pil_image: Image.Image):
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
        raise RuntimeError("No se encontro VGG16 anidado.")
    inputs = keras.Input(shape=model.input_shape[1:])
    x = inputs
    conv_output = None
    for layer in vgg.layers:
        if layer.__class__.__name__ == "InputLayer":
            continue
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x
    vgg_idx = model.layers.index(vgg)
    for layer in model.layers[vgg_idx + 1:]:
        x = layer(x)
    return keras.Model(inputs, [conv_output, x])


def make_gradcam_heatmap(img_preprocessed, model, pred_index=None):
    grad_model = _build_flat_grad_model(model)
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


def _init_state():
    st.session_state.setdefault("dt_state", "welcome")
    st.session_state.setdefault("dt_case_order", [])
    st.session_state.setdefault("dt_case_pos", 0)
    st.session_state.setdefault("dt_user_answer", None)
    st.session_state.setdefault("dt_history", [])


def _reset_session():
    for k in ("dt_state", "dt_case_order", "dt_case_pos",
             "dt_user_answer", "dt_history"):
        st.session_state.pop(k, None)
    _init_state()


def _start_session(starting_case_idx=None):
    order = list(range(N_TOTAL))
    random.shuffle(order)
    if starting_case_idx is not None and starting_case_idx in order:
        order.remove(starting_case_idx)
        order.insert(0, starting_case_idx)
    st.session_state.dt_case_order = order
    st.session_state.dt_case_pos = 0
    st.session_state.dt_user_answer = None
    st.session_state.dt_history = []
    st.session_state.dt_state = "classifying"


_init_state()


with st.sidebar:
    st.markdown("### 🎓 Modo Docente")
    n_done = len(st.session_state.dt_history)
    st.markdown(f"**Progreso:** {n_done}/{N_TOTAL} casos")
    if n_done > 0:
        n_correct = sum(1 for h in st.session_state.dt_history if h["user_correct"])
        acc = n_correct / n_done
        st.markdown(f"**Tu accuracy:** {acc:.0%} ({n_correct}/{n_done})")
        wrong_pairs = [(h["real_label_es"], h["user_answer_es"])
                       for h in st.session_state.dt_history if not h["user_correct"]]
        if wrong_pairs:
            top = Counter(wrong_pairs).most_common(1)[0]
            (real, user_ans), n = top
            st.markdown(f"**Confusion mas frecuente:**  \n{real} → {user_ans} ({n}x)")
    st.divider()
    if st.session_state.dt_state != "welcome":
        if st.button("Reiniciar sesion", type="secondary"):
            _reset_session()
            st.rerun()


def render_welcome():
    with st.container(border=True):
        st.markdown(
            "Aprenderás citología mirando **30 casos seleccionados** del dataset "
            "SIPaKMeD (6 por clase). Mecánica inspirada en **CellaVision "
            "Proficiency Software**, herramienta comercial usada en formación "
            "de técnicos de hematología:"
        )
        st.markdown(
            "1. Miras la imagen sin saber qué es\n"
            "2. Eliges tu respuesta entre las 5 clases\n"
            "3. **Confirmas** — solo entonces se revela la verdad\n"
            "4. Después ves qué dijo el modelo, sus probabilidades y dónde mira"
        )
        st.info(
            "**La autoridad de la verdad es el citopatólogo humano que anotó "
            "SIPaKMeD, no la IA.** El modelo es un compañero de estudio cuya "
            "opinión evaluamos.",
            icon="🧭",
        )

    col_a, col_b, col_c = st.columns(3, gap="medium")
    with col_a:
        with st.container(border=True):
            st.markdown("##### ▶️ Empezar")
            st.caption("Recorre los 30 casos en orden aleatorio.")
            if st.button("Aleatorio", type="primary", use_container_width=True):
                _start_session()
                st.rerun()
    with col_b:
        with st.container(border=True):
            st.markdown("##### 🔢 Caso específico")
            n_input = st.number_input(
                "Número del caso (1-30)",
                min_value=1, max_value=N_TOTAL, value=1, step=1,
                label_visibility="collapsed",
            )
            if st.button(f"Ir al caso #{n_input}", use_container_width=True):
                _start_session(starting_case_idx=int(n_input) - 1)
                st.rerun()
    with col_c:
        with st.container(border=True):
            st.markdown("##### 🎯 Caso clave")
            st.caption("El único caso donde el modelo se equivoca — oro pedagógico.")
            failed_idx = next((i for i, c in enumerate(CASES) if not c["model_correct"]), None)
            if failed_idx is not None:
                if st.button("Ir al caso fallado", use_container_width=True):
                    _start_session(starting_case_idx=failed_idx)
                    st.rerun()


def render_classifying():
    pos = st.session_state.dt_case_pos
    case_idx = st.session_state.dt_case_order[pos]
    case = CASES[case_idx]
    img_path = CASOS_DIR / case["filename"]

    st.markdown(f"#### Caso {pos + 1} de {len(st.session_state.dt_case_order)}")
    st.caption("Mira la imagen y elige tu respuesta. El modelo está mudo en esta pantalla.")

    with st.container(border=True):
        col_img, col_resp = st.columns([1, 1], gap="large")
        with col_img:
            pil_img = Image.open(img_path)
            # LANCZOS para reescalado de calidad (no el bilinear default).
            # Tamano nativo de SIPaKMeD CROPPED varia entre 100 y 433 px.
            TARGET_W = 260
            w, h = pil_img.size
            if w != TARGET_W:
                new_h = int(h * TARGET_W / w)
                pil_img = pil_img.resize((TARGET_W, new_h), Image.LANCZOS)
            # NO mostramos el filename: revela la clase y rompe el lock-then-reveal
            st.image(pil_img)

        with col_resp:
            st.markdown("**¿Qué clase crees que es?**")
            options_with_group = [
                f"{CLASS_LABELS_ES[c]} ({CLASS_GROUP[c]})"
                for c in CLASS_NAMES
            ]
            choice = st.radio(
                "Tu respuesta:",
                options=options_with_group,
                index=None,
                label_visibility="collapsed",
            )

            confirm = st.button(
                "✅ Confirmar mi respuesta",
                type="primary",
                disabled=(choice is None),
            )
            if confirm and choice is not None:
                chosen_es = choice.split(" (")[0]
                st.session_state.dt_user_answer = chosen_es
                st.session_state.dt_state = "revealed"
                st.rerun()

            st.caption(
                "Pista: las 3 normales/benignas son Superficial-Intermedia, "
                "Parabasal y Metaplásica. Las 2 anormales son Koilocítica "
                "(LSIL/HPV) y Disqueratósica (HSIL)."
            )


def render_revealed():
    pos = st.session_state.dt_case_pos
    case_idx = st.session_state.dt_case_order[pos]
    case = CASES[case_idx]
    img_path = CASOS_DIR / case["filename"]

    user_ans = st.session_state.dt_user_answer
    real_es = case["real_label_es"]
    user_correct = (user_ans == real_es)

    pil_img = Image.open(img_path)
    img_rgb_uint8, img_preprocessed = preprocess_image_for_vgg16(pil_img)
    # Imagen para display con LANCZOS (mejor calidad que bilinear default)
    _w, _h = pil_img.size
    if _w != 260:
        _new_h = int(_h * 260 / _w)
        pil_display = pil_img.resize((260, _new_h), Image.LANCZOS)
    else:
        pil_display = pil_img

    real_color = GROUP_COLOR[CLASS_GROUP[case["real_label_key"]]]

    # Computar prediccion del modelo + Grad-CAM (una sola vez)
    probs = model.predict(img_preprocessed, verbose=0)[0]
    model_pred_idx = int(np.argmax(probs))
    model_pred_es = CLASS_LABELS_ES[CLASS_NAMES[model_pred_idx]]
    model_correct = (model_pred_es == real_es)

    with st.spinner("Calculando Grad-CAM..."):
        heatmap = make_gradcam_heatmap(img_preprocessed, model, pred_index=model_pred_idx)
        overlay = overlay_heatmap(img_rgb_uint8, heatmap, alpha=0.4)

    # === Cabecera + navegacion arriba (siempre visible sin scroll) ===
    is_last = (pos == len(st.session_state.dt_case_order) - 1)
    head_l, head_r = st.columns([2, 1])
    with head_l:
        st.markdown(
            f"##### Caso {pos + 1} de {len(st.session_state.dt_case_order)} — Resultado"
        )
    with head_r:
        nav_a, nav_b = st.columns(2, gap="small")
        with nav_a:
            next_label = "🏁 Resultado final" if is_last else "Siguiente →"
            if st.button(next_label, type="primary", use_container_width=True,
                          key="next_top"):
                if is_last:
                    st.session_state.dt_state = "done"
                else:
                    st.session_state.dt_case_pos += 1
                    st.session_state.dt_user_answer = None
                    st.session_state.dt_state = "classifying"
                st.rerun()
        with nav_b:
            if st.button("Terminar", use_container_width=True, key="end_top"):
                st.session_state.dt_state = "done"
                st.rerun()

    # === Layout: 2 columnas. Izquierda imagenes + modelo. Derecha Verdad humana (tall). ===
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        # Card unica: Imagenes + bar chart + comentario (todo del lado del modelo)
        with st.container(border=True):
            st.markdown("#### 🤖 Análisis del modelo")

            c_orig, c_cam = st.columns(2, gap="small")
            with c_orig:
                st.markdown("**📷 Original**")
                st.image(pil_display, width=140)
            with c_cam:
                st.markdown(f"**🔍 Grad-CAM α=0.4**")
                st.image(overlay, width=170)
            st.caption(
                f"Grad-CAM: hipótesis visual de dónde mira la IA sobre clase "
                f"**{model_pred_es}** (Selvaraju et al. 2017)."
            )

            st.markdown(
                f"Sugiere **{model_pred_es}** ({probs[model_pred_idx]:.0%}) · "
                f"{'✓ acertó este caso' if model_correct else '✗ falló este caso'}"
            )
            probs_df = pd.DataFrame({
                "Clase": LABELS_ES_LIST,
                "Probabilidad": probs,
            }).sort_values("Probabilidad", ascending=False)
            st.bar_chart(
                probs_df, x="Clase", y="Probabilidad",
                horizontal=True, height=220, color="#7C3AED",
            )

    with col_right:
        with st.container(border=True):
            st.markdown("#### 📋 Diagnóstico")
            if user_correct:
                st.success(f"✅ **Acertaste.** Tu respuesta: {user_ans}")
            else:
                st.error(f"❌ **Tu respuesta:** {user_ans}")
                st.success(f"✅ **Respuesta correcta:** {real_es}")
            if not user_correct:
                confusion_key = (real_es, user_ans)
                rev = (user_ans, real_es)
                if confusion_key in COMMON_CONFUSIONS:
                    st.caption(f"💡 {COMMON_CONFUSIONS[confusion_key]}")
                elif rev in COMMON_CONFUSIONS:
                    st.caption(f"💡 {COMMON_CONFUSIONS[rev]}")
            # Morfologia: si hay interpretacion por caso (escrita por la autora),
            # usarla; si no, tabla Bethesda generica con disclaimer de IA.
            interp = case.get("interpretacion_morfologica", "").strip()
            if interp:
                st.caption("✍️ Lectura morfológica de la autora (anatomopatóloga)")
                st.markdown(interp)
            else:
                st.caption(
                    "⚠️ Texto orientativo **generado con IA** desde Plissiti 2018 + "
                    "criterios Bethesda — pendiente de revisión clínica por la autora."
                )
                _render_morphology_card(real_es)

    history = st.session_state.dt_history
    already_logged = any(h["pos"] == pos and h["case_idx"] == case_idx for h in history)
    if not already_logged:
        history.append({
            "pos": pos,
            "case_idx": case_idx,
            "real_label_es": real_es,
            "user_answer_es": user_ans,
            "model_pred_es": model_pred_es,
            "user_correct": user_correct,
            "model_correct": model_correct,
        })

    # (Botones de navegacion ya estan arriba en la cabecera)


def render_done():
    history = st.session_state.dt_history
    n = len(history)
    if n == 0:
        st.warning("Sesion sin casos resueltos. Vuelve al inicio.")
        if st.button("Volver al inicio"):
            _reset_session()
            st.rerun()
        return

    n_user_correct = sum(1 for h in history if h["user_correct"])
    n_model_correct = sum(1 for h in history if h["model_correct"])
    user_acc = n_user_correct / n
    model_acc = n_model_correct / n

    st.success(f"🎉 Sesión completada: {n} casos resueltos.")

    # Layout denso: metricas a la izquierda, tabla aciertos por clase a la derecha
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        with st.container(border=True):
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric("Tu accuracy", f"{user_acc:.0%}", f"{n_user_correct}/{n}")
            with mc2:
                st.metric("Modelo", f"{model_acc:.0%}", f"{n_model_correct}/{n}")

            # Comparativa caso a caso (compacta)
            discord = [h for h in history if h["user_answer_es"] != h["model_pred_es"]]
            n_user_right_model_wrong = sum(1 for h in discord
                                            if h["user_correct"] and not h["model_correct"])
            n_user_wrong_model_right = sum(1 for h in discord
                                            if not h["user_correct"] and h["model_correct"])
            n_both_wrong = sum(1 for h in history
                                if not h["user_correct"] and not h["model_correct"])
            st.markdown("**Comparativa con el modelo**")
            st.markdown(
                f"- Desacuerdo: **{len(discord)}** casos\n"
                f"- Tú ✓, modelo ✗: **{n_user_right_model_wrong}**"
                f"{' ← buen ojo' if n_user_right_model_wrong > 0 else ''}\n"
                f"- Tú ✗, modelo ✓: {n_user_wrong_model_right}\n"
                f"- Ambos ✗: {n_both_wrong}"
            )

    with col_right:
        with st.container(border=True):
            st.markdown("**Aciertos por clase**")
            by_class = {lab: {"correct": 0, "total": 0} for lab in LABELS_ES_LIST}
            for h in history:
                by_class[h["real_label_es"]]["total"] += 1
                if h["user_correct"]:
                    by_class[h["real_label_es"]]["correct"] += 1
            rows = []
            for lab in LABELS_ES_LIST:
                c, t = by_class[lab]["correct"], by_class[lab]["total"]
                rows.append({
                    "Clase": lab,
                    "Aciertos": f"{c}/{t}" if t > 0 else "-",
                    "%": f"{(c/t*100):.0f}%" if t > 0 else "-",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            wrong_pairs = [(h["real_label_es"], h["user_answer_es"])
                           for h in history if not h["user_correct"]]
            if wrong_pairs:
                top = Counter(wrong_pairs).most_common(3)
                st.caption("**Confusiones más frecuentes:** " +
                           " · ".join(f"{real}→{ua} ({n}x)" for (real, ua), n in top))

    st.divider()
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        if st.button("🔁 Repetir sesion", use_container_width=True):
            _reset_session()
            _start_session()
            st.rerun()
    with col_r2:
        wrong_idxs = [h["case_idx"] for h in history if not h["user_correct"]]
        if wrong_idxs:
            if st.button(f"🎯 Solo casos fallados ({len(wrong_idxs)})",
                          use_container_width=True):
                _reset_session()
                random.shuffle(wrong_idxs)
                st.session_state.dt_case_order = wrong_idxs
                st.session_state.dt_state = "classifying"
                st.rerun()
    with col_r3:
        if st.button("Volver al inicio", use_container_width=True):
            _reset_session()
            st.rerun()


state = st.session_state.dt_state
if state == "welcome":
    render_welcome()
elif state == "classifying":
    render_classifying()
elif state == "revealed":
    render_revealed()
elif state == "done":
    render_done()
