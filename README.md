# PathSun

Clasificador automático de células cervicales en imágenes de citología Papanicolaou mediante Deep Learning. Pensado como herramienta educativa en formación sanitaria, con énfasis en interpretación morfológica asistida.

## Qué hace

Dada una imagen recortada de una célula cervical (campo de 70×70 a 300×300 px, `.bmp`), PathSun predice a cuál de cinco categorías morfológicas pertenece:

| Categoría | Grupo clínico |
|---|---|
| Superficial-Intermediate | Normal |
| Parabasal | Normal |
| Metaplastic | Benigna |
| Koilocytotic | Anormal (HPV) |
| Dyskeratotic | Anormal (HPV) |

La predicción viene acompañada de un mapa de atención (Grad-CAM) que muestra qué regiones de la célula influyeron más en la decisión, como apoyo pedagógico para interpretar el modelo.

## Dataset

**SIPaKMeD** (Plissiti et al. 2018): 4049 imágenes de células cervicales individuales distribuidas en 5 clases. Fuente oficial: https://www.cs.uoi.gr/~marina/sipakmed.html

El dataset no se incluye en este repositorio por motivos de licencia y tamaño. Ver [data/README.md](data/README.md) para instrucciones de descarga.

## Método

Transfer learning sobre **VGG16** pre-entrenada en ImageNet:
- Feature extractor congelado
- Cabeza personalizada: `GlobalAveragePooling2D → Dropout → BatchNormalization → Dense(128, ReLU) → Dropout → BatchNormalization → Dense(5, softmax)`
- Entrenamiento: Adam (lr=0.001), batch 32, 50 epochs con early stopping, categorical cross-entropy

Elección de arquitectura basada en el estudio comparativo de Kaur et al. (2025), que sitúa a VGG16 como mejor clasificador 5-clases sobre SIPaKMeD (98.66% accuracy).

## Aplicación interactiva

Incluye una app Streamlit multipágina con tres modos:
- **Predicción:** sube una imagen y obtén clasificación + Grad-CAM
- **Modo Docente:** casos seleccionados del conjunto de test con feedback morfológico (vocabulario Bethesda en español)
- **Acerca de:** dataset, método, disclaimers

## Cómo ejecutar

Requisitos: **Python 3.11** (probado en 3.11.9 Windows).

```bash
# 1. Clonar / descomprimir el repositorio y entrar en la carpeta
cd PathSun

# 2. Crear entorno virtual e instalarlo todo
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 3. Arrancar la aplicación Streamlit
streamlit run app/app.py
```

Una vez iniciada, la app queda accesible en `http://localhost:8501`. Las tres páginas (`Predicción`, `Modo Docente`, `Acerca de`) están en la barra lateral.

El modelo entrenado (`models/pathsun_vgg16_best.keras`, 58 MB) y los 30 casos del Modo Docente (`app/casos_docente/`) ya vienen incluidos: no hay que reentrenar para usar la app. Si se desea reproducir el entrenamiento, ejecutar el notebook canónico `notebooks/PathSun_Final.ipynb`.

## Citas

- Plissiti, M.E., Dimitrakopoulos, P., Sfikas, G., Nikou, C., Krikoni, O., Charchanti, A. (2018). *SIPAKMED: A new dataset for feature and image based classification of normal and pathological cervical cells in pap smear images.* IEEE International Conference on Image Processing (ICIP), 3144–3148. https://doi.org/10.1109/ICIP.2018.8451588
- Kaur, H., Sharma, R., Kaur, J. (2025). *Comparison of deep transfer learning models for classification of cervical cancer from pap smear images.* Scientific Reports 15:3945. https://doi.org/10.1038/s41598-024-74531-0

## Aviso legal

Proyecto educativo. **No es un dispositivo médico ni sustituye el criterio de un profesional de anatomía patológica.** Las predicciones y mapas de atención son aproximados y no constituyen diagnóstico.

## Licencia

MIT. Ver [LICENSE](LICENSE) (pendiente de añadir).
