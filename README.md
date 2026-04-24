# PNN SSC Analysis 🧠🔬

Repositorio privado para el análisis de **Redes Perineuronales (PNNs)** e interneuronas **PV+** en la Corteza Somatosensorial (SSC).

## 📋 Descripción del Proyecto
Este proyecto busca automatizar el procesamiento de imágenes de inmunofluorescencia para cuantificar la densidad y morfología de las PNNs y su asociación con células Parvalbúmina positivas.

### Stack Tecnológico
* **Segmentación:** [Cellpose](https://www.cellpose.org/) para la segmentación de núcleos (DAPI) y somas.
* **Interfaz:** [Streamlit](https://streamlit.io/) para la creación de una WebApp de análisis interactivo.
* **Procesamiento:** Python (Scikit-image, OpenCV, Numpy).

---

## 🚀 Pipeline del Experimento (Streamlit App)

La aplicación está diseñada como un flujo de trabajo secuencial:

1.  **Home (`app.py`):** Explicación teórica del proyecto, objetivos y descripción de los marcadores (WFA, PV, DAPI).
2.  **Page 1: Visualización y Calibración:** Herramientas para ajustar contraste, brillo y definir los parámetros de calibración espacial (micras por píxel).
3.  **Page 2: Segmentación de Núcleos:** Implementación de Cellpose utilizando el canal **DAPI** para identificar las poblaciones celulares.
4.  **Page 3: Análisis de PNN:** Cuantificación de la intensidad de WFA y solapamiento con los núcleos segmentados.

---

## 📂 Estructura de Datos
* `data/raw/`: Imágenes originales sin procesar.
    * `/ACF_Cortex`: Muestras de corteza ACF.
    * `/PV_Cortex`: Muestras marcadas para Parvalbúmina.
* `data/processed/`: Resultados de segmentación y máscaras.
* `pages/`: Scripts de Streamlit para cada paso del pipeline.
* `app.py`: Punto de entrada de la aplicación.