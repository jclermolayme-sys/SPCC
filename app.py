import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import tempfile, os
from pathlib import Path
import pandas as pd
import gdown  # <── Librería añadida para la descarga directa en vivo

st.set_page_config(
    page_title="FragmentApp — Análisis de Roca",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── TRUCO DE DESCARGA EN VIVO DESDE GOOGLE DRIVE ─────────────────────────────
if not Path("best.pt").exists():
    with st.spinner("Descargando el modelo YOLO entrenado desde Google Drive (Esto solo toma unos segundos)..."):
        id_drive = "1vVdvUfLMejx0JWocpTc957i0jEFmwh4j"
        url_descarga = f"https://drive.google.com/uc?id={id_drive}"
        try:
            gdown.download(url_descarga, "best.pt", quiet=False)
        except Exception as e:
            st.error(f"Error crítico al intentar descargar el modelo desde Google Drive: {e}")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""<style> ... </style>""", unsafe_allow_html=True)

# ─── INTERVALOS SigmaFrag ─────────────────────────────────────────────────────
def get_intervalos(fino_max, grueso_min):
    return [
        ("Finos",   None,       fino_max,   (220,  50,  50), "fino"),
        ("Medios",  fino_max,   grueso_min, ( 50, 180,  50), "medio"),
        ("Gruesos", grueso_min, None,       (230, 140,  30), "grueso"),
    ]

# ... (funciones auxiliares sin cambios)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# Fragmentometría de Roca")
st.markdown("<p style='color:#6b7280;margin-top:-12px'>Análisis automático de fragmentación · Estilo WipFrag / SigmaFrag</p>", unsafe_allow_html=True)
st.markdown("---")

tab_upload, tab_camara = st.tabs(["📁  Cargar imagen", "📷  Tomar foto"])

imagen_bgr = None

with tab_upload:
    archivo = st.file_uploader("Arrastra tu imagen o haz clic para seleccionar",
                                type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed")
    if archivo:
        datos = np.frombuffer(archivo.read(), np.uint8)
        imagen_bgr = cv2.imdecode(datos, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB),
                 caption="Imagen cargada", width="stretch")

with tab_camara:
    foto = st.camera_input("Toma una foto de la voladura")
    if foto:
        datos = np.frombuffer(foto.read(), np.uint8)
        imagen_bgr = cv2.imdecode(datos, cv2.IMREAD_COLOR)

st.markdown("---")

if imagen_bgr is not None:
    if st.button("⚡  ANALIZAR FRAGMENTACIÓN"):
        if not Path(modelo_path).exists():
            st.error(f"No se encontró el modelo en: **{modelo_path}**")
        else:
            with st.spinner("Ejecutando segmentación YOLO..."):
                fragmentos, curva, escala = procesar(
                    imagen_bgr, ancho_real, conf, iou, intervalos, unidad, modelo_path)

            if fragmentos is None:
                st.error("No se detectaron fragmentos. Ajusta Confianza o IoU.")
            else:
                st.session_state.resultados = (fragmentos, curva, escala, imagen_bgr.copy())

if "resultados" in st.session_state:
    fragmentos, curva, escala, img_orig = st.session_state.resultados
    diams = [f["diam_equiv"] for f in fragmentos]

    st.markdown("## Resultados")

    # ── Métricas clave ────────────────────────────────────────────────────────
    # (sin cambios en lógica)

    # ── Visualización + Curva ─────────────────────────────────────────────────
    col_img, col_curva = st.columns([1.1, 1])

    with col_img:
        st.markdown("<div class='section-title'>Segmentación por Intervalo</div>", unsafe_allow_html=True)
        st.pyplot(fig_segmentacion(img_orig, fragmentos, intervalos), width="stretch")

    with col_curva:
        st.markdown("<div class='section-title'>Curva Granulométrica</div>", unsafe_allow_html=True)
        st.pyplot(fig_curva(curva, intervalos, unidad), width="stretch")

    # ── Histograma ────────────────────────────────────────────
    st.markdown("<div class='section-title'>Histograma</div>", unsafe_allow_html=True)
    st.pyplot(fig_histograma(fragmentos, intervalos, unidad), width="stretch")

    # ── Tabla percentiles ─────────────────────────────────────
    st.markdown("<div class='section-title'>Percentiles Completos</div>", unsafe_allow_html=True)
    percdf = pd.DataFrame([
        {"Percentil": k, f"Valor ({unidad})": f"{v:.4f}"}
        for k, v in curva.items() if k.startswith("D")
    ])
    st.dataframe(percdf, width="stretch", hide_index=True)

    # ── Estadísticas avanzadas ─────────────────────────────────
    # (sin cambios en lógica)

    # ── Descarga CSV ──────────────────────────────────────────
    st.download_button(
        label="⬇  Descargar CSV completo",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="fragmentometria.csv",
        mime="text/csv",
    )

else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #6b7280;">
        <div style="font-size: 48px; margin-bottom: 16px;">🪨</div>
        <div style="font-family:'Syne',sans-serif; font-size: 20px; color: #9ca3af; margin-bottom: 8px;">
            Carga una imagen o toma una foto para comenzar
        </div>
        <div style="font-size: 13px;">
            Soporta JPG · PNG · WEBP · Cámara en vivo
        </div>
    </div>
    """, unsafe_allow_html=True)
