# ══════════════════════════════════════════════════════════════════════════
#  FragmentApp — Análisis de Fragmentación de Roca (estilo WipFrag / SigmaFrag)
#  Autor: Juan Carlos Lermo Layme
#  Demo pública con acceso restringido por usuario y contraseña.
# ══════════════════════════════════════════════════════════════════════════

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import tempfile, os
from pathlib import Path
import pandas as pd
import gdown

# ──────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FragmentApp — Análisis de Roca",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
#  AUTENTICACIÓN (multiusuario vía st.secrets)
# ──────────────────────────────────────────────────────────────────────────
# Las credenciales NUNCA se escriben en este archivo. Se definen en
# .streamlit/secrets.toml (local) o en "Secrets" del panel de Streamlit
# Cloud, con esta forma:
#
# [credenciales]
# usuario1 = "claveSegura123"
# usuario2 = "otraClave456"
#
# Así puedes crear, cambiar o quitar usuarios sin tocar el código.

def verificar_login():
    """Devuelve True si la sesión ya está autenticada. Si no, dibuja el
    formulario de login y detiene la ejecución del resto del script."""

    if st.session_state.get("autenticado", False):
        return True

    st.markdown(
        """
        <div style="max-width:420px;margin:80px auto 0 auto;text-align:center;">
            <div style="font-size:42px;">🪨</div>
            <div style="font-family:'Syne',sans-serif;font-size:26px;font-weight:700;margin-top:4px;">
                FragmentApp
            </div>
            <div style="color:#6b7280;font-size:13px;margin-top:-4px;">
                Acceso restringido · Demo privada
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("form_login", border=True):
            usuario = st.text_input("Usuario")
            clave = st.text_input("Contraseña", type="password")
            enviado = st.form_submit_button("Ingresar", width="stretch")

        if enviado:
            credenciales = st.secrets.get("credenciales", {})
            if not credenciales:
                st.error(
                    "No hay credenciales configuradas en st.secrets. "
                    "Revisa .streamlit/secrets.toml o el panel de Secrets de Streamlit Cloud."
                )
            elif usuario in credenciales and clave == credenciales[usuario]:
                st.session_state["autenticado"] = True
                st.session_state["usuario_actual"] = usuario
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

    st.markdown(
        """
        <div style="text-align:center;color:#9ca3af;font-size:12px;margin-top:30px;">
            © FragmentApp — Autor: Juan Carlos Lermo Layme
        </div>
        """,
        unsafe_allow_html=True,
    )
    return False


if not verificar_login():
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
#  DESCARGA DEL MODELO ENTRENADO DESDE GOOGLE DRIVE (no se sube a GitHub)
# ──────────────────────────────────────────────────────────────────────────
MODELO_PATH = "best.pt"
DRIVE_FILE_ID = "1vVdvUfLMejx0JWocpTc957i0jEFmwh4j"

def asegurar_modelo():
    """Descarga best.pt desde Drive si todavía no existe en el contenedor."""
    if Path(MODELO_PATH).exists() and Path(MODELO_PATH).stat().st_size > 0:
        return True

    with st.spinner("Descargando modelo entrenado (solo la primera vez)..."):
        try:
            # gdown.download maneja automáticamente la página de confirmación
            # de archivos grandes de Google Drive.
            gdown.download(
                id=DRIVE_FILE_ID,
                output=MODELO_PATH,
                quiet=False,
                fuzzy=True,
            )
        except Exception as e:
            st.error(f"Error al descargar el modelo desde Google Drive: {e}")
            return False

    if not Path(MODELO_PATH).exists() or Path(MODELO_PATH).stat().st_size < 1_000_000:
        st.error(
            "La descarga del modelo falló o el archivo está incompleto. "
            "Verifica que el enlace de Drive sea público (Cualquier persona con el enlace)."
        )
        return False
    return True


modelo_listo = asegurar_modelo()

# ──────────────────────────────────────────────────────────────────────────
#  ESTILOS (CSS)
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }

    .section-title {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 16px;
        color: #1f2937;
        margin: 18px 0 10px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #e5e7eb;
    }

    div[data-testid="stMetric"] {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px 16px;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
#  INTERVALOS SigmaFrag (Finos / Medios / Gruesos)
# ──────────────────────────────────────────────────────────────────────────
def get_intervalos(fino_max, grueso_min):
    """Define los tres rangos granulométricos y su color BGR para overlay."""
    return [
        ("Finos",   None,       fino_max,   (220,  50,  50), "fino"),
        ("Medios",  fino_max,   grueso_min, ( 50, 180,  50), "medio"),
        ("Gruesos", grueso_min, None,       (230, 140,  30), "grueso"),
    ]


def clasificar_intervalo(diametro, intervalos):
    """Devuelve la etiqueta del intervalo al que pertenece un diámetro dado."""
    for nombre, dmin, dmax, color, clave in intervalos:
        if dmin is None and diametro <= dmax:
            return nombre, color, clave
        if dmax is None and diametro > dmin:
            return nombre, color, clave
        if dmin is not None and dmax is not None and dmin < diametro <= dmax:
            return nombre, color, clave
    # fallback: el último intervalo (gruesos)
    nombre, _, _, color, clave = intervalos[-1]
    return nombre, color, clave


# ──────────────────────────────────────────────────────────────────────────
#  CARGA DEL MODELO YOLO (cacheada para no recargar en cada interacción)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def cargar_modelo(ruta_modelo):
    from ultralytics import YOLO
    return YOLO(ruta_modelo)


# ──────────────────────────────────────────────────────────────────────────
#  PROCESAMIENTO PRINCIPAL: segmentación + métricas granulométricas
# ──────────────────────────────────────────────────────────────────────────
def procesar(imagen_bgr, ancho_real, conf, iou, intervalos, unidad, modelo_path):
    """
    Ejecuta segmentación YOLO sobre la imagen, calcula el diámetro
    equivalente de cada fragmento detectado (en la unidad real elegida,
    usando una referencia de escala simple: ancho_real / ancho_px de la
    imagen), y construye la curva granulométrica acumulada (D10, D50, D80...).

    Devuelve:
        fragmentos: lista de dicts con máscara, contorno, área_px, diam_equiv
        curva: dict con percentiles D-x y arrays para graficar
        escala: factor de conversión px -> unidad real
    """
    modelo = cargar_modelo(modelo_path)

    alto_px, ancho_px = imagen_bgr.shape[:2]
    escala = ancho_real / ancho_px  # unidad real por pixel

    resultados = modelo.predict(
        source=imagen_bgr,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    if not resultados or resultados[0].masks is None:
        return None, None, None

    r = resultados[0]
    masks = r.masks.data.cpu().numpy()  # (N, H, W) en resolución del modelo
    h_modelo, w_modelo = masks.shape[1:3]

    fragmentos = []
    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(
            mask.astype(np.uint8), (ancho_px, alto_px), interpolation=cv2.INTER_NEAREST
        )
        area_px = float(mask_resized.sum())
        if area_px <= 0:
            continue

        # Diámetro equivalente de un círculo con la misma área
        diam_px = 2.0 * np.sqrt(area_px / np.pi)
        diam_equiv = diam_px * escala

        contornos, _ = cv2.findContours(
            mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contorno = max(contornos, key=cv2.contourArea) if contornos else None

        nombre_intervalo, color, clave = clasificar_intervalo(diam_equiv, intervalos)

        fragmentos.append({
            "id": i,
            "area_px": area_px,
            "diam_equiv": diam_equiv,
            "contorno": contorno,
            "intervalo": nombre_intervalo,
            "color": color,
            "clave": clave,
        })

    if not fragmentos:
        return None, None, None

    curva = calcular_curva_granulometrica(fragmentos)
    return fragmentos, curva, escala


def calcular_curva_granulometrica(fragmentos):
    """Calcula la curva acumulada de % pasante (estilo WipFrag) y percentiles
    D10, D20 ... D90 a partir del área de cada fragmento (proxy de masa)."""
    diams = np.array([f["diam_equiv"] for f in fragmentos])
    areas = np.array([f["area_px"] for f in fragmentos])

    orden = np.argsort(diams)
    diams_ord = diams[orden]
    areas_ord = areas[orden]

    area_acumulada = np.cumsum(areas_ord)
    pct_pasante = 100.0 * area_acumulada / area_acumulada[-1]

    curva = {
        "diametros": diams_ord,
        "pct_pasante": pct_pasante,
    }

    for p in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]:
        idx = np.searchsorted(pct_pasante, p)
        idx = min(idx, len(diams_ord) - 1)
        curva[f"D{p}"] = float(diams_ord[idx])

    return curva


# ──────────────────────────────────────────────────────────────────────────
#  FIGURAS (matplotlib)
# ──────────────────────────────────────────────────────────────────────────
def fig_segmentacion(img_bgr, fragmentos, intervalos):
    """Dibuja los contornos de cada fragmento coloreados por intervalo."""
    overlay = img_bgr.copy()
    for f in fragmentos:
        if f["contorno"] is not None:
            cv2.drawContours(overlay, [f["contorno"]], -1, f["color"], thickness=3)

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(overlay_rgb)
    ax.axis("off")

    leyenda = [
        Patch(facecolor=np.array(color[::-1]) / 255, label=nombre)
        for nombre, _, _, color, _ in intervalos
    ]
    ax.legend(handles=leyenda, loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


def fig_curva(curva, intervalos, unidad):
    """Curva granulométrica acumulada (% pasante vs diámetro)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(curva["diametros"], curva["pct_pasante"], color="#374151", linewidth=2)

    for p in [50, 80]:
        if f"D{p}" in curva:
            ax.axvline(curva[f"D{p}"], color="#9ca3af", linestyle="--", linewidth=1)
            ax.text(
                curva[f"D{p}"], 102, f"D{p}={curva[f'D{p}']:.1f}",
                fontsize=8, ha="center", color="#374151"
            )

    ax.set_xlabel(f"Diámetro equivalente ({unidad})")
    ax.set_ylabel("% Pasante acumulado")
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    fig.tight_layout()
    return fig


def fig_histograma(fragmentos, intervalos, unidad):
    """Histograma de frecuencia de tamaños, coloreado por intervalo."""
    diams = np.array([f["diam_equiv"] for f in fragmentos])

    fig, ax = plt.subplots(figsize=(10, 4))
    n, bins, parches = ax.hist(diams, bins=25, edgecolor="white")

    for parche, borde_izq in zip(parches, bins[:-1]):
        _, color_bgr, _ = clasificar_intervalo(borde_izq, intervalos)
        parche.set_facecolor(np.array(color_bgr[::-1]) / 255)

    ax.set_xlabel(f"Diámetro equivalente ({unidad})")
    ax.set_ylabel("N° de fragmentos")
    ax.grid(alpha=0.3, axis="y")

    leyenda = [
        Patch(facecolor=np.array(color[::-1]) / 255, label=nombre)
        for nombre, _, _, color, _ in intervalos
    ]
    ax.legend(handles=leyenda, fontsize=9)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Parámetros de análisis
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Parámetros")
    st.caption(f"Sesión: **{st.session_state.get('usuario_actual', '')}**")

    ancho_real = st.number_input(
        "Ancho real de la escena", min_value=0.01, value=1.0, step=0.1,
        help="Ancho real (en la unidad elegida) que representa el ancho total de la imagen."
    )
    unidad = st.selectbox("Unidad", ["m", "cm", "mm", "in", "ft"], index=0)

    st.markdown("---")
    conf = st.slider("Confianza (conf)", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU", 0.05, 0.95, 0.45, 0.05)

    st.markdown("---")
    st.markdown("**Intervalos granulométricos**")
    fino_max = st.number_input(f"Máximo Finos ({unidad})", min_value=0.0, value=0.10, step=0.01)
    grueso_min = st.number_input(f"Mínimo Gruesos ({unidad})", min_value=0.0, value=0.40, step=0.01)

    intervalos = get_intervalos(fino_max, grueso_min)

    modelo_path = MODELO_PATH

    if st.button("Cerrar sesión", width="stretch"):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.caption("© FragmentApp — Autor: Juan Carlos Lermo Layme")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
st.markdown("# Fragmentometría de Roca")
st.markdown(
    "<p style='color:#6b7280;margin-top:-12px'>Análisis automático de fragmentación · Estilo WipFrag / SigmaFrag</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

if not modelo_listo:
    st.warning(
        "El modelo no está disponible todavía. Recarga la página o revisa "
        "que el enlace de Google Drive sea público."
    )
    st.stop()

tab_upload, tab_camara = st.tabs(["📁  Cargar imagen", "📷  Tomar foto"])

imagen_bgr = None

with tab_upload:
    archivo = st.file_uploader(
        "Arrastra tu imagen o haz clic para seleccionar",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if archivo:
        datos = np.frombuffer(archivo.read(), np.uint8)
        imagen_bgr = cv2.imdecode(datos, cv2.IMREAD_COLOR)
        st.image(
            cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB),
            caption="Imagen cargada", width="stretch",
        )

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
                    imagen_bgr, ancho_real, conf, iou, intervalos, unidad, modelo_path
                )

            if fragmentos is None:
                st.error("No se detectaron fragmentos. Ajusta Confianza o IoU.")
            else:
                st.session_state.resultados = (fragmentos, curva, escala, imagen_bgr.copy())

if "resultados" in st.session_state:
    fragmentos, curva, escala, img_orig = st.session_state.resultados
    diams = [f["diam_equiv"] for f in fragmentos]

    st.markdown("## Resultados")

    # ── Métricas clave ───────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("N° de fragmentos", f"{len(fragmentos)}")
    col2.metric(f"D50 ({unidad})", f"{curva.get('D50', 0):.3f}")
    col3.metric(f"D80 ({unidad})", f"{curva.get('D80', 0):.3f}")
    col4.metric(f"Diámetro medio ({unidad})", f"{np.mean(diams):.3f}")

    # ── Visualización + Curva ───────────────────────────────────────────
    col_img, col_curva = st.columns([1.1, 1])

    with col_img:
        st.markdown("<div class='section-title'>Segmentación por Intervalo</div>", unsafe_allow_html=True)
        st.pyplot(fig_segmentacion(img_orig, fragmentos, intervalos), width="stretch")

    with col_curva:
        st.markdown("<div class='section-title'>Curva Granulométrica</div>", unsafe_allow_html=True)
        st.pyplot(fig_curva(curva, intervalos, unidad), width="stretch")

    # ── Histograma ───────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Histograma</div>", unsafe_allow_html=True)
    st.pyplot(fig_histograma(fragmentos, intervalos, unidad), width="stretch")

    # ── Tabla percentiles ───────────────────────────────────────────────
    st.markdown("<div class='section-title'>Percentiles Completos</div>", unsafe_allow_html=True)
    percdf = pd.DataFrame([
        {"Percentil": k, f"Valor ({unidad})": f"{v:.4f}"}
        for k, v in curva.items() if k.startswith("D")
    ])
    st.dataframe(percdf, width="stretch", hide_index=True)

    # ── Estadísticas avanzadas ──────────────────────────────────────────
    st.markdown("<div class='section-title'>Estadísticas Avanzadas</div>", unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Mínimo", f"{np.min(diams):.3f} {unidad}")
    col6.metric("Máximo", f"{np.max(diams):.3f} {unidad}")
    col7.metric("Desv. estándar", f"{np.std(diams):.3f} {unidad}")
    n_finos = sum(1 for f in fragmentos if f["clave"] == "fino")
    col8.metric("% Finos", f"{100 * n_finos / len(fragmentos):.1f}%")

    # ── Descarga CSV ─────────────────────────────────────────────────────
    df = pd.DataFrame([
        {
            "id": f["id"],
            "diametro_equivalente": f["diam_equiv"],
            "unidad": unidad,
            "area_px": f["area_px"],
            "intervalo": f["intervalo"],
        }
        for f in fragmentos
    ])

    st.download_button(
        label="⬇  Descargar CSV completo",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="fragmentometria.csv",
        mime="text/csv",
    )

else:
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px; color: #6b7280;">
            <div style="font-size: 48px; margin-bottom: 16px;">🪨</div>
            <div style="font-family:'Syne',sans-serif; font-size: 20px; color: #9ca3af; margin-bottom: 8px;">
                Carga una imagen o toma una foto para comenzar
            </div>
            <div style="font-size: 13px;">
                Soporta JPG · PNG · WEBP · Cámara en vivo
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="text-align:center;color:#9ca3af;font-size:11px;margin-top:40px;">
        FragmentApp · Análisis de Fragmentación de Roca — Autor: Juan Carlos Lermo Layme
    </div>
    """,
    unsafe_allow_html=True,
)
