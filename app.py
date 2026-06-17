# ══════════════════════════════════════════════════════════════════════════
#  FragmentApp — Análisis de Fragmentación de Roca (estilo SigmaFrag)
#  Autor: Juan Carlos Lermo Layme
#  Demo pública con acceso restringido por usuario y contraseña.
# ══════════════════════════════════════════════════════════════════════════

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import tempfile, os, base64, json
from pathlib import Path
import pandas as pd
import gdown
from io import BytesIO

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
#  AUTENTICACIÓN
# ──────────────────────────────────────────────────────────────────────────
def verificar_login():
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
            enviado = st.form_submit_button("Ingresar", use_container_width=True)

        if enviado:
            credenciales = st.secrets.get("credenciales", {})
            if not credenciales:
                st.error("No hay credenciales configuradas en st.secrets.")
            elif usuario in credenciales and clave == credenciales[usuario]:
                st.session_state["autenticado"] = True
                st.session_state["usuario_actual"] = usuario
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

    st.markdown(
        """<div style="text-align:center;color:#9ca3af;font-size:12px;margin-top:30px;">
            © FragmentApp — Autor: Juan Carlos Lermo Layme</div>""",
        unsafe_allow_html=True,
    )
    return False


if not verificar_login():
    st.stop()

# ──────────────────────────────────────────────────────────────────────────
#  DESCARGA DEL MODELO
# ──────────────────────────────────────────────────────────────────────────
MODELO_PATH = "best.pt"
DRIVE_FILE_ID = "1vVdvUfLMejx0JWocpTc957i0jEFmwh4j"

def asegurar_modelo():
    if Path(MODELO_PATH).exists() and Path(MODELO_PATH).stat().st_size > 0:
        return True
    with st.spinner("Descargando modelo entrenado (solo la primera vez)..."):
        try:
            gdown.download(id=DRIVE_FILE_ID, output=MODELO_PATH, quiet=False)
        except Exception as e:
            st.error(f"Error al descargar el modelo: {e}")
            return False
    if not Path(MODELO_PATH).exists() or Path(MODELO_PATH).stat().st_size < 1_000_000:
        st.error("La descarga falló o el archivo está incompleto.")
        return False
    return True


modelo_listo = asegurar_modelo()

# ──────────────────────────────────────────────────────────────────────────
#  PALETA SigmaFrag (8 bandas de percentil)
# ──────────────────────────────────────────────────────────────────────────
# Cada banda: (etiqueta, percentil_inferior, percentil_superior, color_BGR)
# Orden: más fino → más grueso
SIGMAFRAG_BANDAS = [
    ("< D01",       None,  1,  ( 20, 180, 220)),   # cian claro
    ("D01 – D05",    1,    5,  ( 50, 220, 160)),   # verde-agua
    ("D05 – D10",    5,   10,  ( 50, 200,  50)),   # verde
    ("D10 – D20",   10,   20,  (120, 230,  60)),   # verde-amarillo
    ("D20 – D50",   20,   50,  ( 50, 210, 210)),   # amarillo-verde
    ("D50 – D80",   50,   80,  ( 20, 200, 230)),   # amarillo
    ("D80 – D90",   80,   90,  ( 30, 140, 230)),   # naranja
    ("D90 – D95",   90,   95,  (  0,  60, 200)),   # rojo-naranja
    ("> D95",       95,  None, (  0,  30, 160)),   # rojo oscuro
]

# Color de contorno en overlay (celeste como SigmaFrag)
COLOR_CONTORNO_BGR = (220, 180, 40)   # celeste

# ──────────────────────────────────────────────────────────────────────────
#  ESTILOS CSS
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }
    .section-title {
        font-family: 'Syne', sans-serif; font-weight: 700; font-size: 16px;
        color: #1f2937; margin: 18px 0 10px 0; padding-bottom: 6px;
        border-bottom: 2px solid #e5e7eb;
    }
    div[data-testid="stMetric"] {
        background: #f9fafb; border: 1px solid #e5e7eb;
        border-radius: 10px; padding: 12px 16px;
    }
    .stButton > button { border-radius: 8px; font-weight: 600; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
#  FUNCIONES AUXILIARES DE ESCALA
# ──────────────────────────────────────────────────────────────────────────
def escala_desde_ancho(ancho_real, ancho_px):
    """Factor px → unidad real usando ancho total de la imagen."""
    return ancho_real / ancho_px


def escala_desde_2puntos(p1, p2, distancia_real):
    """Factor px → unidad real usando dos puntos marcados en la imagen."""
    dist_px = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if dist_px < 1:
        return None
    return distancia_real / dist_px

# ──────────────────────────────────────────────────────────────────────────
#  CARGA DEL MODELO (cacheada)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def cargar_modelo(ruta):
    from ultralytics import YOLO
    return YOLO(ruta)

# ──────────────────────────────────────────────────────────────────────────
#  PROCESAMIENTO PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────
def procesar(imagen_bgr, escala_px_a_real, conf, iou, modelo_path):
    """
    Segmenta con YOLO, calcula diámetro equivalente y asigna banda SigmaFrag
    basada en percentiles calculados sobre los propios datos.
    """
    modelo = cargar_modelo(modelo_path)
    alto_px, ancho_px = imagen_bgr.shape[:2]

    resultados = modelo.predict(source=imagen_bgr, conf=conf, iou=iou, verbose=False)

    if not resultados or resultados[0].masks is None:
        return None, None

    r = resultados[0]
    masks = r.masks.data.cpu().numpy()

    fragmentos_raw = []
    for i, mask in enumerate(masks):
        mask_r = cv2.resize(
            mask.astype(np.uint8), (ancho_px, alto_px), interpolation=cv2.INTER_NEAREST
        )
        area_px = float(mask_r.sum())
        if area_px <= 0:
            continue
        diam_px  = 2.0 * np.sqrt(area_px / np.pi)
        diam_real = diam_px * escala_px_a_real

        contornos, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contorno = max(contornos, key=cv2.contourArea) if contornos else None

        fragmentos_raw.append({
            "id": i,
            "area_px": area_px,
            "diam_equiv": diam_real,
            "contorno": contorno,
            "mask": mask_r,
        })

    if not fragmentos_raw:
        return None, None

    # ── Calcular umbrales de percentil sobre los diámetros detectados ──
    diams_all = np.array([f["diam_equiv"] for f in fragmentos_raw])
    umbrales = {}   # {1: valor, 5: valor, 10: ..., etc.}
    for p in [1, 5, 10, 20, 50, 80, 90, 95]:
        umbrales[p] = float(np.percentile(diams_all, p))

    def asignar_banda(d):
        for (etiq, p_inf, p_sup, color) in SIGMAFRAG_BANDAS:
            v_inf = umbrales[p_inf] if p_inf is not None else -np.inf
            v_sup = umbrales[p_sup] if p_sup is not None else  np.inf
            if v_inf <= d < v_sup:
                return etiq, color
        # fallback al último
        etiq, _, _, color = SIGMAFRAG_BANDAS[-1]
        return etiq, color

    fragmentos = []
    for f in fragmentos_raw:
        etiq, color = asignar_banda(f["diam_equiv"])
        f["banda"] = etiq
        f["color"] = color
        fragmentos.append(f)

    curva = calcular_curva(fragmentos)
    return fragmentos, curva


def calcular_curva(fragmentos):
    diams = np.array([f["diam_equiv"] for f in fragmentos])
    areas = np.array([f["area_px"]    for f in fragmentos])
    orden = np.argsort(diams)
    diams_ord = diams[orden]
    areas_ord = areas[orden]
    acum = np.cumsum(areas_ord)
    pct  = 100.0 * acum / acum[-1]

    curva = {"diametros": diams_ord, "pct_pasante": pct}
    for p in [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]:
        idx = min(np.searchsorted(pct, p), len(diams_ord) - 1)
        curva[f"D{p}"] = float(diams_ord[idx])
    return curva

# ──────────────────────────────────────────────────────────────────────────
#  FIGURAS
# ──────────────────────────────────────────────────────────────────────────
def fig_segmentacion(img_bgr, fragmentos):
    """
    Overlay SigmaFrag: relleno semitransparente + contorno celeste.
    """
    overlay = img_bgr.copy()
    mask_color = np.zeros_like(img_bgr, dtype=np.uint8)

    for f in fragmentos:
        if f["mask"] is not None:
            color_bgr = f["color"]
            for c in range(3):
                mask_color[:, :, c][f["mask"].astype(bool)] = color_bgr[c]

    # Mezcla semitransparente (alpha = 0.55)
    cv2.addWeighted(mask_color, 0.55, overlay, 0.45, 0, overlay)

    # Contorno celeste encima
    for f in fragmentos:
        if f["contorno"] is not None:
            cv2.drawContours(overlay, [f["contorno"]], -1, COLOR_CONTORNO_BGR, thickness=2)

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(overlay_rgb)
    ax.axis("off")

    # Leyenda con los colores SigmaFrag
    handles = []
    bandas_presentes = {f["banda"] for f in fragmentos}
    for etiq, _, _, color in SIGMAFRAG_BANDAS:
        if etiq in bandas_presentes:
            handles.append(Patch(facecolor=np.array(color[::-1]) / 255, label=etiq))
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.9, title="Bandas (SigmaFrag)", title_fontsize=8)
    fig.tight_layout()
    return fig


def fig_curva(curva, unidad):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(curva["diametros"], curva["pct_pasante"], color="#1e3a5f", linewidth=2.5)

    lineas_ref = [("D50", "#6b7280"), ("D80", "#e67e22"), ("D20", "#27ae60")]
    for key, col in lineas_ref:
        if key in curva:
            v = curva[key]
            p = int(key[1:])
            ax.axvline(v, color=col, linestyle="--", linewidth=1.2, alpha=0.8)
            ax.axhline(p, color=col, linestyle=":", linewidth=0.8, alpha=0.5)
            ax.text(v, p + 2, f" {key}={v:.2f}", fontsize=8, color=col, va="bottom")

    ax.set_xlabel(f"Diámetro equivalente ({unidad})", fontsize=10)
    ax.set_ylabel("% Pasante acumulado", fontsize=10)
    ax.set_ylim(0, 108)
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Curva Granulométrica — Estilo SigmaFrag", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_histograma(fragmentos, unidad):
    diams = np.array([f["diam_equiv"] for f in fragmentos])
    fig, ax = plt.subplots(figsize=(10, 4))
    n, bins, patches = ax.hist(diams, bins=28, edgecolor="white", linewidth=0.6)

    # Colorear cada bin con la banda SigmaFrag correspondiente
    diams_all = np.array([f["diam_equiv"] for f in fragmentos])
    umbrales = {p: float(np.percentile(diams_all, p)) for p in [1,5,10,20,50,80,90,95]}

    def color_bin(d):
        for etiq, p_inf, p_sup, color in SIGMAFRAG_BANDAS:
            v_inf = umbrales[p_inf] if p_inf is not None else -np.inf
            v_sup = umbrales[p_sup] if p_sup is not None else np.inf
            if v_inf <= d < v_sup:
                return np.array(color[::-1]) / 255
        return np.array(SIGMAFRAG_BANDAS[-1][3][::-1]) / 255

    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(color_bin(left))

    ax.set_xlabel(f"Diámetro equivalente ({unidad})", fontsize=10)
    ax.set_ylabel("N° de fragmentos", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    handles = [
        Patch(facecolor=np.array(c[::-1])/255, label=e)
        for e, _, _, c in SIGMAFRAG_BANDAS
    ]
    ax.legend(handles=handles, fontsize=8, ncol=2, title="Bandas", title_fontsize=8)
    ax.set_title("Histograma de Tamaños por Banda SigmaFrag", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────
#  COMPONENTE DE CALIBRACIÓN POR 2 PUNTOS (canvas HTML + streamlit component)
# ──────────────────────────────────────────────────────────────────────────
def canvas_calibracion(imagen_bgr, key="calib"):
    """
    Muestra la imagen en un canvas HTML interactivo donde el usuario puede
    hacer click en 2 puntos. Devuelve (p1, p2) en coordenadas de píxel
    originales, o (None, None) si aún no hay 2 puntos marcados.
    """
    alto_orig, ancho_orig = imagen_bgr.shape[:2]

    # Codificar imagen como PNG base64
    _, buf = cv2.imencode(".png", imagen_bgr)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    # Ancho del canvas (responsivo, máx 720px)
    canvas_w = 720
    canvas_h = int(canvas_w * alto_orig / ancho_orig)

    html_code = f"""
    <div style="font-family:Inter,sans-serif;font-size:13px;margin-bottom:6px;">
        <b>Calibración:</b> haz clic en 2 puntos de referencia sobre la imagen.<br>
        <span style="color:#6b7280;">Punto 1: 🔴 &nbsp; Punto 2: 🔵</span>
    </div>
    <canvas id="calib_canvas_{key}" width="{canvas_w}" height="{canvas_h}"
        style="border:2px solid #d1d5db;border-radius:8px;cursor:crosshair;max-width:100%;"></canvas>
    <div id="status_{key}" style="margin-top:6px;font-size:12px;color:#374151;"></div>
    <button id="reset_{key}"
        style="margin-top:6px;padding:4px 12px;border-radius:6px;border:1px solid #d1d5db;
               background:#f9fafb;cursor:pointer;font-size:12px;">
        Reiniciar puntos
    </button>
    <input type="hidden" id="pts_out_{key}" value="">

    <script>
    (function() {{
        const canvas  = document.getElementById("calib_canvas_{key}");
        const ctx     = canvas.getContext("2d");
        const status  = document.getElementById("status_{key}");
        const resetBtn= document.getElementById("reset_{key}");
        const ptsOut  = document.getElementById("pts_out_{key}");
        const scaleX  = {ancho_orig} / {canvas_w};
        const scaleY  = {alto_orig}  / {canvas_h};
        let pts = [];
        let img = new Image();
        img.onload = function() {{ ctx.drawImage(img, 0, 0, {canvas_w}, {canvas_h}); }};
        img.src = "data:image/png;base64,{img_b64}";

        function redraw() {{
            ctx.drawImage(img, 0, 0, {canvas_w}, {canvas_h});
            pts.forEach(function(p, i) {{
                ctx.beginPath();
                ctx.arc(p[0], p[1], 7, 0, 2*Math.PI);
                ctx.fillStyle = i === 0 ? "rgba(220,40,40,0.85)" : "rgba(40,90,220,0.85)";
                ctx.fill();
                ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
                ctx.fillStyle = "#fff"; ctx.font = "bold 12px Inter,sans-serif";
                ctx.fillText("P" + (i+1), p[0]+10, p[1]-6);
            }});
            if (pts.length === 2) {{
                ctx.beginPath();
                ctx.moveTo(pts[0][0], pts[0][1]);
                ctx.lineTo(pts[1][0], pts[1][1]);
                ctx.strokeStyle = "#facc15"; ctx.lineWidth = 2;
                ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);
            }}
        }}

        canvas.addEventListener("click", function(e) {{
            if (pts.length >= 2) return;
            const rect = canvas.getBoundingClientRect();
            const cx = (e.clientX - rect.left) * ({canvas_w} / rect.width);
            const cy = (e.clientY - rect.top)  * ({canvas_h} / rect.height);
            pts.push([cx, cy]);
            redraw();
            if (pts.length === 1) {{
                status.textContent = "Punto 1 marcado. Haz clic en el Punto 2.";
            }} else {{
                const px1 = Math.round(pts[0][0] * scaleX);
                const py1 = Math.round(pts[0][1] * scaleY);
                const px2 = Math.round(pts[1][0] * scaleX);
                const py2 = Math.round(pts[1][1] * scaleY);
                ptsOut.value = JSON.stringify([px1, py1, px2, py2]);
                status.textContent = "✅ 2 puntos marcados. Ingresa la distancia real abajo.";
                // Enviar al iframe padre via postMessage
                const data = {{ type: "calib_pts_{key}", value: ptsOut.value }};
                window.parent.postMessage(JSON.stringify(data), "*");
            }}
        }});

        resetBtn.addEventListener("click", function() {{
            pts = []; ptsOut.value = ""; status.textContent = "Puntos reiniciados.";
            redraw();
            window.parent.postMessage(JSON.stringify({{type:"calib_pts_{key}", value:""}}), "*");
        }});
    }})();
    </script>
    """
    st.components.v1.html(html_code, height=canvas_h + 100, scrolling=False)

    # Recibir coordenadas vía session_state (el usuario las copia manualmente
    # si el navegador bloquea postMessage — alternativa robusta)
    st.caption("Si los puntos no se detectan automáticamente, ingresa las coordenadas manualmente:")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        x1 = st.number_input("X del Punto 1 (px)", min_value=0, max_value=ancho_orig, value=0, key=f"x1_{key}")
        y1 = st.number_input("Y del Punto 1 (px)", min_value=0, max_value=alto_orig,  value=0, key=f"y1_{key}")
    with col_p2:
        x2 = st.number_input("X del Punto 2 (px)", min_value=0, max_value=ancho_orig, value=0, key=f"x2_{key}")
        y2 = st.number_input("Y del Punto 2 (px)", min_value=0, max_value=alto_orig,  value=0, key=f"y2_{key}")

    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return None, None
    if (x1, y1) == (x2, y2):
        return None, None
    return (x1, y1), (x2, y2)


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Parámetros")
    st.caption(f"Sesión: **{st.session_state.get('usuario_actual', '')}**")

    unidad = st.selectbox("Unidad de medida", ["m", "cm", "mm", "in", "ft"], index=0)

    st.markdown("---")
    st.markdown("**Método de escala**")
    metodo_escala = st.radio(
        "¿Cómo definir la escala?",
        ["Ancho total de la imagen", "Dos puntos de calibración"],
        index=0,
    )

    if metodo_escala == "Ancho total de la imagen":
        ancho_real = st.number_input(
            f"Ancho real de la escena ({unidad})",
            min_value=0.001, value=1.0, step=0.1,
            help="Ancho real en la unidad elegida que cubre el ancho completo de la imagen."
        )
    else:
        distancia_real_calib = st.number_input(
            f"Distancia real entre los 2 puntos ({unidad})",
            min_value=0.001, value=0.5, step=0.05,
            help="Introduce la distancia conocida entre los dos puntos que marcarás en la imagen."
        )

    st.markdown("---")
    conf = st.slider("Confianza (conf)", 0.05, 0.95, 0.25, 0.05)
    iou  = st.slider("IoU",             0.05, 0.95, 0.45, 0.05)

    st.markdown("---")
    st.markdown(
        """**Bandas SigmaFrag (automáticas)**  
        Los percentiles se calculan sobre los fragmentos detectados en cada imagen."""
    )
    for etiq, p_inf, p_sup, color in SIGMAFRAG_BANDAS:
        rgb = tuple(int(v) for v in color[::-1])
        st.markdown(
            f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;"
            f"background:rgb{rgb};vertical-align:middle;margin-right:6px;'></span>{etiq}",
            unsafe_allow_html=True,
        )

    if st.button("Cerrar sesión", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.caption("© FragmentApp — Autor: Juan Carlos Lermo Layme")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
st.markdown("# Fragmentometría de Roca")
st.markdown(
    "<p style='color:#6b7280;margin-top:-12px'>Análisis automático · Estilo SigmaFrag · Percentiles automáticos</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

if not modelo_listo:
    st.warning("El modelo no está disponible. Recarga la página o revisa el enlace de Google Drive.")
    st.stop()

# ── Carga de imagen ───────────────────────────────────────────────────────
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
        st.image(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB), caption="Imagen cargada", use_container_width=True)

with tab_camara:
    foto = st.camera_input("Toma una foto de la voladura")
    if foto:
        datos = np.frombuffer(foto.read(), np.uint8)
        imagen_bgr = cv2.imdecode(datos, cv2.IMREAD_COLOR)

st.markdown("---")

# ── Calibración por 2 puntos (si aplica) ──────────────────────────────────
escala_px_a_real = None

if imagen_bgr is not None:
    alto_px, ancho_px = imagen_bgr.shape[:2]

    if metodo_escala == "Ancho total de la imagen":
        escala_px_a_real = escala_desde_ancho(ancho_real, ancho_px)
        st.info(f"📏 Escala: 1 px = {escala_px_a_real:.6f} {unidad}")

    else:
        st.markdown("<div class='section-title'>📍 Calibración por 2 Puntos</div>", unsafe_allow_html=True)
        st.markdown(
            "Marca dos puntos sobre la imagen cuya distancia real conoces. "
            "Puedes usar el canvas de abajo o ingresar las coordenadas manualmente."
        )
        p1, p2 = canvas_calibracion(imagen_bgr, key="main")

        if p1 is not None and p2 is not None:
            escala_px_a_real = escala_desde_2puntos(p1, p2, distancia_real_calib)
            if escala_px_a_real:
                dist_px = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
                st.success(
                    f"✅ Puntos: P1={p1}, P2={p2} | "
                    f"Distancia en px: {dist_px:.1f} px → "
                    f"1 px = {escala_px_a_real:.6f} {unidad}"
                )
            else:
                st.warning("Los dos puntos son idénticos; ajusta las coordenadas.")
        else:
            st.info("Marca los 2 puntos de calibración para continuar.")

    st.markdown("---")

    # ── Botón de análisis ─────────────────────────────────────────────────
    boton_disabled = (escala_px_a_real is None)
    if st.button("⚡  ANALIZAR FRAGMENTACIÓN", disabled=boton_disabled):
        if not Path(MODELO_PATH).exists():
            st.error(f"No se encontró el modelo en: **{MODELO_PATH}**")
        else:
            with st.spinner("Ejecutando segmentación YOLO..."):
                fragmentos, curva = procesar(
                    imagen_bgr, escala_px_a_real, conf, iou, MODELO_PATH
                )

            if fragmentos is None:
                st.error("No se detectaron fragmentos. Ajusta Confianza o IoU.")
            else:
                st.session_state.resultados = (fragmentos, curva, escala_px_a_real, imagen_bgr.copy())

# ── Resultados ────────────────────────────────────────────────────────────
if "resultados" in st.session_state:
    fragmentos, curva, escala_px, img_orig = st.session_state.resultados
    diams = [f["diam_equiv"] for f in fragmentos]

    st.markdown("## Resultados")

    # Métricas clave
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("N° fragmentos",          f"{len(fragmentos)}")
    col2.metric(f"D20 ({unidad})",        f"{curva.get('D20', 0):.3f}")
    col3.metric(f"D50 ({unidad})",        f"{curva.get('D50', 0):.3f}")
    col4.metric(f"D80 ({unidad})",        f"{curva.get('D80', 0):.3f}")
    col5.metric(f"D95 ({unidad})",        f"{curva.get('D95', 0):.3f}")

    # Segmentación + Curva
    col_img, col_curva = st.columns([1.15, 1])

    with col_img:
        st.markdown("<div class='section-title'>Segmentación SigmaFrag</div>", unsafe_allow_html=True)
        st.pyplot(fig_segmentacion(img_orig, fragmentos), use_container_width=True)

    with col_curva:
        st.markdown("<div class='section-title'>Curva Granulométrica</div>", unsafe_allow_html=True)
        st.pyplot(fig_curva(curva, unidad), use_container_width=True)

    # Histograma
    st.markdown("<div class='section-title'>Histograma por Banda</div>", unsafe_allow_html=True)
    st.pyplot(fig_histograma(fragmentos, unidad), use_container_width=True)

    # Percentiles completos
    st.markdown("<div class='section-title'>Percentiles Completos</div>", unsafe_allow_html=True)
    percdf = pd.DataFrame([
        {"Percentil": k, f"Valor ({unidad})": f"{v:.4f}"}
        for k, v in curva.items() if k.startswith("D")
    ])
    st.dataframe(percdf, use_container_width=True, hide_index=True)

    # Estadísticas
    st.markdown("<div class='section-title'>Estadísticas Avanzadas</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mínimo",        f"{np.min(diams):.3f} {unidad}")
    c2.metric("Máximo",        f"{np.max(diams):.3f} {unidad}")
    c3.metric("Media",         f"{np.mean(diams):.3f} {unidad}")
    c4.metric("Desv. estándar",f"{np.std(diams):.3f} {unidad}")

    # % por banda
    st.markdown("<div class='section-title'>Distribución por Banda</div>", unsafe_allow_html=True)
    conteo = {}
    for f in fragmentos:
        conteo[f["banda"]] = conteo.get(f["banda"], 0) + 1
    tabla_bandas = pd.DataFrame([
        {"Banda": k, "N° fragmentos": v, "% del total": f"{100*v/len(fragmentos):.1f}%"}
        for k, v in sorted(conteo.items(), key=lambda x: list(b[0] for b in SIGMAFRAG_BANDAS).index(x[0]))
    ])
    st.dataframe(tabla_bandas, use_container_width=True, hide_index=True)

    # Descarga CSV
    df = pd.DataFrame([
        {
            "id":               f["id"],
            "diametro_equiv":   f["diam_equiv"],
            "unidad":           unidad,
            "area_px":          f["area_px"],
            "banda_sigmafrag":  f["banda"],
        }
        for f in fragmentos
    ])
    st.download_button(
        label="⬇  Descargar CSV completo",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="fragmentometria_sigmafrag.csv",
        mime="text/csv",
    )

else:
    if imagen_bgr is None:
        st.markdown(
            """
            <div style="text-align:center;padding:60px 20px;color:#6b7280;">
                <div style="font-size:48px;margin-bottom:16px;">🪨</div>
                <div style="font-family:'Syne',sans-serif;font-size:20px;color:#9ca3af;margin-bottom:8px;">
                    Carga una imagen o toma una foto para comenzar
                </div>
                <div style="font-size:13px;">Soporta JPG · PNG · WEBP · Cámara en vivo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """<div style="text-align:center;color:#9ca3af;font-size:11px;margin-top:40px;">
        FragmentApp · Estilo SigmaFrag — Autor: Juan Carlos Lermo Layme</div>""",
    unsafe_allow_html=True,
)
