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
# Si el modelo no existe físicamente en el servidor de Streamlit Cloud, lo descarga en un instante
if not Path("best.pt").exists():
    with st.spinner("Descargando el modelo YOLO entrenado desde Google Drive (Esto solo toma unos segundos)..."):
        id_drive = "1vVdvUfLMejx0JWocpTc957i0jEFmwh4j"  # <── Tu ID extraído de la URL
        url_descarga = f"https://drive.google.com/uc?id={id_drive}"
        try:
            gdown.download(url_descarga, "best.pt", quiet=False)
        except Exception as e:
            st.error(f"Error crítico al intentar descargar el modelo desde Google Drive: {e}")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.stApp { background: #0d0f12; color: #e8e6e1; }

section[data-testid="stSidebar"] {
    background: #13161b;
    border-right: 1px solid #2a2d35;
}

.metric-card {
    background: #181b22;
    border: 1px solid #2a2d35;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 10px;
}
.metric-card .label { color: #6b7280; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; }
.metric-card .value { color: #f0ede8; font-size: 26px; font-weight: 700; font-family: 'Syne', sans-serif; }
.metric-card .unit  { color: #6b7280; font-size: 12px; }

.interval-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    margin: 3px;
    letter-spacing: 1px;
}
.fino   { background: #3d1515; color: #f87171; border: 1px solid #f87171; }
.medio  { background: #0f2d1a; color: #4ade80; border: 1px solid #4ade80; }
.grueso { background: #2d1f08; color: #fb923c; border: 1px solid #fb923c; }

.upload-zone {
    border: 2px dashed #2a2d35;
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    background: #13161b;
    transition: border-color 0.3s;
}

div[data-testid="stFileUploader"] > div {
    background: #13161b !important;
    border: 2px dashed #2a2d35 !important;
    border-radius: 12px !important;
}

.stButton > button {
    background: #e8e6e1 !important;
    color: #0d0f12 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 32px !important;
    width: 100% !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { background: #ffffff !important; transform: translateY(-1px); }

.stSlider > div > div > div { background: #2a2d35 !important; }
.stSlider > div > div > div > div { background: #e8e6e1 !important; }

hr { border-color: #2a2d35 !important; }

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 16px;
    margin-top: 28px;
}

.stDataFrame { background: #13161b !important; }
table { color: #e8e6e1 !important; }
</style>
""", unsafe_allow_html=True)

# ─── INTERVALOS SigmaFrag ─────────────────────────────────────────────────────
def get_intervalos(fino_max, grueso_min):
    return [
        ("Finos",   None,       fino_max,   (220,  50,  50), "fino"),
        ("Medios",  fino_max,   grueso_min, ( 50, 180,  50), "medio"),
        ("Gruesos", grueso_min, None,       (230, 140,  30), "grueso"),
    ]

def color_intervalo(diam, intervalos):
    for _, lo, hi, rgb, _ in intervalos:
        if (lo is None or diam >= lo) and (hi is None or diam < hi):
            return rgb
    return (128, 128, 128)

def nombre_intervalo(diam, intervalos):
    for nombre, lo, hi, _, _ in intervalos:
        if (lo is None or diam >= lo) and (hi is None or diam < hi):
            return nombre
    return "Otro"

# ─── ANÁLISIS ─────────────────────────────────────────────────────────────────
def metricas_mascara(mask_bin, escala):
    contornos, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    c = max(contornos, key=cv2.contourArea)
    area_px = cv2.contourArea(c)
    perim_px = cv2.arcLength(c, True)
    if area_px < 50 or len(c) < 5:
        return None
    area_real  = area_px / escala**2
    diam_equiv = 2 * np.sqrt(area_real / np.pi)
    circ = (4 * np.pi * area_px) / perim_px**2 if perim_px > 0 else 0
    elipse = cv2.fitEllipse(c)
    (_, _), (a, b), _ = elipse
    eje_mayor = max(a, b) / escala
    eje_menor = min(a, b) / escala
    M = cv2.moments(c)
    cx = M["m10"] / M["m00"] if M["m00"] else 0
    cy = M["m01"] / M["m00"] if M["m00"] else 0
    return dict(area_real=area_real, diam_equiv=diam_equiv,
                eje_mayor=eje_mayor, eje_menor=eje_menor,
                elongacion=eje_mayor/eje_menor if eje_menor > 0 else 1,
                circularidad=circ, centroide=(cx, cy), contorno=c)

def curva_granulometrica(diametros):
    d = np.array(sorted(diametros))
    areas = np.pi * (d / 2)**2
    ac = np.cumsum(areas)
    pct = 100 * ac / ac[-1]
    def prc(p): return float(np.interp(p, pct, d))
    return {"d": d, "pct": pct,
            "D10": prc(10), "D20": prc(20), "D30": prc(30),
            "D50": prc(50), "D60": prc(60), "D70": prc(70),
            "D80": prc(80), "D90": prc(90)}

def run_yolo(imagen_path, conf, iou):
    from ultralytics import YOLO
    modelo = st.session_state.modelo
    res = modelo.predict(source=imagen_path, conf=conf, iou=iou, imgsz=1280, verbose=False)[0]
    return res

def procesar(imagen_bgr, ancho_real, conf, iou, intervalos, unidad, modelo_path):
    H, W = imagen_bgr.shape[:2]
    escala = W / ancho_real

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, imagen_bgr)
        tmp_path = tmp.name

    try:
        from ultralytics import YOLO
        if "modelo" not in st.session_state or st.session_state.modelo_path != modelo_path:
            st.session_state.modelo = YOLO(modelo_path)
            st.session_state.modelo_path = modelo_path
        res = st.session_state.modelo.predict(
            source=tmp_path, conf=conf, iou=iou, imgsz=1280, verbose=False)[0]
    finally:
        os.unlink(tmp_path)

    if res.masks is None or len(res.masks) == 0:
        return None, None, None

    fragmentos = []
    for mask in res.masks.data.cpu().numpy():
        m = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        met = metricas_mascara((m > 0.5).astype(np.uint8) * 255, escala)
        if met:
            met["intervalo"] = nombre_intervalo(met["diam_equiv"], intervalos)
            fragmentos.append(met)

    return fragmentos, curva_granulometrica([f["diam_equiv"] for f in fragmentos]), escala

def fig_segmentacion(imagen_bgr, fragmentos, intervalos):
    vis = imagen_bgr.copy()
    overlay = vis.copy()
    for frag in fragmentos:
        rgb = color_intervalo(frag["diam_equiv"], intervalos)
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.drawContours(overlay, [frag["contorno"]], -1, bgr, -1)
        cv2.drawContours(vis,      [frag["contorno"]], -1, bgr,  2)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0d0f12")
    ax.set_facecolor("#0d0f12")
    ax.imshow(vis_rgb)
    ax.axis("off")

    leyenda = [Patch(facecolor=tuple(r/255 for r in rgb), edgecolor="white", linewidth=0.5,
                     label=nombre)
               for nombre, _, _, rgb, _ in intervalos]
    leg = ax.legend(handles=leyenda, loc="lower right", fontsize=10,
                    facecolor="#181b22", edgecolor="#2a2d35", labelcolor="#e8e6e1")
    plt.tight_layout(pad=0)
    return fig

def fig_curva(curva, intervalos, unidad):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d0f12")
    ax.set_facecolor("#13161b")

    ax.semilogx(curva["d"], curva["pct"], color="#e8e6e1", linewidth=2.5, label="Curva acumulada")

    colores_prc = {"D10":"#4ade80","D50":"#fb923c","D80":"#f87171","D90":"#a78bfa"}
    for key, color in colores_prc.items():
        val = curva[key]
        p = float(key[1:])
        ax.axvline(val, linestyle="--", color=color, alpha=0.7, linewidth=1.2)
        ax.axhline(p,   linestyle=":",  color=color, alpha=0.4, linewidth=1)
        ax.annotate(f"{key}={val:.3f}{unidad}", xy=(val, p),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8.5, color=color, fontfamily="monospace")

    for nombre, lo, hi, rgb, _ in intervalos:
        x0 = lo if lo is not None else curva["d"].min() * 0.5
        x1 = hi if hi is not None else curva["d"].max() * 1.5
        ax.axvspan(x0, x1, alpha=0.08, color=tuple(r/255 for r in rgb))

    ax.set_xlabel(f"Tamaño de fragmento ({unidad})", color="#9ca3af", fontsize=11)
    ax.set_ylabel("Pasante acumulado (%)", color="#9ca3af", fontsize=11)
    ax.set_title("Curva Granulométrica", color="#e8e6e1", fontsize=13,
                 fontweight="bold", fontfamily="monospace")
    ax.set_ylim(0, 105)
    ax.tick_params(colors="#6b7280")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d35")
    ax.grid(True, which="both", linestyle="--", alpha=0.2, color="#6b7280")
    ax.legend(fontsize=9, facecolor="#181b22", edgecolor="#2a2d35", labelcolor="#e8e6e1")
    plt.tight_layout()
    return fig

def fig_histograma(fragmentos, intervalos, unidad):
    diams = [f["diam_equiv"] for f in fragmentos]
    colores_hist = []
    for d in sorted(diams):
        rgb = color_intervalo(d, intervalos)
        colores_hist.append(tuple(r/255 for r in rgb))

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0d0f12")
    ax.set_facecolor("#13161b")

    n, bins, patches = ax.hist(diams, bins=40, edgecolor="#0d0f12", linewidth=0.5)
    for patch, left_edge in zip(patches, bins[:-1]):
        rgb = color_intervalo(left_edge + (bins[1]-bins[0])/2, intervalos)
        patch.set_facecolor(tuple(r/255 for r in rgb))
        patch.set_alpha(0.85)

    ax.set_xlabel(f"Diámetro equivalente ({unidad})", color="#9ca3af", fontsize=10)
    ax.set_ylabel("Frecuencia", color="#9ca3af", fontsize=10)
    ax.set_title("Histograma de Fragmentos", color="#e8e6e1", fontsize=12, fontfamily="monospace")
    ax.tick_params(colors="#6b7280")
    for spine in ax.spines.values(): spine.set_edgecolor("#2a2d35")
    ax.grid(axis="y", linestyle="--", alpha=0.2, color="#6b7280")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🪨 FragmentApp")
    st.markdown("<p style='color:#6b7280;font-size:12px;margin-top:-10px'>Fragmentometría de Roca · v1.0</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<div class='section-title'>Modelo YOLO</div>", unsafe_allow_html=True)
    modelo_path = st.text_input("Ruta al best.pt", value="best.pt", label_visibility="collapsed",
                                 placeholder="ruta/al/best.pt")

    st.markdown("<div class='section-title'>Escala</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ancho_real = st.number_input("Ancho real", value=2.5, min_value=0.01, step=0.1, format="%.2f")
    with col2:
        unidad = st.selectbox("Unidad", ["m", "cm", "mm"], index=0)

    st.markdown("<div class='section-title'>Inferencia</div>", unsafe_allow_html=True)
    conf = st.slider("Confianza mínima", 0.1, 0.9, 0.35, 0.05)
    iou  = st.slider("IoU NMS",          0.1, 0.9, 0.45, 0.05)

    st.markdown("<div class='section-title'>Intervalos SigmaFrag</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        fino_max   = st.number_input("Fino < ", value=0.05, min_value=0.001, format="%.3f")
    with col4:
        grueso_min = st.number_input("Grueso ≥", value=0.30, min_value=0.01, format="%.3f")

    st.markdown("---")
    st.markdown("<p style='color:#6b7280;font-size:10px;text-align:center'>Estilo SigmaFrag · YOLO11</p>", unsafe_allow_html=True)

intervalos = get_intervalos(fino_max, grueso_min)


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
                 caption="Imagen cargada", use_container_width="stretch")

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
    c1, c2, c3, c4, c5 = st.columns(5)
    metricas = [
        (c1, "FRAGMENTOS", len(fragmentos), ""),
        (c2, "D50",  f"{curva['D50']:.4f}", unidad),
        (c3, "D80",  f"{curva['D80']:.4f}", unidad),
        (c4, "D90",  f"{curva['D90']:.4f}", unidad),
        (c5, "Dmax", f"{max(diams):.4f}", unidad),
    ]
    for col, label, val, u in metricas:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div class="unit">{u}</div>
            </div>""", unsafe_allow_html=True)

    # ── Badges de intervalos ──────────────────────────────────────────────────
    st.markdown("<div style='margin: 16px 0;'>", unsafe_allow_html=True)
    for nombre, lo, hi, rgb, cls in intervalos:
        grupo = [d for d in diams if (lo is None or d >= lo) and (hi is None or d < hi)]
        pct = 100 * len(grupo) / len(diams)
        rango = f"{'< '+str(hi) if lo is None else str(lo)+('–'+str(hi) if hi else '+')}"
        st.markdown(
            f'<span class="interval-badge {cls}">{nombre} ({rango} {unidad}) — '
            f'{len(grupo)} frags · {pct:.1f}%</span>',
            unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Visualización + Curva ─────────────────────────────────────────────────
    col_img, col_curva = st.columns([1.1, 1])

    with col_img:
        st.markdown("<div class='section-title'>Segmentación por Intervalo</div>", unsafe_allow_html=True)
        st.pyplot(fig_segmentacion(img_orig, fragmentos, intervalos), use_container_width="stretch")

    with col_curva:
        st.markdown("<div class='section-title'>Curva Granulométrica</div>", unsafe_allow_html=True)
        st.pyplot(fig_curva(curva, intervalos, unidad), use_container_width="stretch")

    # ── Histograma ────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Histograma</div>", unsafe_allow_html=True)
    st.pyplot(fig_histograma(fragmentos, intervalos, unidad), use_container_width="stretch")

    # ── Tabla percentiles ─────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Percentiles Completos</div>", unsafe_allow_html=True)
    percdf = pd.DataFrame([
        {"Percentil": k, f"Valor ({unidad})": f"{v:.4f}"}
        for k, v in curva.items() if k.startswith("D")
    ])
    st.dataframe(percdf, use_container_width="stretch", hide_index=True)

    # ── Estadísticas avanzadas ────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Estadísticas Avanzadas</div>", unsafe_allow_html=True)
    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Elongación media</div>
            <div class="value">{np.mean([f['elongacion'] for f in fragmentos]):.3f}</div>
            <div class="unit">1 = esférico</div>
        </div>""", unsafe_allow_html=True)
    with cb:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Circularidad media</div>
            <div class="value">{np.mean([f['circularidad'] for f in fragmentos]):.3f}</div>
            <div class="unit">1 = círculo perfecto</div>
        </div>""", unsafe_allow_html=True)
    with cc:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Área total analizada</div>
            <div class="value">{sum(f['area_real'] for f in fragmentos):.4f}</div>
            <div class="unit">{unidad}²</div>
        </div>""", unsafe_allow_html=True)

    # ── Descarga CSV ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>Exportar Datos</div>", unsafe_allow_html=True)
    df = pd.DataFrame([{
        "id": i+1,
        "intervalo": f["intervalo"],
        f"diam_equiv ({unidad})": round(f["diam_equiv"], 6),
        f"eje_mayor ({unidad})": round(f["eje_mayor"], 6),
        f"eje_menor ({unidad})": round(f["eje_menor"], 6),
        "elongacion": round(f["elongacion"], 4),
        "circularidad": round(f["circularidad"], 4),
        f"area ({unidad}²)": round(f["area_real"], 6),
    } for i, f in enumerate(fragmentos)])

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
