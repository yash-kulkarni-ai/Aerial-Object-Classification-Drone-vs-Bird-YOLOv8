"""
Aerial Object Classification — Drone vs Bird
Streamlit deployment app using YOLOv8 classification model.
Run: streamlit run app.py
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import time

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Drone vs Bird Classifier",
    page_icon="🚁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Hide default Streamlit header/footer */
    #MainMenu, footer, header { visibility: hidden; }

    /* Banner */
    .banner {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        border-left: 5px solid #e94560;
        border-radius: 10px;
        padding: 28px 30px;
        margin-bottom: 24px;
    }
    .banner h1 { color: #f0f6fc; font-size: 1.9rem; margin: 0; }
    .banner p  { color: #8b949e; margin: 8px 0 0; font-size: 0.97rem; }

    /* Result card */
    .result-card {
        border-radius: 10px;
        padding: 22px 26px;
        margin-top: 18px;
        text-align: center;
    }
    .result-bird  { background: #0d2b1a; border: 2px solid #27ae60; }
    .result-drone { background: #0d1e2b; border: 2px solid #2980b9; }
    .result-card .label {
        font-size: 2rem; font-weight: 700; letter-spacing: 2px;
    }
    .result-bird  .label { color: #2ecc71; }
    .result-drone .label { color: #3498db; }
    .result-card .conf {
        font-size: 1.15rem; color: #adb5bd; margin-top: 6px;
    }

    /* Confidence bar track */
    .conf-bar-bg {
        background: #21262d; border-radius: 8px;
        height: 14px; margin-top: 14px; overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%; border-radius: 8px;
        transition: width 0.4s ease;
    }

    /* Upload zone */
    .uploadedFile { border: 2px dashed #30363d !important; border-radius: 10px; }

    /* Info pills */
    .pill {
        display: inline-block; padding: 3px 12px;
        border-radius: 20px; font-size: 0.8rem;
        margin: 3px; font-weight: 600;
    }
    .pill-green { background:#1a3a2a; color:#2ecc71; border:1px solid #27ae60; }
    .pill-blue  { background:#0d2035; color:#3498db; border:1px solid #2980b9; }
    .pill-gray  { background:#21262d; color:#8b949e; border:1px solid #30363d; }
</style>
""", unsafe_allow_html=True)


# ─── Load model (cached so it loads only once) ────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str = "best.pt") -> YOLO:
    return YOLO(weights_path)


# ─── Draw prediction overlay on image ────────────────────────────────────────
def annotate_image(img: Image.Image, label: str, confidence: float) -> Image.Image:
    """Draw a labelled banner at the top of the image."""
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    color = (39, 174, 96) if label.lower() == "bird" else (41, 128, 185)

    # Banner background
    banner_h = max(36, img.height // 12)
    draw.rectangle([(0, 0), (img.width, banner_h)], fill=(*color, 200))

    # Label text (use default font; PIL truetype optional)
    text = f"{label.upper()}  {confidence:.1f}%"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                   size=banner_h - 8)
    except Exception:
        font = ImageFont.load_default()

    # Centre text
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (img.width - tw) // 2
    ty = (banner_h - th) // 2
    draw.text((tx, ty), text, fill="white", font=font)

    return img


# ─── Run inference ────────────────────────────────────────────────────────────
def predict(model: YOLO, img: Image.Image):
    """Return (label, confidence, class_index) from YOLOv8-cls."""
    results    = model.predict(source=img, verbose=False)
    probs      = results[0].probs
    top_idx    = int(probs.top1)
    confidence = float(probs.top1conf) * 100
    label      = results[0].names[top_idx].capitalize()
    return label, confidence, top_idx


# ─── Confidence bar HTML ──────────────────────────────────────────────────────
def confidence_bar(confidence: float, label: str) -> str:
    color = "#27ae60" if label.lower() == "bird" else "#2980b9"
    return f"""
    <div class='conf-bar-bg'>
        <div class='conf-bar-fill'
             style='width:{confidence:.1f}%; background:{color};'></div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════════
#  APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='banner'>
  <h1>🚁🦅 Aerial Object Classifier</h1>
  <p>Upload an aerial image to classify it as a <strong>Bird</strong> or a
     <strong>Drone</strong> using a YOLOv8 deep learning model trained on
     real-world aerial imagery.</p>
</div>
""", unsafe_allow_html=True)

# ── Dataset info pills ────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:20px'>
  <span class='pill pill-green'>🦅 Bird</span>
  <span class='pill pill-blue'>🚁 Drone</span>
  <span class='pill pill-gray'>YOLOv8 Classification</span>
  <span class='pill pill-gray'>224 × 224 input</span>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model weights…"):
    try:
        model = load_model("best.pt")
    except Exception as e:
        st.error(f"❌ Could not load `best.pt`: {e}")
        st.stop()

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    label="Upload an aerial image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Supported formats: JPG, PNG, WEBP, BMP",
)

if uploaded_file is not None:

    # ── Read image ────────────────────────────────────────────────────────────
    img = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = img.size

    # ── Two-column layout: original | annotated ───────────────────────────────
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown("**📸 Uploaded Image**")
        st.image(img, use_container_width=True)
        st.caption(f"{orig_w} × {orig_h} px  ·  {uploaded_file.type}")

    # ── Run inference ─────────────────────────────────────────────────────────
    with st.spinner("Running classification…"):
        t_start     = time.time()
        label, conf, idx = predict(model, img)
        latency_ms  = (time.time() - t_start) * 1000

    # ── Annotated image ───────────────────────────────────────────────────────
    annotated = annotate_image(img, label, conf)

    with col2:
        st.markdown("**🔍 Prediction Overlay**")
        st.image(annotated, use_container_width=True)
        st.caption(f"Inference: {latency_ms:.0f} ms")

    # ── Result card ───────────────────────────────────────────────────────────
    card_class = "result-bird" if label.lower() == "bird" else "result-drone"
    icon       = "🦅" if label.lower() == "bird" else "🚁"

    st.markdown(f"""
    <div class='result-card {card_class}'>
        <div class='label'>{icon} &nbsp; {label.upper()}</div>
        <div class='conf'>Confidence: <strong>{conf:.2f}%</strong></div>
        {confidence_bar(conf, label)}
    </div>
    """, unsafe_allow_html=True)

    # ── Detailed metrics expander ──────────────────────────────────────────────
    with st.expander("📊 Detailed Probabilities", expanded=False):
        results_full = model.predict(source=img, verbose=False)
        probs_all    = results_full[0].probs.data.tolist()
        names        = results_full[0].names

        prob_data = {
            "Class"      : [names[i].capitalize() for i in range(len(names))],
            "Probability": [f"{p*100:.4f}%" for p in probs_all],
        }

        st.table(prob_data)

        # ── Download annotated image ──────────────────────────────────────────
        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="⬇️  Download annotated image",
            data=buf.getvalue(),
            file_name=f"prediction_{label.lower()}_{conf:.0f}pct.jpg",
            mime="image/jpeg",
        )

else:
    # ── Placeholder when no image uploaded ───────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:50px 20px;
                border:2px dashed #30363d; border-radius:10px;
                color:#8b949e; background:#161b22; margin-top:10px;'>
        <div style='font-size:3rem'>📡</div>
        <div style='font-size:1.1rem; margin-top:10px; font-weight:600'>
            Upload an image to begin classification
        </div>
        <div style='font-size:0.85rem; margin-top:6px; color:#6e7681'>
            Drag &amp; drop or click the uploader above
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#484f58; font-size:0.8rem; padding:12px 0'>
    Aerial Object Classifier &nbsp;·&nbsp; YOLOv8 &nbsp;·&nbsp;
    Built with Streamlit &nbsp;·&nbsp; Model: <code>best.pt</code>
</div>
""", unsafe_allow_html=True)
