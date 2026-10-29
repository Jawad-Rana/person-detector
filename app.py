"""
Person Detection System — Faster R-CNN (ResNet50-FPN)
Streamlit web app: upload an image, the trained model draws a box around every person.
"""
import io
import os
import time

import numpy as np
import streamlit as st
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ------------------------------------------------------------------
# Project details (edit these to your own names)
# ------------------------------------------------------------------
PROJECT_TITLE = "Person Detection System"
PROJECT_SUBTITLE = "Deep-learning object detection with Faster R-CNN"
DEVELOPER = "Rana Jawad"
COURSE = "Computer Vision Project"
DATASET_URL = "https://drive.google.com/drive/folders/10d_w51uRXeIsoWih5UTTd6MxENb0CRpr?usp=drive_link"
MODEL_PATH = "rcnn_model.pth"
MODEL_DRIVE_ID = "1M0uYhKeZBYrOL7uYcsD7xxU8OUtHaunz"  # rcnn_model.pth on Google Drive (shared: anyone with link)
CLASS_NAME = "Person"
BOX_COLOR = (34, 211, 238)      # cyan
MAX_SIDE = 1333                 # big images are scaled down for speed

st.set_page_config(page_title=PROJECT_TITLE, page_icon="🎯", layout="wide")

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Poppins', sans-serif; }
.block-container { padding-top: 3.2rem; max-width: 1200px; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }

.hero {
  position: relative; overflow: hidden;
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0891b2 100%);
  border-radius: 22px; padding: 38px 40px; color: #fff; margin-bottom: 26px;
  box-shadow: 0 18px 40px rgba(15, 23, 42, .28);
}
.hero::after {
  content: ""; position: absolute; right: -80px; top: -80px; width: 280px; height: 280px;
  border-radius: 50%; background: radial-gradient(circle, rgba(34,211,238,.45), transparent 70%);
}
.hero .tag {
  display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
  padding: 5px 14px; border-radius: 999px; font-size: 12px; letter-spacing: .08em; text-transform: uppercase;
}
.hero h1 { color: #fff; font-size: 40px; font-weight: 700; margin: 14px 0 6px; line-height: 1.15; }
.hero p { color: #cbd5e1; font-size: 16px; margin: 0; }
.chips { margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap; }
.ds-btn {
  position: relative; z-index: 1; display: inline-block; margin-top: 22px;
  background: #22d3ee; color: #0f172a !important; text-decoration: none !important;
  font-weight: 600; font-size: 15px; padding: 11px 20px; border-radius: 12px;
  box-shadow: 0 8px 20px rgba(34,211,238,.35); transition: transform .15s, box-shadow .15s;
}
.ds-btn:hover { transform: translateY(-2px); box-shadow: 0 12px 26px rgba(34,211,238,.5); }
section[data-testid="stSidebar"] a { color: #67e8f9 !important; }
.chip { background: rgba(15,23,42,.45); border: 1px solid rgba(148,163,184,.35);
        padding: 6px 12px; border-radius: 10px; font-size: 13px; color: #e2e8f0; }

.metric-card {
  background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 18px 20px;
  box-shadow: 0 4px 14px rgba(15,23,42,.06); border-top: 4px solid #0891b2;
}
.metric-card .label { color: #64748b; font-size: 13px; font-weight: 500; }
.metric-card .value { color: #0f172a; font-size: 30px; font-weight: 700; margin-top: 2px; }

.panel-title { font-weight: 600; color: #0f172a; font-size: 17px; margin: 6px 0 10px; }
.empty {
  border: 2px dashed #94a3b8; border-radius: 18px; padding: 46px 30px; text-align: center;
  background: linear-gradient(180deg, #f8fafc, #eef2ff); color: #334155;
}
.empty h3 { margin: 8px 0 6px; color: #0f172a; }
.steps { display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; margin-top: 18px; }
.step { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 16px; font-size: 14px; }
.step b { color: #0891b2; }

section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .side-card {
  background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
  border-radius: 12px; padding: 12px 14px; font-size: 13px; margin-bottom: 10px;
}
.footer { text-align: center; color: #64748b; font-size: 13px; margin-top: 40px;
          padding-top: 18px; border-top: 1px solid #e2e8f0; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained model…")
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # background + person
    if not os.path.exists(MODEL_PATH):  # on the cloud the model is fetched from Google Drive once
        import gdown
        gdown.download(id=MODEL_DRIVE_ID, output=MODEL_PATH, quiet=True)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def detect(model, image, threshold):
    tensor = T.ToTensor()(image).unsqueeze(0)
    start = time.perf_counter()
    with torch.no_grad():
        out = model(tensor)[0]
    elapsed = time.perf_counter() - start
    keep = out["scores"] >= threshold
    return out["boxes"][keep].numpy(), out["scores"][keep].numpy(), elapsed


def get_font(size):
    for name in ("DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_boxes(image, boxes, scores, show_scores=True):
    img = image.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    line = max(2, round(min(w, h) / 220))
    font = get_font(max(12, round(min(w, h) / 38)))
    for i, (box, score) in enumerate(zip(boxes, scores), start=1):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], fill=BOX_COLOR + (28,))
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR + (255,), width=line)
        text = f"{CLASS_NAME} {i}" + (f"  {score * 100:.0f}%" if show_scores else "")
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = max(3, line + 1)
        ty = y1 - th - 2 * pad if y1 - th - 2 * pad > 0 else y1
        draw.rectangle([x1, ty, x1 + tw + 2 * pad, ty + th + 2 * pad], fill=BOX_COLOR + (255,))
        draw.text((x1 + pad, ty + pad - tb[1]), text, fill=(15, 23, 42), font=font)
    return img


def metric(col, label, value):
    col.markdown(
        f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎯 Control Panel")
    threshold = st.slider("Confidence threshold", 0.10, 0.95, 0.50, 0.05,
                          help="Only detections above this confidence are shown.")
    show_scores = st.toggle("Show confidence on boxes", value=True)
    st.markdown("---")
    st.markdown("### 🧠 Model")
    st.markdown(
        """
<div class="side-card"><b>Architecture</b><br>Faster R-CNN · ResNet50-FPN</div>
<div class="side-card"><b>Framework</b><br>PyTorch / Torchvision</div>
<div class="side-card"><b>Detects</b><br>Person (all people in the image)</div>
<div class="side-card"><b>Training</b><br>Fine-tuned on a custom person dataset</div>
<div class="side-card"><b>Dataset</b><br><a href="{DATASET_URL}" target="_blank" rel="noopener">📂 Open dataset on Google Drive ↗</a></div>
""".replace("{DATASET_URL}", DATASET_URL),
        unsafe_allow_html=True,
    )
    st.markdown("### 📌 How it works")
    st.caption("1. Region Proposal Network suggests candidate regions.\n\n"
               "2. ROI head classifies each region and refines the box.\n\n"
               "3. Boxes above the threshold are drawn on the image.")

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    f"""
<div class="hero">
  <span class="tag">AI · Computer Vision</span>
  <h1>{PROJECT_TITLE}</h1>
  <p>{PROJECT_SUBTITLE}. Upload a photo and every person is located with a bounding box.</p>
  <div class="chips">
    <span class="chip">⚡ Faster R-CNN</span><span class="chip">🧱 ResNet50-FPN</span>
    <span class="chip">🔥 PyTorch</span><span class="chip">👤 Multi-person detection</span>
  </div>
  <a class="ds-btn" href="{DATASET_URL}" target="_blank" rel="noopener">📂 View Our Dataset (Google Drive) ↗</a>
</div>
""",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.markdown(
        """
<div class="empty">
  <div style="font-size:46px">🖼️</div>
  <h3>Upload an image to start detection</h3>
  <div>Best results with clear photos of people — streets, classrooms, groups, sports.</div>
  <div class="steps">
    <div class="step"><b>1</b> · Upload image</div>
    <div class="step"><b>2</b> · Adjust threshold</div>
    <div class="step"><b>3</b> · Download result</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
else:
    image = Image.open(uploaded).convert("RGB")
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE))

    try:
        model = load_model()
    except Exception as e:
        st.error("Could not load the model. Make sure rcnn_model.pth is shared on Google Drive "
                 f"as 'Anyone with the link'. Details: {e}")
        st.stop()

    with st.spinner("Detecting people…"):
        boxes, scores, elapsed = detect(model, image, threshold)
    result = draw_boxes(image, boxes, scores, show_scores)

    c1, c2, c3, c4 = st.columns(4)
    metric(c1, "👤 People detected", len(boxes))
    metric(c2, "📊 Average confidence", f"{scores.mean() * 100:.1f}%" if len(scores) else "—")
    metric(c3, "🏆 Highest confidence", f"{scores.max() * 100:.1f}%" if len(scores) else "—")
    metric(c4, "⏱️ Inference time", f"{elapsed:.2f}s")
    st.write("")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel-title">Original image</div>', unsafe_allow_html=True)
        st.image(image, width="stretch")
    with right:
        st.markdown('<div class="panel-title">Detection result</div>', unsafe_allow_html=True)
        st.image(result, width="stretch")

    if len(boxes) == 0:
        st.info("No person found above this threshold. Try lowering the confidence threshold in the sidebar.")
    else:
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button("⬇️ Download result image", buf.getvalue(),
                           file_name="detection_result.png", mime="image/png")
        with st.expander("📋 Detection details"):
            rows = [{"#": i, "Class": CLASS_NAME, "Confidence": f"{s * 100:.1f}%",
                     "x1": int(b[0]), "y1": int(b[1]), "x2": int(b[2]), "y2": int(b[3])}
                    for i, (b, s) in enumerate(zip(boxes, scores), start=1)]
            st.dataframe(rows, hide_index=True, width="stretch")

st.markdown(
    f'<div class="footer">{PROJECT_TITLE} · {COURSE} · Developed by <b>{DEVELOPER}</b></div>',
    unsafe_allow_html=True,
)
