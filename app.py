import base64
import html
import io
import json
import time

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from advice import ADVICE

st.set_page_config(page_title="Smart Plant Disease Identifier", page_icon="🌿", layout="wide")

IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.60
CIRC = 339.3  # circumference of the gauge circle (r = 54)

CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {
  background: radial-gradient(circle at 15% 10%, rgba(46,230,166,.13), transparent 40%),
              radial-gradient(circle at 85% 80%, rgba(182,255,107,.09), transparent 45%),
              linear-gradient(135deg, #04140d, #0b2a1a 55%, #06170f);
}
.stApp, .stApp p, .stApp label, .stApp span, .stApp li {color: #e8fff2;}
.hero {text-align: center; padding: 1.6rem 1rem .6rem;}
.hero h1 {
  font-size: 2.9rem; font-weight: 800; margin: 0;
  background: linear-gradient(90deg, #7CFFB2, #2ee6a6, #b6ff6b, #7CFFB2);
  background-size: 300% 100%; -webkit-background-clip: text; background-clip: text;
  color: transparent !important; animation: flow 6s linear infinite;
}
.hero p {opacity: .75; font-size: 1.05rem; margin-top: .3rem;}
.leaf {display: inline-block; animation: float 3s ease-in-out infinite;}
@keyframes flow {to {background-position: 300% 0;}}
@keyframes float {50% {transform: translateY(-10px) rotate(10deg);}}
.card {
  background: rgba(255,255,255,.05); border: 1px solid rgba(124,255,178,.18);
  border-radius: 18px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
  backdrop-filter: blur(8px); animation: rise .7s ease both;
}
.card h4 {margin: 0 0 .5rem; color: #7CFFB2 !important; letter-spacing: .5px;}
@keyframes rise {from {opacity: 0; transform: translateY(26px);} to {opacity: 1; transform: none;}}
.d2 {animation-delay: .25s;} .d3 {animation-delay: .5s;}
.scanbox {position: relative; overflow: hidden; padding: 0; border-radius: 18px;}
.scanbox img {width: 100%; display: block;}
.glow {box-shadow: 0 0 28px rgba(46,230,166,.25);}
.scanline {
  position: absolute; left: 0; right: 0; height: 4px;
  background: linear-gradient(90deg, transparent, #7CFFB2, transparent);
  box-shadow: 0 0 18px 6px rgba(124,255,178,.6); animation: scan 1.5s ease-in-out infinite;
}
@keyframes scan {0% {top: 0;} 50% {top: 100%;} 100% {top: 0;}}
.pulse {font-size: 1.15rem; animation: pulse 1.1s ease-in-out infinite;}
@keyframes pulse {50% {opacity: .35;}}
.result {display: flex; align-items: center; gap: 1.4rem; flex-wrap: wrap;}
.badge {
  display: inline-block; padding: .25rem .8rem; border-radius: 999px;
  font-size: .8rem; font-weight: 700; letter-spacing: 1px; margin-bottom: .4rem;
}
.ok {background: rgba(46,230,166,.18); color: #2ee6a6;}
.bad {background: rgba(255,107,107,.18); color: #ff8a8a;}
.warn {background: rgba(255,200,87,.18); color: #ffc857;}
.title {font-size: 1.6rem; font-weight: 800; line-height: 1.2;}
.g {stroke-dashoffset: 339.3; animation: fill 1.5s ease forwards;}
@keyframes fill {from {stroke-dashoffset: 339.3;} to {stroke-dashoffset: var(--off);}}
.row {display: flex; justify-content: space-between; font-size: .92rem; margin-top: .6rem;}
.bar {height: 10px; border-radius: 8px; background: rgba(255,255,255,.1); overflow: hidden;}
.bar > span {
  display: block; height: 100%; width: var(--w); border-radius: 8px;
  background: linear-gradient(90deg, #2ee6a6, #b6ff6b); animation: grow 1.3s ease both;
}
@keyframes grow {from {width: 0;}}
.stat {text-align: center;}
.stat b {font-size: 2rem; color: #7CFFB2;}
[data-testid="stFileUploader"] section {
  border: 2px dashed rgba(124,255,178,.45); border-radius: 18px;
  background: rgba(255,255,255,.04); transition: .3s;
}
[data-testid="stFileUploader"] section:hover {border-color: #7CFFB2; box-shadow: 0 0 22px rgba(46,230,166,.25);}
.stTabs [data-baseweb="tab"] {font-size: 1.05rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_mobilenet.keras")


@st.cache_data
def load_classes():
    with open("class_names.json") as f:
        return json.load(f)


def parse_label(label):
    crop, condition = label.split("___")
    key = condition.strip("_")
    return crop.replace("_", " "), key, key.replace("_", " ")


def predict(image, model):
    arr = np.expand_dims(np.array(image.resize(IMG_SIZE), dtype=np.float32), axis=0)
    return model.predict(arr, verbose=0)[0]  # scaling is inside the saved model


def to_b64(image, max_side=700):
    im = image.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def image_card(b64, scanning):
    line = '<div class="scanline"></div>' if scanning else ""
    cls = "card scanbox" if scanning else "card scanbox glow"
    return f'<div class="{cls}"><img src="data:image/jpeg;base64,{b64}">{line}</div>'


def gauge(conf, color):
    off = CIRC * (1 - conf)
    return (
        '<svg width="130" height="130" viewBox="0 0 130 130">'
        '<circle cx="65" cy="65" r="54" fill="none" stroke="rgba(255,255,255,.1)" stroke-width="10"/>'
        f'<circle class="g" cx="65" cy="65" r="54" fill="none" stroke="{color}" stroke-width="10" '
        f'stroke-linecap="round" stroke-dasharray="{CIRC}" style="--off:{off:.1f}" transform="rotate(-90 65 65)"/>'
        f'<text x="65" y="72" text-anchor="middle" fill="#ffffff" font-size="22" font-weight="700">{conf*100:.1f}%</text>'
        '</svg>'
    )


def result_html(probs, class_names):
    top3 = np.argsort(probs)[::-1][:3]
    best = int(top3[0])
    conf = float(probs[best])
    crop, key, pretty = parse_label(class_names[best])
    desc, action = ADVICE.get(key, ("No description available.", "Consult a local agricultural expert."))

    if conf < CONF_THRESHOLD:
        badge, cls, color = "LOW CONFIDENCE", "warn", "#ffc857"
    elif key == "healthy":
        badge, cls, color = "HEALTHY", "ok", "#2ee6a6"
    else:
        badge, cls, color = "DISEASE DETECTED", "bad", "#ff6b6b"

    title = f"{crop}: {'Healthy' if key == 'healthy' else pretty}"
    parts = [
        '<div class="card"><div class="result">', gauge(conf, color),
        f'<div><span class="badge {cls}">{badge}</span>',
        f'<div class="title">{html.escape(title)}</div></div></div>',
    ]
    if conf < CONF_THRESHOLD:
        parts.append('<p style="margin-top:.8rem">Not sure about this one. Retake the photo: one leaf, good light, plain background.</p>')
    parts.append('</div>')
    parts.append(f'<div class="card d2"><h4>🔎 About this result</h4>{html.escape(desc)}</div>')
    parts.append(f'<div class="card d2"><h4>💊 Suggested action</h4>{html.escape(action)}</div>')

    bars = ['<div class="card d3"><h4>📊 Top 3 predictions</h4>']
    for i in top3:
        p = float(probs[i]) * 100
        name = html.escape(class_names[int(i)].replace("___", " - ").replace("_", " "))
        bars.append(f'<div class="row"><span>{name}</span><b>{p:.2f}%</b></div>'
                    f'<div class="bar"><span style="--w:{p:.2f}%"></span></div>')
    bars.append('</div>')
    parts.append("".join(bars))
    return "".join(parts)


# ---------------- UI ----------------
st.markdown(
    '<div class="hero"><h1><span class="leaf">🌿</span> Smart Plant Disease Identifier</h1>'
    '<p>Upload a leaf photo and let deep learning diagnose it in seconds(version 1.0)</p></div>',
    unsafe_allow_html=True,
)

model = load_model()
class_names = load_classes()

tab1, tab2 = st.tabs(["🔬 Diagnose", "ℹ️ About the project"])

with tab1:
    uploaded = st.file_uploader("Drop a leaf image here", type=["jpg", "jpeg", "png"])
    with st.expander("📷 Or use your camera"):
        camera = st.camera_input("Take a photo of the leaf")
    source = uploaded or camera

    if source is None:
        st.markdown(
            '<div class="card"><h4>📸 Tips for the best result</h4>'
            'Photograph a <b>single leaf</b> in good light, on a plain background, '
            'filling most of the frame.</div>', unsafe_allow_html=True)
    else:
        image = Image.open(source).convert("RGB")
        b64 = to_b64(image)
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            img_slot = st.empty()
            img_slot.markdown(image_card(b64, scanning=True), unsafe_allow_html=True)
        with right:
            out_slot = st.empty()
            out_slot.markdown('<div class="card"><div class="pulse">🔬 Analyzing leaf patterns...</div></div>',
                              unsafe_allow_html=True)

        probs = predict(image, model)
        time.sleep(1.2)  # short pause so the scanning animation is visible
        img_slot.markdown(image_card(b64, scanning=False), unsafe_allow_html=True)
        out_slot.markdown(result_html(probs, class_names), unsafe_allow_html=True)

        st.caption("A decision aid, not a replacement for an agricultural expert. "
                   "Trained on single-leaf photos with plain backgrounds, so accuracy may drop on field photos.")

with tab2:
    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in [(c1, "38", "Classes"), (c2, "14", "Crops"),
                            (c3, "54,305", "Training images"), (c4, "97.77%", "Test accuracy")]:
        col.markdown(f'<div class="card stat"><b>{num}</b><br>{label}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card d2"><h4>How it works</h4>'
        '1. The photo is resized to 224 x 224 pixels.<br>'
        '2. A MobileNetV2 network (transfer learning from ImageNet, fine-tuned on PlantVillage) extracts features.<br>'
        '3. A softmax layer gives a probability for each of the 38 classes.<br>'
        '4. The app shows the top prediction, confidence, and a suggested action.</div>'
        '<div class="card d3"><h4>Model comparison</h4>'
        'Custom CNN: <b>93.38%</b> &nbsp;|&nbsp; MobileNetV2: <b>97.77%</b> test accuracy.</div>',
        unsafe_allow_html=True)
