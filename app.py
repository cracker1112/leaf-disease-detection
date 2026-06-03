"""
╔══════════════════════════════════════════════════════════════╗
║   LEAF DISEASE DETECTION — Production Streamlit App          ║
║   Developer  : Sumit Jadhav                                  ║
║   Course     : M.Sc. Computer Science                        ║
║   College    : Annasaheb Magar Mahavidyalaya, Pune           ║
║   Internship : CodeAlpha                                     ║
║   Guide      : Prof. Gadekar M.J.                            ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import streamlit as st
st.set_page_config(
    page_title="LeafAI — Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

import numpy as np
from PIL import Image
import time

# ── MODEL LOADER (FIXED FOR KERAS VERSION MISMATCH) ───────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

        tf.get_logger().setLevel('ERROR')

        base_model = MobileNetV2(
            input_shape=(256, 256, 3),
            include_top=False,
            weights=None
        )
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(15, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output)

        # Weights load karo by name
        model.load_weights(path, by_name=True, skip_mismatch=True)

        # Warmup
        dummy = np.random.rand(1, 256, 256, 3).astype(np.float32)
        _ = model.predict(dummy, verbose=0)

        return model, None
    except Exception as e:
        return None, str(e)
# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MODEL_PATH = "mobilenet_plantdisease.h5"
IMG_SIZE   = (256, 256)

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# ── DISEASE DATABASE ───────────────────────────────────────────────────────────
DISEASE_DB = {
    "Pepper__bell___Bacterial_spot": {
        "display":    "Pepper Bell — Bacterial Spot",
        "status":     "DISEASED",
        "severity":   "Moderate",
        "pathogen":   "Xanthomonas campestris pv. vesicatoria",
        "plant":      "Pepper Bell",
        "emoji":      "🫑",
        "color":      "#f97316",
        "symptoms":   "Small water-soaked spots on leaves that enlarge and turn dark brown with yellow halos.",
        "causes":     "Bacterial infection spread through contaminated seeds, rain splash, and overhead irrigation.",
        "prevention": ["Use certified disease-free seeds","Apply copper-based bactericides","Avoid overhead irrigation","Practice 2–3 year crop rotation","Remove infected plant debris"],
        "treatment":  "Copper hydroxide sprays (0.3%) every 7–10 days during wet weather.",
        "yield_loss": "10–50%",
    },
    "Pepper__bell___healthy": {
        "display":    "Pepper Bell — Healthy",
        "status":     "HEALTHY",
        "severity":   "None",
        "pathogen":   "None",
        "plant":      "Pepper Bell",
        "emoji":      "🫑",
        "color":      "#22c55e",
        "symptoms":   "No disease symptoms. Normal green coloration with no spots or discoloration.",
        "causes":     "N/A",
        "prevention": ["Continue regular monitoring","Maintain proper watering","Ensure good air circulation","Keep field clean"],
        "treatment":  "No treatment required.",
        "yield_loss": "0%",
    },
    "Potato___Early_blight": {
        "display":    "Potato — Early Blight",
        "status":     "DISEASED",
        "severity":   "Moderate",
        "pathogen":   "Alternaria solani",
        "plant":      "Potato",
        "emoji":      "🥔",
        "color":      "#f97316",
        "symptoms":   "Dark brown circular lesions with concentric rings (target-board pattern) on older leaves.",
        "causes":     "Fungal infection favored by warm (24–29°C) humid conditions.",
        "prevention": ["Apply mancozeb every 7–10 days","Use resistant varieties","Avoid overhead irrigation","Practice 3-year crop rotation"],
        "treatment":  "Mancozeb 75 WP at 2g/litre. Start sprays at first sign.",
        "yield_loss": "20–40%",
    },
    "Potato___Late_blight": {
        "display":    "Potato — Late Blight",
        "status":     "DISEASED",
        "severity":   "Severe",
        "pathogen":   "Phytophthora infestans",
        "plant":      "Potato",
        "emoji":      "🥔",
        "color":      "#ef4444",
        "symptoms":   "Water-soaked brown lesions on leaf margins. White cottony growth on undersides in humid conditions.",
        "causes":     "Oomycete pathogen in cool (10–25°C) wet conditions. Caused the Irish Potato Famine of 1845.",
        "prevention": ["Apply metalaxyl + mancozeb preventively","Plant certified disease-free seed tubers","Avoid planting near tomatoes","Harvest during dry weather"],
        "treatment":  "Metalaxyl 8% + Mancozeb 64% at 2.5g/litre every 5–7 days.",
        "yield_loss": "40–100%",
    },
    "Potato___healthy": {
        "display":    "Potato — Healthy",
        "status":     "HEALTHY",
        "severity":   "None",
        "pathogen":   "None",
        "plant":      "Potato",
        "emoji":      "🥔",
        "color":      "#22c55e",
        "symptoms":   "No disease symptoms. Leaves appear dark green and healthy.",
        "causes":     "N/A",
        "prevention": ["Scout plants every 3–5 days","Maintain balanced fertilization","Monitor weather for blight risk"],
        "treatment":  "No treatment required.",
        "yield_loss": "0%",
    },
    "Tomato_Bacterial_spot": {
        "display":    "Tomato — Bacterial Spot",
        "status":     "DISEASED",
        "severity":   "Moderate",
        "pathogen":   "Xanthomonas campestris pv. vesicatoria",
        "plant":      "Tomato",
        "emoji":      "🍅",
        "color":      "#f97316",
        "symptoms":   "Small dark water-soaked leaf spots with yellow halos causing defoliation.",
        "causes":     "Bacterial spread through rain, irrigation, contaminated tools, and infected seeds.",
        "prevention": ["Use hot water-treated seeds","Apply copper bactericides","Avoid overhead irrigation","Disinfect tools with bleach"],
        "treatment":  "Copper hydroxide 53.8% WG at 1.5g/litre every 7 days.",
        "yield_loss": "15–35%",
    },
    "Tomato_Early_blight": {
        "display":    "Tomato — Early Blight",
        "status":     "DISEASED",
        "severity":   "Moderate",
        "pathogen":   "Alternaria solani",
        "plant":      "Tomato",
        "emoji":      "🍅",
        "color":      "#f97316",
        "symptoms":   "Dark brown lesions with concentric rings on older leaves. Yellowing and leaf drop.",
        "causes":     "Fungal infection favored by warm temperatures (24–29°C) and high humidity.",
        "prevention": ["Apply chlorothalonil at first symptoms","Stake plants for air circulation","Mulch soil to prevent rain splash","Remove lower infected leaves"],
        "treatment":  "Chlorothalonil 75 WP at 2g/litre every 7–10 days.",
        "yield_loss": "20–40%",
    },
    "Tomato_Late_blight": {
        "display":    "Tomato — Late Blight",
        "status":     "DISEASED",
        "severity":   "Severe",
        "pathogen":   "Phytophthora infestans",
        "plant":      "Tomato",
        "emoji":      "🍅",
        "color":      "#ef4444",
        "symptoms":   "Large greasy-green to dark brown water-soaked lesions on leaves, stems, and fruits.",
        "causes":     "Oomycete thriving in cool wet conditions. Spreads rapidly under favorable conditions.",
        "prevention": ["Apply systemic fungicides preventively","Plant on raised beds","Remove and burn infected material","Avoid planting near potatoes"],
        "treatment":  "Metalaxyl-M + Mancozeb at 2.5g/litre before rain.",
        "yield_loss": "40–100%",
    },
    "Tomato_Leaf_Mold": {
        "display":    "Tomato — Leaf Mold",
        "status":     "DISEASED",
        "severity":   "Moderate",
        "pathogen":   "Passalora fulva",
        "plant":      "Tomato",
        "emoji":      "🍅",
        "color":      "#f97316",
        "symptoms":   "Pale yellow spots on upper leaf surface. Olive-green velvety mold on lower surface.",
        "causes":     "Fungal pathogen in warm (22–25°C) high humidity (>85%) conditions.",
        "prevention": ["Improve ventilation to reduce humidity","Apply chlorothalonil","Use resistant varieties","Avoid wetting leaves"],
        "treatment":  "Chlorothalonil 75 WP at 2g/litre every 7 days.",
        "yield_loss": "10–30%",
    },
    "Tomato_Septoria_leaf_spot": {
        "display":    "Tomato — Septoria Leaf Spot",
        "status":     "DISEASED",
        "severity":   "Moderate",
        "pathogen":   "Septoria lycopersici",
        "plant":      "Tomato",
        "emoji":      "🍅",
        "color":      "#f97316",
        "symptoms":   "Numerous small circular spots (2–4mm) with dark brown borders and light gray centers.",
        "causes":     "Fungal disease spread by rain splash. Starts on lower leaves and progresses upward.",
        "prevention": ["Apply mancozeb at first sign","Remove infected lower leaves","Mulch around plant base","Avoid overhead irrigation"],
        "treatment":  "Mancozeb 75 WP at 2g/litre every 7–10 days.",
        "yield_loss": "15–30%",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "display":    "Tomato — Spider Mites",
        "status":     "DISEASED",
        "severity":   "Moderate–Severe",
        "pathogen":   "Tetranychus urticae",
        "plant":      "Tomato",
        "emoji":      "🍅",
        "color":      "#f97316",
        "symptoms":   "Bronze or yellow stippling on leaves. Fine webbing on leaf undersides. Brown leaf drop.",
        "causes":     "Arthropod pest favored by hot dry conditions and dusty environments.",
        "prevention": ["Apply miticides at early infestation","Maintain adequate irrigation","Introduce predatory mites","Remove heavily infested leaves"],
        "treatment":  "Abamectin 1.8 EC at 0.5ml/litre on leaf undersides.",
        "yield_loss": "20–50%",
    },
    "Tomato__Target_Spot": {
        "display":    "Tomato — Target Spot",
        "status":     "DISEASED",
        "severity":   "Moderate",
        "pathogen":   "Corynespora cassiicola",
        "plant":      "Tomato",
        "emoji":      "🍅",
        "color":      "#f97316",
        "symptoms":   "Circular brown lesions with distinctive concentric rings (target-like) and yellow halos.",
        "causes":     "Fungal pathogen spread by wind and rain in warm humid tropical conditions.",
        "prevention": ["Apply chlorothalonil preventively","Maintain plant nutrition","Ensure good airflow","Practice crop rotation"],
        "treatment":  "Chlorothalonil 75 WP at 2g/litre every 7–10 days.",
        "yield_loss": "15–30%",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display":    "Tomato — Yellow Leaf Curl Virus",
        "status":     "DISEASED",
        "severity":   "Severe",
        "pathogen":   "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "plant":      "Tomato",
        "emoji":      "🍅",
        "color":      "#ef4444",
        "symptoms":   "Severe upward curling and yellowing of leaf margins. Stunted growth and poor fruit set.",
        "causes":     "Viral disease transmitted by whiteflies (Bemisia tabaci). No cure once infected.",
        "prevention": ["Use TYLCV-resistant varieties","Control whitefly with insecticides","Install yellow sticky traps","Remove and destroy infected plants"],
        "treatment":  "No cure available. Remove infected plants and focus on whitefly control.",
        "yield_loss": "50–100%",
    },
    "Tomato__Tomato_mosaic_virus": {
        "display":    "Tomato — Mosaic Virus",
        "status":     "DISEASED",
        "severity":   "Moderate–Severe",
        "pathogen":   "Tomato Mosaic Virus (ToMV)",
        "plant":      "Tomato",
        "emoji":      "🍅",
        "color":      "#ef4444",
        "symptoms":   "Mosaic pattern of light and dark green on leaves. Leaf malformation and stunted growth.",
        "causes":     "Virus transmitted through contaminated tools, contact with infected plants and seeds.",
        "prevention": ["Use certified virus-free seeds","Wash hands before handling plants","Disinfect tools with bleach","Remove and burn infected plants"],
        "treatment":  "No chemical cure. Focus on strict sanitation and removing infected plants.",
        "yield_loss": "25–50%",
    },
    "Tomato_healthy": {
        "display":    "Tomato — Healthy",
        "status":     "HEALTHY",
        "severity":   "None",
        "pathogen":   "None",
        "plant":      "Tomato",
        "emoji":      "🍅",
        "color":      "#22c55e",
        "symptoms":   "No disease symptoms. Vibrant green coloration with no spots or discoloration.",
        "causes":     "N/A",
        "prevention": ["Weekly scouting for disease signs","Balanced NPK fertilization","Consistent watering schedule","Apply preventive copper spray"],
        "treatment":  "No treatment required.",
        "yield_loss": "0%",
    },
}

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Playfair+Display:wght@700;800&display=swap');

    :root {
        --bg:        #070e0b;
        --card:      rgba(14,24,18,0.9);
        --glass:     rgba(255,255,255,0.03);
        --border:    rgba(74,222,128,0.12);
        --border-hi: rgba(74,222,128,0.35);
        --green:     #4ade80;
        --green-d:   #16a34a;
        --red:       #f87171;
        --orange:    #fb923c;
        --text:      #f0fdf4;
        --muted:     #6b7280;
        --sub:       #94a3b8;
    }

    html, body, .stApp {
        background: var(--bg) !important;
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text) !important;
    }
    .stApp {
        background-image:
            radial-gradient(ellipse 70% 40% at 15% 5%,  rgba(74,222,128,0.07) 0%, transparent 55%),
            radial-gradient(ellipse 50% 30% at 85% 90%, rgba(34,197,94,0.05)  0%, transparent 50%) !important;
        background-attachment: fixed !important;
    }
    .block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1280px !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(175deg, #091510 0%, #050c08 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* ── Cards ── */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(16px);
        margin-bottom: 1rem;
        transition: border-color .2s;
    }
    .card:hover { border-color: var(--border-hi); }

    /* ── Info chip ── */
    .info-chip {
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: .6rem .9rem;
        margin: .3rem 0;
    }
    .chip-label { font-size:.6rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }
    .chip-value { font-size:.82rem; color:var(--text); font-weight:500; }

    /* ── Status badges ── */
    .badge {
        display:inline-flex; align-items:center; gap:.4rem;
        padding:.35rem 1rem; border-radius:999px;
        font-weight:600; font-size:.75rem; letter-spacing:.04em;
    }
    .badge-healthy  { background:rgba(34,197,94,.12); border:1px solid rgba(34,197,94,.35); color:#4ade80; }
    .badge-diseased { background:rgba(251,146,60,.10); border:1px solid rgba(251,146,60,.35); color:#fb923c; }
    .badge-severe   { background:rgba(248,113,113,.10); border:1px solid rgba(248,113,113,.35); color:#f87171; }

    /* ── Confidence bar ── */
    .conf-row { margin:.45rem 0; }
    .conf-label { display:flex; justify-content:space-between; font-size:.72rem; margin-bottom:.2rem; }
    .conf-track { background:rgba(255,255,255,.06); height:5px; border-radius:999px; overflow:hidden; }
    .conf-fill  { height:100%; border-radius:999px; transition:width .6s ease; }

    /* ── Metric tiles ── */
    [data-testid="stMetric"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    [data-testid="stMetricValue"] { color: var(--green) !important; font-size:1.5rem !important; font-weight:700 !important; }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size:.75rem !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #15803d, #22c55e) !important;
        color: white !important; border:none !important;
        border-radius: 10px !important; font-weight:600 !important;
        padding: .55rem 1.8rem !important; font-size:.9rem !important;
        transition: opacity .2s !important;
    }
    .stButton > button:hover { opacity:.88 !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(74,222,128,0.04) !important;
        border: 1.5px dashed rgba(74,222,128,0.3) !important;
        border-radius: 12px !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
    }
    .streamlit-expanderContent {
        background: rgba(14,24,18,0.6) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
    }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility:hidden; }
    </style>
    """, unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.2rem 0 .8rem;">
            <div style="font-size:3rem;margin-bottom:.3rem;">🌿</div>
            <div style="font-weight:700;font-size:1.15rem;color:#4ade80;letter-spacing:.02em;">LeafAI</div>
            <div style="font-size:.7rem;color:#6b7280;margin-top:.2rem;">Plant Disease Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        chips = [
            ("Developer",   "Sumit Jadhav"),
            ("Course",      "M.Sc. Computer Science"),
            ("College",     "AMM Hadapsar, Pune"),
            ("Internship",  "CodeAlpha"),
            ("Guide",       "Prof. Gadekar M.J."),
            ("Model",       "MobileNetV2"),
            ("Accuracy",    "93.19%"),
            ("Classes",     "15 Disease Classes"),
            ("Dataset",     "PlantVillage (20K imgs)"),
        ]
        for label, val in chips:
            st.markdown(f"""
            <div class="info-chip">
                <div class="chip-label">{label}</div>
                <div class="chip-value">{val}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        <div style="font-size:.68rem;color:#374151;text-align:center;line-height:1.6;">
            Powered by TensorFlow · Keras<br>
            Transfer Learning · ImageNet
        </div>""", unsafe_allow_html=True)

# ── PREDICTION ────────────────────────────────────────────────────────────────
def preprocess(uploaded_file) -> np.ndarray:
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, *IMG_SIZE, 3)

def predict(model, arr):
    preds = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(preds))
    conf  = float(preds[idx]) * 100
    top5  = [(CLASS_NAMES[i], float(preds[i])*100) for i in np.argsort(preds)[::-1][:5]]
    return CLASS_NAMES[idx], conf, top5

# ── RESULTS ───────────────────────────────────────────────────────────────────
def render_results(cls, conf, top5, pil_img):
    db       = DISEASE_DB.get(cls, {})
    display  = db.get("display",  cls.replace("_"," "))
    status   = db.get("status",   "UNKNOWN")
    severity = db.get("severity", "—")
    pathogen = db.get("pathogen", "—")
    emoji    = db.get("emoji",    "🌿")
    color    = db.get("color",    "#4ade80")
    symptoms = db.get("symptoms", "—")
    causes   = db.get("causes",   "—")
    prev     = db.get("prevention", [])
    treat    = db.get("treatment",  "—")
    yloss    = db.get("yield_loss", "—")

    if status == "HEALTHY":
        badge_cls = "badge-healthy"
        badge_icon = "✓"
    elif severity == "Severe":
        badge_cls = "badge-severe"
        badge_icon = "⚠"
    else:
        badge_cls = "badge-diseased"
        badge_icon = "●"

    st.divider()

    # ── Result header ──
    st.markdown(f"""
    <div class="card" style="border-color:{color}33;background:linear-gradient(135deg,rgba(14,24,18,.95),rgba(7,14,11,.9));">
        <div style="display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap;">
            <div style="font-size:3.8rem;line-height:1;">{emoji}</div>
            <div style="flex:1;">
                <div style="margin-bottom:.5rem;">
                    <span class="badge {badge_cls}">{badge_icon} {status}</span>
                </div>
                <div style="font-family:'Playfair Display',serif;font-size:1.7rem;font-weight:700;
                            color:#f0fdf4;line-height:1.2;margin-bottom:.3rem;">
                    {display}
                </div>
                <div style="font-size:.75rem;color:#6b7280;">
                    Pathogen: <span style="color:#9ca3af;">{pathogen}</span>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:2.8rem;font-weight:700;color:{color};line-height:1;">
                    {conf:.1f}%
                </div>
                <div style="font-size:.7rem;color:#6b7280;">Confidence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Image + confidence bars ──
    col1, col2 = st.columns([1, 1.1])
    with col1:
        st.image(pil_img, use_container_width=True, caption="Uploaded Leaf Image")

    with col2:
        st.markdown('<div class="card"><div style="font-weight:600;margin-bottom:.8rem;">📊 Top Predictions</div>', unsafe_allow_html=True)
        for cname, cpct in top5:
            disp2 = DISEASE_DB.get(cname, {}).get("display", cname.replace("_"," "))
            c2    = color if cname == cls else "#374151"
            fw    = "600" if cname == cls else "400"
            st.markdown(f"""
            <div class="conf-row">
                <div class="conf-label">
                    <span style="color:#d1fae5;font-weight:{fw};max-width:75%;overflow:hidden;
                                 text-overflow:ellipsis;white-space:nowrap;">{disp2[:42]}</span>
                    <span style="color:{c2};font-weight:600;">{cpct:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{cpct:.1f}%;background:{c2};"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Metric row ──
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence",    f"{conf:.1f}%")
    m2.metric("Severity",       severity)
    m3.metric("Yield Loss Risk", yloss)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Details ──
    c3, c4 = st.columns(2)
    with c3:
        with st.expander("🔬 Symptoms", expanded=True):
            st.markdown(f'<p style="color:#d1fae5;font-size:.88rem;line-height:1.7;">{symptoms}</p>', unsafe_allow_html=True)
    with c4:
        with st.expander("⚗️ Causes", expanded=True):
            st.markdown(f'<p style="color:#d1fae5;font-size:.88rem;line-height:1.7;">{causes}</p>', unsafe_allow_html=True)

    with st.expander("🛡️ Prevention & Management", expanded=True):
        for p in prev:
            st.markdown(f'<div style="color:#d1fae5;font-size:.88rem;padding:.3rem 0;border-bottom:1px solid rgba(74,222,128,.07);">• {p}</div>', unsafe_allow_html=True)

    with st.expander("💊 Recommended Treatment", expanded=True):
        st.markdown(f"""
        <div style="background:rgba(74,222,128,.05);border:1px solid rgba(74,222,128,.2);
                    border-radius:10px;padding:1rem;color:#d1fae5;font-size:.88rem;line-height:1.7;">
            {treat}
        </div>""", unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    render_sidebar()

    # Model load
    with st.spinner("🔄 Loading AI Model..."):
        model, err = load_model(MODEL_PATH)

    if err or model is None:
        st.error(f"⚠️ Model load failed: {err}")
        st.info(f"Make sure `{MODEL_PATH}` is in the same folder as `app.py`")
        st.stop()

    # ── Hero ──
    st.markdown("""
    <div style="text-align:center;padding:2rem 1rem 1.5rem;">
        <div style="display:inline-flex;align-items:center;gap:.5rem;
                    background:rgba(74,222,128,.07);border:1px solid rgba(74,222,128,.18);
                    border-radius:999px;padding:.3rem .9rem;margin-bottom:1rem;font-size:.78rem;color:#4ade80;">
            ⚡ AI-Powered Agriculture · 93.19% Accuracy
        </div>
        <h1 style="font-family:'Playfair Display',serif;
                   font-size:clamp(1.8rem,4vw,3rem);
                   background:linear-gradient(135deg,#f0fdf4 30%,#4ade80 100%);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin:.3rem 0 .7rem;line-height:1.15;">
            Detect Plant Leaf Diseases<br>with Deep Learning
        </h1>
        <p style="color:#6b7280;max-width:500px;margin:0 auto;font-size:.9rem;line-height:1.6;">
            Upload a leaf image — get instant AI diagnosis powered by MobileNetV2 Transfer Learning.
            Supports 15 disease classes across Tomato, Potato &amp; Pepper plants.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Leaf Image")
    st.caption("Supported: JPG, JPEG, PNG, BMP, WEBP  ·  Best results with clear, well-lit photos")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","bmp","webp"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        with st.spinner("🔬 Analyzing leaf — running AI inference..."):
            time.sleep(0.15)
            arr              = preprocess(uploaded)
            cls, conf, top5  = predict(model, arr)
        render_results(cls, conf, top5, pil_img)

    # ── Footer ──
    st.markdown("""
    <div style="border-top:1px solid rgba(74,222,128,.1);padding:1.2rem 0 .3rem;
                text-align:center;color:#374151;font-size:.7rem;margin-top:2.5rem;">
        🌿 LeafAI &nbsp;·&nbsp; Leaf Disease Detection &nbsp;·&nbsp;
        MobileNetV2 + TensorFlow &nbsp;·&nbsp;
        Developed by <strong style="color:#4b5563;">Sumit Jadhav</strong> &nbsp;·&nbsp;
        CodeAlpha Internship 2026
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()