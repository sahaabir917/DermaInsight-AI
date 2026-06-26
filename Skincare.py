import streamlit as st
from datasets import load_dataset
import os
from PIL import Image
import warnings
from matplotlib import pyplot as plt
import base64
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from chromadb.utils.data_loaders import ImageLoader
import tempfile
import io

# Suppress warnings
warnings.filterwarnings("ignore")
load_dotenv()

def load_hosted_secrets():
    for key in ("OPENAI_API_KEY", "HF_TOKEN"):
        if os.getenv(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = str(value)

load_hosted_secrets()

# === Page Config ===
st.set_page_config(
    page_title="DermaInsight AI | Skin Lesion Support",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# === Global CSS ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ─────────────────────────────────────────────
   SHIMMER ANIMATION
───────────────────────────────────────────── */
@keyframes shimmer {
    0%   { background-position: -800px 0; }
    100% { background-position:  800px 0; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(99,102,241,0.35); }
    70%  { box-shadow: 0 0 0 10px rgba(99,102,241,0); }
    100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.shimmer-box {
    background: linear-gradient(90deg, #f0f4f8 25%, #e2e8f0 50%, #f0f4f8 75%);
    background-size: 800px 100%;
    animation: shimmer 1.4s infinite linear;
    border-radius: 12px;
    height: 220px;
}

/* ─────────────────────────────────────────────
   RESET & BASE
───────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
footer, #MainMenu { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ─────────────────────────────────────────────
   HERO
───────────────────────────────────────────── */
.hero {
    background: linear-gradient(-45deg, #0d1117, #0f2942, #1a1a3e, #0d2137);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    padding: 3.5rem 4rem 3rem;
    position: relative;
    overflow: hidden;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.hero::before {
    content: "";
    position: absolute;
    top: -100px; right: -80px;
    width: 420px; height: 420px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
}
.hero::after {
    content: "";
    position: absolute;
    bottom: -120px; left: 20%;
    width: 500px; height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(6,182,212,0.12) 0%, transparent 70%);
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    border: 1px solid rgba(165,180,252,0.25);
    margin-bottom: 1.2rem;
}
.hero-title {
    color: #f1f5f9;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 0.7rem;
    line-height: 1.1;
}
.hero-title .accent { color: #818cf8; }
.hero-title .accent2 { color: #22d3ee; }
.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    line-height: 1.7;
    margin: 0 0 2rem;
    max-width: 560px;
}
.hero-stats { display: flex; gap: 1rem; flex-wrap: wrap; }
.stat-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    padding: 0.5rem 1.1rem;
    color: #cbd5e1;
    font-size: 0.8rem;
    font-weight: 500;
    backdrop-filter: blur(4px);
}
.stat-pill strong { color: #818cf8; }

/* ─────────────────────────────────────────────
   SECTION WRAPPER
───────────────────────────────────────────── */
.section-wrap {
    padding: 2rem 3rem 2rem;
    max-width: 1280px;
    margin: 0 auto;
    animation: fadeInUp 0.4s ease both;
}

/* ─────────────────────────────────────────────
   INPUT FORM CARD
───────────────────────────────────────────── */
.form-card {
    background: #161b27;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 2rem 2.2rem 1.6rem;
    box-shadow: 0 4px 30px rgba(0,0,0,0.35);
    margin-bottom: 1.5rem;
}
.step-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.step-1 { color: #818cf8; }
.step-2 { color: #22d3ee; }
.field-title {
    font-size: 1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.2rem;
}
.field-sub {
    font-size: 0.8rem;
    color: #64748b;
    margin-bottom: 0.8rem;
}

/* ─────────────────────────────────────────────
   INPUT OVERRIDES (dark theme)
───────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: #0d1117 !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-size: 0.92rem !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.18) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #475569 !important; }
[data-testid="stTextInput"] label { color: #64748b !important; font-size: 0.01rem !important; }

[data-testid="stFileUploader"] > div {
    background: #0d1117 !important;
    border: 2px dashed rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: #22d3ee !important;
    background: rgba(34,211,238,0.04) !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small { color: #64748b !important; }
[data-testid="stFileUploader"] label { color: #64748b !important; font-size: 0.01rem !important; }

/* ─────────────────────────────────────────────
   SUBMIT BUTTON
───────────────────────────────────────────── */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #4338ca 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    transition: all 0.25s ease !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
    box-shadow: 0 6px 28px rgba(99,102,241,0.55) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ─────────────────────────────────────────────
   SECTION TITLES
───────────────────────────────────────────── */
.section-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* ─────────────────────────────────────────────
   MATCH CARDS
───────────────────────────────────────────── */
.match-card {
    background: #161b27;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    animation: fadeInUp 0.5s ease both;
}
.match-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.45);
}
.match-card-header {
    padding: 0.9rem 1.2rem 0.7rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.match-num-1 { background: linear-gradient(135deg, #6366f1, #818cf8); }
.match-num-2 { background: linear-gradient(135deg, #06b6d4, #22d3ee); }
.match-num {
    font-size: 0.7rem;
    font-weight: 700;
    color: #fff;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
}
.match-dx {
    font-size: 0.82rem;
    font-weight: 600;
    color: #e2e8f0;
}
.match-card-body { padding: 0.9rem 1.2rem 1.1rem; }
.meta-row { display: flex; gap: 0.45rem; flex-wrap: wrap; margin-top: 0.6rem; }
.meta-chip {
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.22rem 0.65rem;
    border-radius: 6px;
}
.chip-loc  { background: rgba(99,102,241,0.15);  color: #a5b4fc; border: 1px solid rgba(99,102,241,0.2);  }
.chip-age  { background: rgba(34,211,238,0.12);  color: #67e8f9; border: 1px solid rgba(34,211,238,0.2);  }
.chip-sex  { background: rgba(244,114,182,0.12); color: #f9a8d4; border: 1px solid rgba(244,114,182,0.2); }
.chip-type { background: rgba(251,191,36,0.12);  color: #fde68a; border: 1px solid rgba(251,191,36,0.2);  }

/* ─────────────────────────────────────────────
   RETRIEVED CHUNK CARDS
───────────────────────────────────────────── */
.chunk-card {
    background: #0f172a;
    border: 1.5px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
    margin-bottom: 0.5rem;
}
.chunk-card:hover { border-color: #334155; box-shadow: 0 4px 18px rgba(0,0,0,0.35); }
.chunk-header {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.5rem 0.75rem;
    background: #0d1117;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.chunk-badge {
    font-size: 0.6rem; font-weight: 700; padding: 2px 7px;
    border-radius: 999px; letter-spacing: 0.04em; white-space: nowrap;
}
.cb-1 { background: #312e81; color: #a5b4fc; }
.cb-2 { background: #164e63; color: #67e8f9; }
.cb-3 { background: #14532d; color: #86efac; }
.cb-4 { background: #7c2d12; color: #fdba74; }
.cb-5 { background: #4a1d96; color: #d8b4fe; }
.chunk-dx {
    font-size: 0.65rem; font-weight: 600; color: #e2e8f0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.chunk-sim { font-size: 0.58rem; color: #475569; margin-left: auto; white-space: nowrap; }
.chunk-meta {
    padding: 0.45rem 0.75rem 0.6rem;
    font-size: 0.6rem; color: #64748b; line-height: 1.65;
}

/* ─────────────────────────────────────────────
   PREVIEW CARD
───────────────────────────────────────────── */
.preview-card {
    background: #161b27;
    border: 1px solid rgba(34,211,238,0.2);
    border-radius: 18px;
    padding: 1.2rem;
    box-shadow: 0 0 20px rgba(34,211,238,0.08);
    animation: fadeInUp 0.4s ease both;
}
.preview-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #22d3ee;
    margin-bottom: 0.8rem;
}

/* ─────────────────────────────────────────────
   INSIGHTS CARD (AI result)
───────────────────────────────────────────── */
.insights-wrap {
    background: #161b27;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.07);
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
    animation: fadeInUp 0.5s ease both;
    margin-top: 0.5rem;
}
.insights-topbar {
    background: linear-gradient(135deg, #4f46e5, #6366f1, #06b6d4);
    padding: 1rem 1.8rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.insights-body {
    padding: 2rem 2.2rem;
    color: #cbd5e1;
    line-height: 1.8;
    font-size: 0.95rem;
}
.insights-body h2 {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-top: 1.8rem;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.insights-body h2:first-child { margin-top: 0; }
.insights-body strong { color: #e2e8f0; }
.insights-body ul, .insights-body ol { padding-left: 1.3rem; }
.insights-body li { margin-bottom: 0.4rem; }
.insights-body hr { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 1.2rem 0; }
.insights-body p { margin-bottom: 0.7rem; }

/* ─────────────────────────────────────────────
   DISCLAIMER BAR
───────────────────────────────────────────── */
.warn-bar {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.2);
    border-left: 4px solid #f59e0b;
    border-radius: 12px;
    padding: 0.9rem 1.4rem;
    color: #fde68a;
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 1.5rem;
    line-height: 1.6;
}

/* ─────────────────────────────────────────────
   DIVIDER
───────────────────────────────────────────── */
.styled-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 2rem 0;
}

/* ─────────────────────────────────────────────
   SPINNER
───────────────────────────────────────────── */
[data-testid="stSpinner"] > div {
    border-top-color: #818cf8 !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)

# === Load up to 100 samples per dx category from train split ===
IMAGE_DIR = "./data/skin_images"
SAMPLES_PER_CATEGORY = 100

@st.cache_data
def load_and_prepare_dataset():
    """Stream the dataset directly from the Hub (no download_and_prepare step),
    save each image to disk, and return only serializable data."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    ds = load_dataset(
        "marmal88/skin_cancer",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    counts: dict = {}
    uris, metadatas, ids = [], [], []
    for i, sample in enumerate(ds):
        cat = sample.get("dx", "unknown")
        if counts.get(cat, 0) >= SAMPLES_PER_CATEGORY:
            continue
        image_id = sample.get("image_id", f"img_{i}")
        img_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            sample["image"].save(img_path)
        uris.append(img_path)
        ids.append(image_id)
        metadatas.append({
            "image_id":     image_id,
            "lesion_id":    sample.get("lesion_id", ""),
            "dx":           cat,
            "dx_type":      sample.get("dx_type", ""),
            "age":          str(sample.get("age", "")),
            "sex":          sample.get("sex", ""),
            "localization": sample.get("localization", ""),
        })
        counts[cat] = counts.get(cat, 0) + 1
    return uris, metadatas, ids

@st.cache_resource(show_spinner="Initialising AI models and database — one moment…")
def init_db():
    _client = chromadb.PersistentClient(path=os.path.abspath("./data/skin.db"))
    _loader = ImageLoader()
    _emb_fn = OpenCLIPEmbeddingFunction()
    _col = _client.get_or_create_collection(
        "skin_collection",
        embedding_function=_emb_fn,
        data_loader=_loader,
    )
    if _col.count() == 0:
        # DB is empty — stream, save images, and embed in batches to avoid OOM.
        _uris, _metas, _ids = load_and_prepare_dataset()
        batch_size = 50
        for start in range(0, len(_ids), batch_size):
            end = start + batch_size
            _col.add(
                uris=_uris[start:end],
                metadatas=_metas[start:end],
                ids=_ids[start:end],
            )
    # If count > 0 the .db file already has embeddings — skip entirely.
    existing = _col.get(include=["metadatas", "uris"])
    return _col, _emb_fn, existing["ids"], existing["metadatas"], existing["uris"]

# init_db() is called later in the app flow after page config so the HTTP server
# starts up before the heavy CLIP model / dataset loading begins.

# === Helper: display image from URI ===
def show_image_from_uri(uri, caption="", width=250):
    img = Image.open(uri)
    st.image(img, caption=caption, width=width)

# === Helper: encode image to base64 (from file path or PIL) ===
def encode_image_to_base64(source):
    if isinstance(source, str):  # file path
        with open(source, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(source, Image.Image):  # PIL image
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_image_data_url(source, mime_type="image/jpeg"):
    """Return a data URL string ready to pass to GPT-4 Vision."""
    b64 = encode_image_to_base64(source)
    return f"data:{mime_type};base64,{b64}"

# === Helper: format prompt inputs ===
# def format_prompt_inputs(data, user_query, uploaded_image_b64=None):
#     inputs = {}
#     inputs["user_query"] = user_query

#     # Retrieved images from ChromaDB
#     image_path_1 = data["uris"][0][0]
#     image_path_2 = data["uris"][0][1]
#     inputs["image_data_1"] = encode_image_to_base64(image_path_1)
#     inputs["image_data_2"] = encode_image_to_base64(image_path_2)

#     # Metadata context string
#     meta_1 = data.get("metadatas", [[{}, {}]])[0][0]
#     meta_2 = data.get("metadatas", [[{}, {}]])[0][1]
#     inputs["metadata_context"] = (
#         f"Image 1 — Diagnosis: {meta_1.get('dx','N/A')}, "
#         f"Type: {meta_1.get('dx_type','N/A')}, "
#         f"Age: {meta_1.get('age','N/A')}, Sex: {meta_1.get('sex','N/A')}, "
#         f"Localization: {meta_1.get('localization','N/A')}. "
#         f"Image 2 — Diagnosis: {meta_2.get('dx','N/A')}, "
#         f"Type: {meta_2.get('dx_type','N/A')}, "
#         f"Age: {meta_2.get('age','N/A')}, Sex: {meta_2.get('sex','N/A')}, "
#         f"Localization: {meta_2.get('localization','N/A')}."
#     )

#     # Optional uploaded image
#     inputs["uploaded_image_data"] = uploaded_image_b64 or ""
#     return inputs

def build_multi_case_message(results, user_query, uploaded_image_b64=None, uploaded_mime="image/jpeg"):
    """Build LLM message content with all N retrieved reference cases + optional user image."""
    uris  = results["uris"][0]
    metas = results["metadatas"][0]
    n     = len(uris)

    meta_lines = []
    for i, meta in enumerate(metas, 1):
        meta_lines.append(
            f"Case {i}: Diagnosis={meta.get('dx','N/A')}, Type={meta.get('dx_type','N/A')}, "
            f"Age={meta.get('age','N/A')}, Sex={meta.get('sex','N/A')}, "
            f"Localization={meta.get('localization','N/A')}"
        )
    meta_context = "\n".join(meta_lines)

    if uploaded_image_b64:
        intro = (
            f"User description (if any): {user_query if user_query.strip() else 'None provided.'}\n\n"
            f"You are given {n} reference cases from the knowledge base. Images follow in order: "
            f"Case 1, Case 2, … Case {n}. The FINAL image is the USER IMAGE — the primary subject.\n\n"
            f"Reference case metadata:\n{meta_context}\n\n"
            "Select the best-matching reference case and provide your full structured assessment."
        )
    else:
        intro = (
            f"User description: {user_query}\n\n"
            f"You are given {n} reference cases from the knowledge base. Images follow in order: "
            f"Case 1, Case 2, … Case {n}.\n\n"
            f"Reference case metadata:\n{meta_context}\n\n"
            "Select the best-matching reference case and provide your full structured assessment."
        )

    content = [{"type": "text", "text": intro}]
    for uri in uris:
        b64 = encode_image_to_base64(uri)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    if uploaded_image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:{uploaded_mime};base64,{uploaded_image_b64}"}})
    return content


def extract_chosen_case(response, results):
    """Parse 'Case N' from the LLM's Selected Reference Case section and return (uri, meta)."""
    import re
    uris  = results["uris"][0]
    metas = results["metadatas"][0]
    m = re.search(r'Selected Reference Case[^\n]*\n\*\*Case\s+(\d+)', response, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(uris):
            return uris[idx], metas[idx]
    return uris[0], metas[0]

# === Query ChromaDB ===
def query_db(query_text=None, query_image=None, results=5):
    import numpy as np
    import torch

    if query_image is not None and query_text:
        # Both provided — combine image + text embeddings in OpenCLIP's shared vector space
        model      = embedding_function._model
        preprocess = embedding_function._preprocess
        tokenizer  = embedding_function._tokenizer

        img_tensor = preprocess(query_image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            img_emb  = model.encode_image(img_tensor).float().numpy()[0]
            txt_emb  = model.encode_text(tokenizer([query_text])).float().numpy()[0]

        combined = (img_emb + txt_emb) / 2
        combined = combined / np.linalg.norm(combined)          # normalise

        res = skin_collection.query(
            query_embeddings=[combined.tolist()],
            n_results=results,
            include=["uris", "distances", "metadatas"],
        )

    elif query_image is not None:
        # Image only — let ChromaDB embed via OpenCLIP
        res = skin_collection.query(
            query_images=[np.array(query_image.convert("RGB"))],
            n_results=results,
            include=["uris", "distances", "metadatas"],
        )

    else:
        # Text only
        res = skin_collection.query(
            query_texts=[query_text],
            n_results=results,
            include=["uris", "distances", "metadatas"],
        )

    return res

def classify_top_match(results):
    """
    From k=5 retrieved results, pick the most common dx (majority vote).
    Returns (winning_dx, best_uri, best_meta) — the URI/meta of the
    highest-ranked image that belongs to the winning class.
    """
    from collections import Counter
    uris      = results["uris"][0]       # list of 5 paths
    metas     = results["metadatas"][0]  # list of 5 dicts

    dx_labels = [m.get("dx", "unknown") for m in metas]
    winning_dx = Counter(dx_labels).most_common(1)[0][0]

    # Pick the first (closest) image that belongs to the winning class
    for uri, meta in zip(uris, metas):
        if meta.get("dx") == winning_dx:
            return winning_dx, uri, meta

    # Fallback: just use the top result
    return dx_labels[0], uris[0], metas[0]

# === Vision Model ===
@st.cache_resource
def get_vision_model():
    return ChatOpenAI(model="gpt-4o", temperature=0.0)

parser = StrOutputParser()

SYSTEM_PROMPT = """You are a dermatology education assistant. Your job is to visually analyze \
skin lesion images and explain what you see — based solely on what is visible in the images \
and the matched cases from the knowledge base. Never invent symptoms the user did not report.

You will be given:
- Several reference cases retrieved from a medical knowledge base, labeled Case 1, Case 2, etc. \
  Each has an image and clinical metadata.
- Optionally, a USER IMAGE as the final image — this is the PRIMARY subject to analyse.

Your first task is to SELECT the single reference case that best matches the condition visible \
in the user image (or the description if no image is uploaded). Then provide the full assessment.

Respond in this EXACT structure:

---

## 🎯 Selected Reference Case
**Case [N] — [Diagnosis in plain English]**
[1 sentence: the key visual or clinical reason this case was chosen over the others.]

---

## 🖼️ Image Match
Your image visually resembles cases of **[condition name in plain English]** found in our knowledge base.

**Why it matches:**
- [Visual feature 1 that aligns — e.g. color, border, texture, shape]
- [Visual feature 2]
- [Visual feature 3]

---

## 🔬 About This Condition
[2–3 sentences: what this condition is, who typically gets it, and what it generally looks like. \
Factual, plain English, no alarm.]

---

## ⚠️ When Does It Become Dangerous?
[1–2 sentences explaining what makes this condition progress to something serious, \
e.g. malignant transformation, infection risk, spread.]

---

## 📋 Watch For These Warning Signs
Based specifically on the condition you identified above, generate exactly 5 condition-specific \
warning signs the user should watch for. Each must be:
- A concrete, observable change relevant to THIS condition (not generic advice)
- Written as: "If [specific observable sign], then [action]"
- Grounded in clinical knowledge of the identified condition

1. If [condition-specific sign], [action].
2. ...
3. ...
4. ...
5. ...

---

## 🩺 Overall Assessment
**Severity: [Minor / Moderate / Major]** — based on what is visible in the image.

[1–2 warm, direct sentences about the finding and recommended next step.]

---

## ⚠️ Disclaimer
This is an educational tool only and does not constitute medical advice. \
Please consult a licensed dermatologist for an accurate diagnosis.

---

TONE: Calm, factual, and reassuring. Only describe what is visible — never assume \
symptoms the user did not mention. Be direct about severity without causing panic."""

# Prompt for text-only path (no uploaded image)
# image_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", SYSTEM_PROMPT),
#         (
#             "user",
#             [
#                 {
#                     "type": "text",
#                     "text": (
#                         "Additional context from the user (if any): {user_query}\n\n"
#                         "Matched cases from knowledge base — clinical metadata:\n{metadata_context}\n\n"
#                         "The two images are the closest matching cases retrieved from the database. "
#                         "Analyse what you see visually in these reference images and provide "
#                         "your full structured assessment now."
#                         "{uploaded_note}"
#                     ),
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": "data:image/jpeg;base64,{image_data_1}",
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": "data:image/jpeg;base64,{image_data_2}",
#                 },
#             ],
#         ),
#     ]
# )

image_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "user",
            [
                {
                    "type": "text",
                    "text": (
                        "Additional context from the user (if any): {user_query}\n\n"
                        "Classified case from knowledge base — clinical metadata:\n{metadata_context}\n\n"
                        "The image below is the closest matching reference case. "
                        "Analyse it and provide your full structured assessment."
                        "{uploaded_note}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_1}",
                },
            ],
        ),
    ]
)

# Run once per server process — HTTP server is already up by this point so
# Render's health check passes before the heavy CLIP/dataset init begins.
skin_collection, embedding_function, ids, metadatas, uris = init_db()

# ═══════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <div class="hero-badge">✦ AI-Powered Dermatology Assistant</div>
    <h1 class="hero-title">Derma<span class="accent">Insight</span> <span class="accent2">AI</span></h1>
    <p class="hero-subtitle">
        Upload a skin lesion image or describe your concern. Our AI instantly retrieves
        visually similar clinical cases and generates a structured educational assessment.
    </p>
    <div class="hero-stats">
        <div class="stat-pill">🔬 <strong>{len(ids)}</strong>&nbsp;Clinical Cases</div>
        <div class="stat-pill">🧠 <strong>GPT-4o</strong>&nbsp;Vision</div>
        <div class="stat-pill">📊 <strong>HAM10000</strong>&nbsp;Dataset</div>
        <div class="stat-pill">⚡ <strong>OpenCLIP</strong>&nbsp;Embeddings</div>
        <div class="stat-pill">🎯 <strong>7</strong>&nbsp;Condition Categories</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# INPUT FORM
# ═══════════════════════════════════════════════
st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
st.markdown('<div class="form-card">', unsafe_allow_html=True)

with st.form("query_form"):
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("""
        <div class="step-label step-1">① Step One — Describe</div>
        <div class="field-title">What do you notice?</div>
        <div class="field-sub">Location, duration, colour change, size, pain — any detail helps</div>
        """, unsafe_allow_html=True)
        query = st.text_input(
            label="desc",
            label_visibility="collapsed",
            placeholder="e.g. dark asymmetric mole on forearm, growing for 3 weeks…",
        )

    with col_right:
        st.markdown("""
        <div class="step-label step-2">② Step Two — Upload</div>
        <div class="field-title">Photo of the lesion</div>
        <div class="field-sub">JPG or PNG · well-lit, in-focus photo gives best results</div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            label="img",
            label_visibility="collapsed",
            type=["jpg", "jpeg", "png"],
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔬  Run Analysis", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)   # form-card
st.markdown('</div>', unsafe_allow_html=True)   # section-wrap

# ═══════════════════════════════════════════════
# UPLOADED IMAGE PREVIEW
# ═══════════════════════════════════════════════
uploaded_image_b64 = None
uploaded_mime      = "image/jpeg"
uploaded_pil       = None

if uploaded_file:
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    uploaded_mime = "image/png" if ext == "png" else "image/jpeg"
    uploaded_pil  = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
    prev_col, _ = st.columns([1, 2])
    with prev_col:
        st.markdown('<div class="preview-card">', unsafe_allow_html=True)
        st.markdown('<div class="preview-label">📷 Your uploaded image</div>', unsafe_allow_html=True)
        st.image(uploaded_pil, width=400)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════
if submitted:
    if not query and not uploaded_file:
        st.markdown('<div class="section-wrap">', unsafe_allow_html=True)
        st.warning("Please enter a description or upload an image before running the analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    if uploaded_pil is not None:
        uploaded_image_b64 = encode_image_to_base64(uploaded_pil)

    search_query = query.strip() if query.strip() else "skin lesion condition"

    st.markdown('<div class="section-wrap">', unsafe_allow_html=True)

    # ── Retrieval shimmers (5 columns) ──────────
    st.markdown('<div class="section-title">🔎 Retrieving similar cases from knowledge base</div>', unsafe_allow_html=True)
    shimmer_cols = st.columns(5, gap="small")
    shimmer_phs  = [col.markdown('<div class="shimmer-box" style="height:220px"></div>', unsafe_allow_html=True)
                    for col in shimmer_cols]

    if uploaded_pil is not None:
        results = query_db(query_image=uploaded_pil, query_text=query.strip() or None)
    else:
        results = query_db(query_text=search_query)

    # ── Replace shimmers with retrieved chunk cards ──
    for ph in shimmer_phs:
        ph.empty()

    uris_r  = results["uris"][0]
    metas_r = results["metadatas"][0]
    dists_r = results["distances"][0]
    cb_cls  = ["cb-1", "cb-2", "cb-3", "cb-4", "cb-5"]

    chunk_cols = st.columns(len(uris_r), gap="small")
    for idx, (col, uri, meta, dist) in enumerate(zip(chunk_cols, uris_r, metas_r, dists_r)):
        with col:
            dx_lbl  = meta.get("dx", "unknown").replace("_", " ").title()
            loc_lbl = meta.get("localization", "—").replace("_", " ").title()
            age_lbl = meta.get("age", "—")
            sex_lbl = meta.get("sex", "—").title()
            sim_pct = max(0.0, 1.0 - dist) * 100
            st.markdown(f"""
            <div class="chunk-card">
                <div class="chunk-header">
                    <span class="chunk-badge {cb_cls[idx]}">Case {idx+1}</span>
                    <span class="chunk-dx">{dx_lbl}</span>
                    <span class="chunk-sim">{sim_pct:.0f}%</span>
                </div>
                <div class="chunk-meta">
                    📍 {loc_lbl}<br>🧑 Age {age_lbl} &nbsp;⚧ {sex_lbl}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(Image.open(uri), use_container_width=True)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # ── AI Analysis ─────────────────────────────
    st.markdown('<div class="section-title">🤖 AI Clinical Assessment</div>', unsafe_allow_html=True)
    ai_placeholder = st.markdown('<div class="shimmer-box" style="height:320px"></div>', unsafe_allow_html=True)

    if not os.getenv("OPENAI_API_KEY"):
        ai_placeholder.empty()
        st.error("OPENAI_API_KEY is not configured. Add it as a hosted secret before running analysis.")
        st.stop()

    with st.spinner(""):
        vision_model = get_vision_model()
        user_content = build_multi_case_message(results, search_query, uploaded_image_b64, uploaded_mime)
        raw          = vision_model.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)])
        response     = parser.invoke(raw)

    ai_placeholder.empty()
    st.markdown(f"""
    <div class="insights-wrap">
        <div class="insights-topbar">💡 Clinical Assessment — DermaInsight AI</div>
        <div class="insights-body">{response}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Final Classification (LLM-chosen case) ───
    chosen_uri, chosen_meta = extract_chosen_case(response, results)

    st.markdown('<div class="section-title">🧬 Final Classification</div>', unsafe_allow_html=True)
    card_col, _ = st.columns([1, 1], gap="large")

    dx      = chosen_meta.get("dx", "unknown").replace("_", " ").title()
    loc     = chosen_meta.get("localization", "—").replace("_", " ").title()
    age     = chosen_meta.get("age", "—")
    sex     = chosen_meta.get("sex", "—").title()
    dx_type = chosen_meta.get("dx_type", "—")

    with card_col:
        st.markdown(f"""
        <div class="match-card">
            <div class="match-card-header">
                <span class="match-num match-num-1">AI Selected</span>
                <span class="match-dx">{dx}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.image(Image.open(chosen_uri), width=400)
        st.markdown(f"""
        <div class="match-card-body">
            <div class="meta-row">
                <span class="meta-chip chip-loc">📍 {loc}</span>
                <span class="meta-chip chip-age">🧑 Age {age}</span>
                <span class="meta-chip chip-sex">⚧ {sex}</span>
                <span class="meta-chip chip-type">🔬 {dx_type}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-bar">
        ⚠️ <strong>Educational use only.</strong>&nbsp; This tool does not provide medical diagnosis.
        Always consult a licensed dermatologist for professional evaluation and treatment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
