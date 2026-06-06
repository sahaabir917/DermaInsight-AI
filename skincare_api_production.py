"""
DermaInsight AI — Production REST API
--------------------------------------
POST /analyze  →  multipart/form-data  (image file + text query)
Returns a structured JSON response with distinct clinical sections
so the frontend can render each block as its own card.

Run:
    uvicorn skincare_api_production:app --reload --port 8000

Postman:
    POST  http://127.0.0.1:8000/analyze
    Body  → form-data
           key: query  (Text)   — optional text description
           key: image  (File)   — optional JPG/PNG upload

Note on DB compatibility:
    The old skin.db was created with ChromaDB <1.0 (Python-only backend).
    ChromaDB 1.x uses Rust bindings with a new storage format.
    This API auto-migrates metadata from the old SQLite and re-embeds images
    into a fresh skin_api.db on first launch (one-time cost, ~5-10 min).
"""

from __future__ import annotations

import base64
import io
import os
import re
import sqlite3
import warnings
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

warnings.filterwarnings("ignore")
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
IMAGE_DIR        = "./data/skin_images"
OLD_DB_SQLITE    = os.path.abspath("./data/skin.db/chroma.sqlite3")   # legacy
NEW_DB_PATH      = os.path.abspath("./data/skin_api.db")              # chromadb 1.x

# ──────────────────────────────────────────────────────────────────────────────
# One-time migration: read metadata from old SQLite → re-embed into new DB
# ──────────────────────────────────────────────────────────────────────────────
def _load_legacy_metadata() -> list[dict]:
    """Pull every row's metadata + URI from the old ChromaDB SQLite."""
    conn = sqlite3.connect(OLD_DB_SQLITE)
    rows = conn.execute("""
        SELECT e.embedding_id,
            MAX(CASE WHEN m.key = 'chroma:uri'    THEN m.string_value END) AS uri,
            MAX(CASE WHEN m.key = 'image_id'      THEN m.string_value END) AS image_id,
            MAX(CASE WHEN m.key = 'lesion_id'     THEN m.string_value END) AS lesion_id,
            MAX(CASE WHEN m.key = 'dx'            THEN m.string_value END) AS dx,
            MAX(CASE WHEN m.key = 'dx_type'       THEN m.string_value END) AS dx_type,
            MAX(CASE WHEN m.key = 'age'           THEN m.string_value END) AS age,
            MAX(CASE WHEN m.key = 'sex'           THEN m.string_value END) AS sex,
            MAX(CASE WHEN m.key = 'localization'  THEN m.string_value END) AS localization
        FROM embeddings e
        JOIN embedding_metadata m ON e.id = m.id
        GROUP BY e.embedding_id
    """).fetchall()
    conn.close()

    records = []
    for r in rows:
        emb_id, uri, image_id, lesion_id, dx, dx_type, age, sex, loc = r
        if uri and os.path.exists(uri):
            records.append({
                "id":  emb_id,
                "uri": uri,
                "metadata": {
                    "image_id":     image_id or emb_id,
                    "lesion_id":    lesion_id or "",
                    "dx":           dx or "",
                    "dx_type":      dx_type or "",
                    "age":          age or "",
                    "sex":          sex or "",
                    "localization": loc or "",
                },
            })
    return records


def _migrate_to_new_db(collection: chromadb.Collection) -> None:
    """Populate the new ChromaDB collection from old-DB metadata + disk images."""
    print("  Reading metadata from legacy SQLite …")
    records = _load_legacy_metadata()
    print(f"  Found {len(records)} records with images on disk.")

    existing_ids = set(collection.get(include=[])["ids"])
    to_add = [r for r in records if r["id"] not in existing_ids]
    print(f"  {len(to_add)} records not yet embedded — starting embedding …")

    batch_size = 50
    for i in range(0, len(to_add), batch_size):
        batch = to_add[i : i + batch_size]
        collection.add(
            uris      =[b["uri"]      for b in batch],
            metadatas =[b["metadata"] for b in batch],
            ids       =[b["id"]       for b in batch],
        )
        print(f"  Embedded {min(i + batch_size, len(to_add))} / {len(to_add)} …")

    print(f"  Migration complete — {collection.count()} embeddings in new DB.")


# ──────────────────────────────────────────────────────────────────────────────
# Global singletons (populated in lifespan)
# ──────────────────────────────────────────────────────────────────────────────
embedding_function: OpenCLIPEmbeddingFunction = None  # type: ignore[assignment]
skin_collection: chromadb.Collection = None           # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_function, skin_collection

    print("=== DermaInsight API startup ===")
    image_loader       = ImageLoader()
    embedding_function = OpenCLIPEmbeddingFunction()

    os.makedirs(NEW_DB_PATH, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=NEW_DB_PATH)
    skin_collection = chroma_client.get_or_create_collection(
        "skin_collection",
        embedding_function=embedding_function,
        data_loader=image_loader,
    )

    if skin_collection.count() == 0 and os.path.exists(OLD_DB_SQLITE):
        print("New DB is empty — migrating from legacy skin.db …")
        _migrate_to_new_db(skin_collection)
    else:
        print(f"Collection ready: {skin_collection.count()} embeddings.")

    yield  # server runs here

    print("=== DermaInsight API shutdown ===")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app  (defined after lifespan so the reference is valid)
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DermaInsight AI API",
    description="Multimodal RAG skin-lesion analysis powered by OpenCLIP + GPT-4o",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Vision model  (mirrors Skincare.py)
# ──────────────────────────────────────────────────────────────────────────────
vision_model = ChatOpenAI(model="gpt-4o", temperature=0.0)
parser = StrOutputParser()

# Same system prompt as Skincare.py — unchanged so behaviour is identical
SYSTEM_PROMPT = """You are a dermatology education assistant. Your job is to visually analyze \
skin lesion images and explain what you see — based solely on what is visible in the image \
and the matched cases from the knowledge base. Never invent symptoms the user did not report.

You will be given:
- The user's uploaded skin image (primary subject)
- 2 similar cases retrieved from a medical knowledge base, with their clinical metadata

Respond in this EXACT structure:

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

Example format (do NOT copy these — generate ones specific to your identified condition):
1. If [condition-specific sign], [action].
2. ...
3. ...
4. ...
5. ...

---

## 🩺 Overall Assessment
**Severity: [Minor / Moderate / Major]** — based on what is visible in the image.

[1–2 warm, direct sentences: e.g. "This looks like a minor concern based on the image, \
but given its location and appearance, a routine check with a GP or dermatologist is a sensible \
next step. You do not need to rush, but do not ignore it either."]

---

## ⚠️ Disclaimer
This is an educational tool only and does not constitute medical advice. \
Please consult a licensed dermatologist for an accurate diagnosis.

---

TONE: Calm, factual, and reassuring. Only describe what is visible — never assume \
symptoms the user did not mention. Be direct about severity without causing panic."""


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic response schemas
# ──────────────────────────────────────────────────────────────────────────────
class MatchedCase(BaseModel):
    match_number: int
    diagnosis: str
    dx_type: str
    age: str
    sex: str
    localization: str
    image_url: str  # e.g. "http://127.0.0.1:8001/images/ISIC_0024329.jpg"


class AnalysisResponse(BaseModel):
    # Similar images from the knowledge base (base64 data-URLs)
    image_url: List[str]

    # AI-generated clinical sections (each maps to its own card in the UI)
    image_match: str
    about_condition: str
    warning_symptoms: str       # "When Does It Become Dangerous?"
    watch_for: List[str]        # exactly 5 numbered warning signs
    overall_assessment: str
    severity: str               # "Minor" | "Moderate" | "Major"
    disclaimer: str

    # Full matched-case metadata with per-case images
    matched_cases: List[MatchedCase]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers  (identical logic to Skincare.py)
# ──────────────────────────────────────────────────────────────────────────────
def encode_image_to_base64(source) -> str:
    if isinstance(source, str):
        with open(source, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    if isinstance(source, Image.Image):
        buf = io.BytesIO()
        source.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    raise TypeError(f"Unsupported source type: {type(source)}")


def image_data_url(source, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{encode_image_to_base64(source)}"


def query_db(query_text: Optional[str] = None,
             query_image: Optional[Image.Image] = None,
             n_results: int = 2) -> dict:
    """Query ChromaDB — mirrors Skincare.py query_db() exactly."""
    if query_image is not None and query_text:
        model      = embedding_function._model
        preprocess = embedding_function._preprocess
        tokenizer  = embedding_function._tokenizer

        img_tensor = preprocess(query_image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            img_emb = model.encode_image(img_tensor).float().numpy()[0]
            txt_emb = model.encode_text(tokenizer([query_text])).float().numpy()[0]

        combined = (img_emb + txt_emb) / 2
        combined = combined / np.linalg.norm(combined)

        return skin_collection.query(
            query_embeddings=[combined.tolist()],
            n_results=n_results,
            include=["uris", "distances", "metadatas"],
        )

    if query_image is not None:
        return skin_collection.query(
            query_images=[np.array(query_image.convert("RGB"))],
            n_results=n_results,
            include=["uris", "distances", "metadatas"],
        )

    return skin_collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["uris", "distances", "metadatas"],
    )


def _extract_section(text: str, header_pattern: str) -> str:
    """Return the content between two consecutive ## headers."""
    pattern = rf"{header_pattern}\s*(.*?)(?=\n##\s|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_watch_for(section_text: str) -> List[str]:
    """Extract the five numbered warning signs from the Watch-For section."""
    lines = []
    for line in section_text.splitlines():
        m = re.match(r"^\s*\d+\.\s+(.+)", line)
        if m:
            lines.append(m.group(1).strip())
    return lines


def _extract_severity(assessment_text: str) -> str:
    """Pull out 'Minor', 'Moderate', or 'Major' from the assessment block."""
    m = re.search(r"\*{1,2}Severity:\s*(Minor|Moderate|Major)\*{1,2}", assessment_text, re.IGNORECASE)
    return m.group(1).title() if m else "Unknown"


def parse_ai_response(raw: str) -> dict:
    """
    Break the structured markdown response into discrete fields
    so the mobile/web frontend can render each as its own card.
    """
    image_match      = _extract_section(raw, r"##\s+[^\n]*Image Match")
    about_condition  = _extract_section(raw, r"##\s+[^\n]*About This Condition")
    warning_symptoms = _extract_section(raw, r"##\s+[^\n]*When Does It Become Dangerous")
    watch_raw        = _extract_section(raw, r"##\s+[^\n]*Watch For These Warning Signs")
    overall          = _extract_section(raw, r"##\s+[^\n]*Overall Assessment")
    disclaimer       = _extract_section(raw, r"##\s+[^\n]*Disclaimer")

    watch_for = _extract_watch_for(watch_raw)
    severity  = _extract_severity(overall)

    return {
        "image_match":      image_match,
        "about_condition":  about_condition,
        "warning_symptoms": warning_symptoms,
        "watch_for":        watch_for,
        "overall_assessment": overall,
        "severity":         severity,
        "disclaimer":       disclaimer,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "DermaInsight AI API",
        "version": "1.0.0",
        "endpoint": "POST /analyze",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection_count": skin_collection.count(),
    }


@app.get("/images/{filename}")
def serve_image(filename: str):
    """
    Serve a skin image by filename.
    Use the filename from matched_cases[].image_url to load it directly.

    Example:
        GET http://127.0.0.1:8001/images/ISIC_0024329.jpg
    """
    path = os.path.join(os.path.abspath(IMAGE_DIR), filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    query: Optional[str] = Form(None, description="Text description of the skin concern"),
    image: Optional[UploadFile] = File(None, description="JPG/PNG photo of the lesion"),
):
    """
    Analyse a skin lesion using multimodal RAG.

    Send at least one of `query` (text) or `image` (file).
    Returns a structured JSON response — each field maps to a separate UI card.

    **Postman setup:**
    - Method: POST
    - URL: http://127.0.0.1:8000/analyze
    - Body → form-data
        - query  (Text)  — e.g. "dark asymmetric mole on forearm"
        - image  (File)  — select a JPG/PNG
    """
    if not query and not image:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: 'query' (text) or 'image' (file).",
        )

    # ── Read uploaded image ──────────────────────────────────────────────────
    uploaded_pil: Optional[Image.Image] = None
    uploaded_mime = "image/jpeg"
    uploaded_b64: Optional[str] = None

    if image is not None:
        raw_bytes = await image.read()
        ext = (image.filename or "").rsplit(".", 1)[-1].lower()
        uploaded_mime = "image/png" if ext == "png" else "image/jpeg"
        uploaded_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        uploaded_b64 = base64.b64encode(raw_bytes).decode("utf-8")

    # ── Query ChromaDB ───────────────────────────────────────────────────────
    search_text = (query or "").strip() or "skin lesion condition"

    try:
        if uploaded_pil is not None:
            db_results = query_db(query_text=search_text if query else None,
                                  query_image=uploaded_pil)
        else:
            db_results = query_db(query_text=search_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ChromaDB query failed: {exc}")

    # ── Build per-case payload ───────────────────────────────────────────────
    matched_cases: List[MatchedCase] = []
    similar_image_urls: List[str] = []

    base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
    for idx, (uri, meta) in enumerate(
        zip(db_results["uris"][0], db_results["metadatas"][0])
    ):
        filename = os.path.basename(uri)
        img_url = f"{base_url}/images/{filename}"
        similar_image_urls.append(img_url)
        matched_cases.append(
            MatchedCase(
                match_number=idx + 1,
                diagnosis=meta.get("dx", "unknown").replace("_", " ").title(),
                dx_type=meta.get("dx_type", "—"),
                age=str(meta.get("age", "—")),
                sex=meta.get("sex", "—").title(),
                localization=meta.get("localization", "—").replace("_", " ").title(),
                image_url=img_url,
            )
        )

    # ── Build metadata context string (mirrors Skincare.py) ─────────────────
    meta_1 = db_results["metadatas"][0][0]
    meta_2 = db_results["metadatas"][0][1]
    metadata_context = (
        f"Image 1 — Diagnosis: {meta_1.get('dx','N/A')}, "
        f"Type: {meta_1.get('dx_type','N/A')}, "
        f"Age: {meta_1.get('age','N/A')}, Sex: {meta_1.get('sex','N/A')}, "
        f"Localization: {meta_1.get('localization','N/A')}. "
        f"Image 2 — Diagnosis: {meta_2.get('dx','N/A')}, "
        f"Type: {meta_2.get('dx_type','N/A')}, "
        f"Age: {meta_2.get('age','N/A')}, Sex: {meta_2.get('sex','N/A')}, "
        f"Localization: {meta_2.get('localization','N/A')}."
    )

    img_data_1 = encode_image_to_base64(db_results["uris"][0][0])
    img_data_2 = encode_image_to_base64(db_results["uris"][0][1])

    # ── Call GPT-4o Vision (mirrors Skincare.py) ─────────────────────────────
    try:
        if uploaded_b64:
            # User image + 2 reference images → 3-image prompt
            user_text = (
                f"Additional context from the user (if any): {query or 'None provided.'}\n\n"
                f"Matched cases from knowledge base — clinical metadata:\n{metadata_context}\n\n"
                "Images 1 and 2 are the closest matching cases from the database. "
                "Image 3 is the user's uploaded skin image — this is the PRIMARY subject to analyse. "
                "Visually compare Image 3 with Images 1 and 2, then provide your full structured assessment."
            )
            user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data_1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data_2}"}},
                {"type": "image_url", "image_url": {"url": f"data:{uploaded_mime};base64,{uploaded_b64}"}},
            ]
            raw_response = vision_model.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)]
            )
            ai_text = parser.invoke(raw_response)
        else:
            # Text-only path — use 2 reference images
            user_text = (
                f"Additional context from the user (if any): {search_text}\n\n"
                f"Matched cases from knowledge base — clinical metadata:\n{metadata_context}\n\n"
                "The two images are the closest matching cases retrieved from the database. "
                "Analyse what you see visually in these reference images and provide "
                "your full structured assessment now."
            )
            user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data_1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data_2}"}},
            ]
            raw_response = vision_model.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)]
            )
            ai_text = parser.invoke(raw_response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GPT-4o call failed: {exc}")

    # ── Parse the structured markdown into discrete fields ───────────────────
    parsed = parse_ai_response(ai_text)

    return AnalysisResponse(
        image_url=similar_image_urls,
        matched_cases=matched_cases,
        **parsed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("skincare_api_production:app", host="0.0.0.0", port=8001, reload=True)
