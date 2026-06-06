import base64
import io
import json
import os
import re
from typing import Any

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from datasets import load_dataset
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image

load_dotenv()

APP_NAME = "DermaInsight AI API"
IMAGE_DIR = "./data/skin_images"
DB_PATH = "./data/skin.db"
COLLECTION_NAME = "skin_collection"
EXPECTED_COUNT = 700

app = FastAPI(
    title=APP_NAME,
    description="Multimodal skin lesion education API using image/text retrieval and GPT-4o.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(IMAGE_DIR, exist_ok=True)
app.mount(
    "/static/skin-images",
    StaticFiles(directory=IMAGE_DIR),
    name="skin_images",
)


def load_balanced_dataset():
    dataset = load_dataset("marmal88/skin_cancer", split="train")
    counts = {}
    selected = []
    for index, sample in enumerate(dataset):
        category = sample.get("dx", "unknown")
        if counts.get(category, 0) < 100:
            selected.append(index)
            counts[category] = counts.get(category, 0) + 1
    return dataset.select(selected)


def prepare_images_and_metadata(dataset):
    uris, metadatas, ids = [], [], []
    for index, sample in enumerate(dataset):
        image_id = sample.get("image_id", f"img_{index}")
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            sample["image"].save(image_path)

        uris.append(image_path)
        ids.append(image_id)
        metadatas.append(
            {
                "image_id": image_id,
                "lesion_id": sample.get("lesion_id", ""),
                "dx": sample.get("dx", ""),
                "dx_type": sample.get("dx_type", ""),
                "age": str(sample.get("age", "")),
                "sex": sample.get("sex", ""),
                "localization": sample.get("localization", ""),
            }
        )
    return uris, metadatas, ids


image_loader = ImageLoader()
embedding_function = None
chroma_client = None
skin_collection = None
vision_model = None


def get_embedding_function():
    global embedding_function
    if embedding_function is None:
        embedding_function = OpenCLIPEmbeddingFunction()
    return embedding_function


def get_skin_collection():
    global chroma_client, skin_collection
    if skin_collection is None:
        chroma_client = chromadb.PersistentClient(path=os.path.abspath(DB_PATH))
        skin_collection = chroma_client.get_or_create_collection(
            COLLECTION_NAME,
            embedding_function=get_embedding_function(),
            data_loader=image_loader,
        )
    return skin_collection


def get_vision_model():
    global vision_model
    if vision_model is None:
        vision_model = ChatOpenAI(model="gpt-4o", temperature=0.0)
    return vision_model


def ensure_knowledge_base_ready():
    collection = get_skin_collection()
    if collection.count() >= EXPECTED_COUNT:
        return

    dataset = load_balanced_dataset()
    uris, metadatas, ids = prepare_images_and_metadata(dataset)
    existing_ids = set(collection.get().get("ids", []))
    to_add = [
        (uri, metadata, image_id)
        for uri, metadata, image_id in zip(uris, metadatas, ids)
        if image_id not in existing_ids
    ]

    if to_add:
        collection.add(
            uris=[item[0] for item in to_add],
            metadatas=[item[1] for item in to_add],
            ids=[item[2] for item in to_add],
        )


def encode_image_to_base64(source: str | Image.Image) -> str:
    if isinstance(source, str):
        with open(source, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    buffer = io.BytesIO()
    source.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def read_upload_as_image(uploaded_file: UploadFile | None) -> Image.Image | None:
    if uploaded_file is None:
        return None

    if not uploaded_file.content_type or not uploaded_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = uploaded_file.file.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read uploaded image.") from exc


def query_db(query_text: str | None = None, query_image: Image.Image | None = None, results: int = 2):
    import numpy as np
    import torch

    collection = get_skin_collection()

    if query_image is not None and query_text:
        embedder = get_embedding_function()
        model = embedder._model
        preprocess = embedder._preprocess
        tokenizer = embedder._tokenizer

        image_tensor = preprocess(query_image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            image_embedding = model.encode_image(image_tensor).float().numpy()[0]
            text_embedding = model.encode_text(tokenizer([query_text])).float().numpy()[0]

        combined = (image_embedding + text_embedding) / 2
        combined = combined / np.linalg.norm(combined)

        return collection.query(
            query_embeddings=[combined.tolist()],
            n_results=results,
            include=["uris", "distances", "metadatas"],
        )

    if query_image is not None:
        return collection.query(
            query_images=[np.array(query_image.convert("RGB"))],
            n_results=results,
            include=["uris", "distances", "metadatas"],
        )

    return collection.query(
        query_texts=[query_text],
        n_results=results,
        include=["uris", "distances", "metadatas"],
    )


def image_url_for(request: Request, image_path: str) -> str:
    filename = os.path.basename(image_path)
    return str(request.url_for("skin_images", path=filename))


def build_similar_images(results: dict[str, Any], request: Request) -> list[dict[str, Any]]:
    uris = results.get("uris", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    similar_images = []
    for index, uri in enumerate(uris):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        similar_images.append(
            {
                "rank": index + 1,
                "image_url": image_url_for(request, uri),
                "source_path": uri,
                "distance": distance,
                "metadata": metadata,
            }
        )
    return similar_images


def build_metadata_context(similar_images: list[dict[str, Any]]) -> str:
    context_lines = []
    for image in similar_images:
        metadata = image["metadata"]
        context_lines.append(
            (
                f"Image {image['rank']} - Diagnosis: {metadata.get('dx', 'N/A')}; "
                f"Type: {metadata.get('dx_type', 'N/A')}; "
                f"Age: {metadata.get('age', 'N/A')}; "
                f"Sex: {metadata.get('sex', 'N/A')}; "
                f"Localization: {metadata.get('localization', 'N/A')}."
            )
        )
    return "\n".join(context_lines)


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def analyze_with_model(
    user_text: str,
    uploaded_image: Image.Image | None,
    similar_images: list[dict[str, Any]],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "User text/context:\n"
                f"{user_text or 'None provided.'}\n\n"
                "Retrieved similar cases:\n"
                f"{build_metadata_context(similar_images)}\n\n"
                "Return only valid JSON. Do not wrap it in markdown."
            ),
        }
    ]

    for image in similar_images:
        image_b64 = encode_image_to_base64(image["source_path"])
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            }
        )

    if uploaded_image is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(uploaded_image)}"},
            }
        )

    system_prompt = """
You are a dermatology education assistant. Analyze the user's uploaded skin image when present,
the user's text, and the retrieved similar cases. This is educational support only, not diagnosis.
Never invent symptoms the user did not report.

Return only valid JSON with this exact shape:
{
  "condition_match": "plain English possible visual match",
  "reasoning": {
    "visual_observations": ["observable visual point 1", "observable visual point 2"],
    "similar_case_reasoning": ["why retrieved case 1 may be relevant", "why retrieved case 2 may be relevant"],
    "user_context_reasoning": "how the user's text affects the explanation"
  },
  "about_condition": "2-3 plain English educational sentences",
  "warning_symptoms": [
    "If a specific observable sign happens, then take a specific action.",
    "If it is hurting, bleeding, rapidly changing, or becoming infected, then arrange medical review promptly."
  ],
  "overall_assessment": {
    "severity": "Minor, Moderate, or Major",
    "summary": "short calm assessment"
  },
  "recommended_next_steps": ["step 1", "step 2"],
  "disclaimer": "educational tool only; consult a licensed dermatologist"
}
"""

    raw_response = get_vision_model().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]
    )

    try:
        return extract_json(raw_response.content)
    except Exception:
        return {
            "condition_match": "Unable to parse model response",
            "reasoning": {
                "visual_observations": [],
                "similar_case_reasoning": [],
                "user_context_reasoning": user_text or "None provided.",
            },
            "about_condition": "",
            "warning_symptoms": [],
            "overall_assessment": {
                "severity": "Unknown",
                "summary": raw_response.content,
            },
            "recommended_next_steps": ["Consult a licensed dermatologist for an accurate diagnosis."],
            "disclaimer": "This is an educational tool only and does not constitute medical advice.",
        }


@app.get("/health")
def health():
    collection_count = None
    try:
        collection_count = get_skin_collection().count()
    except Exception as exc:
        return {
            "status": "degraded",
            "collection_count": None,
            "detail": str(exc),
        }

    return {
        "status": "ok",
        "collection_count": collection_count,
    }


@app.post("/api/v1/skin-analysis")
def skin_analysis(
    request: Request,
    text: str = Form(default=""),
    image: UploadFile | None = File(default=None),
):
    if not text.strip() and image is None:
        raise HTTPException(status_code=400, detail="Provide text, image, or both.")

    try:
        ensure_knowledge_base_ready()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not open or prepare the ChromaDB knowledge base. "
                "If Streamlit is running, stop it and retry the API request. "
                f"Error: {exc}"
            ),
        ) from exc

    uploaded_image = read_upload_as_image(image)
    search_text = text.strip() or "skin lesion condition"
    results = query_db(query_text=search_text if text.strip() else None, query_image=uploaded_image)
    similar_images = build_similar_images(results, request)
    model_analysis = analyze_with_model(text.strip(), uploaded_image, similar_images)

    return {
        "status": "success",
        "input": {
            "text": text.strip(),
            "image_uploaded": uploaded_image is not None,
        },
        "image_url": [image["image_url"] for image in similar_images],
        "similar_images": similar_images,
        "analysis": model_analysis,
        "warning_symptoms": model_analysis.get("warning_symptoms", []),
        "disclaimer": model_analysis.get(
            "disclaimer",
            "This is an educational tool only and does not constitute medical advice.",
        ),
    }
