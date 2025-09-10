# --------------------------
# Imports & Setup
# --------------------------
import gradio as gr
import uuid
import base64
import io
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from PIL import Image

# --------------------------
# Qdrant Cloud Connection
# --------------------------
QDRANT_URL = "https://ff4da494-27b1-413c-ba58-d5ea14932fe1.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.jjeB1JgnUSlb1hOOKMdRpVvMrUER57-udT-X1AWXT1E"
COLLECTION_NAME = "lost_and_found"

# CLIP model (text + image embeddings)
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
embedder = SentenceTransformer(MODEL_NAME)

# CLIP ViT-B/32 always gives 512-dimensional embeddings
VECTOR_SIZE = 512

# Qdrant Client (Cloud)
qclient = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Ensure collection exists
qclient.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

# --------------------------
# Helper Functions
# --------------------------

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def base64_to_image(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))

def embed_text(text: str):
    return embedder.encode(text).tolist()

def embed_image(img: Image.Image):
    return embedder.encode(img).tolist()

# --------------------------
# Core Logic: Add, Search, Clear
# --------------------------

def add_item(image, description, finder_name, finder_phone):
    if image is None or description.strip() == "":
        return "Please provide both an image and a description."

    embedding = embed_image(image)
    img_b64 = image_to_base64(image)

    metadata = {
        "description": description,
        "finder_name": finder_name if finder_name.strip() else "NA",
        "finder_phone": finder_phone if finder_phone.strip() else "NA",
        "image_b64": img_b64
    }

    qclient.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=metadata
            )
        ]
    )
    return "Item successfully added!"

def search_items(query_text, query_image):
    if not query_text and query_image is None:
        return "Please enter text or upload an image to search.", []

    if query_image:
        query_vector = embed_image(query_image)
    else:
        query_vector = embed_text(query_text)

    results = qclient.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5
    )

    if not results:
        return "No matches found.", []

    gallery = []
    output_text = "### Matches Found\n\n"
    for r in results:
        desc = r.payload.get("description", "No description")
        name = r.payload.get("finder_name", "NA")
        phone = r.payload.get("finder_phone", "NA")
        output_text += f"- *{desc}* — Finder: {name}, Phone: {phone}\n"

        if "image_b64" in r.payload:
            try:
                img = base64_to_image(r.payload["image_b64"])
                gallery.append(img)
            except Exception:
                pass

    return output_text, gallery

def clear_database():
    qclient.delete_collection(COLLECTION_NAME)
    qclient.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    return "Database cleared successfully."

