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
