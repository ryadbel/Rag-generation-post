#########################################################
# ASphere API — FINAL BACKEND
#########################################################

import os, re, json, uuid, base64, shutil, tempfile
from datetime import datetime
from typing import List, Optional
from io import BytesIO

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from moviepy.editor import VideoClip

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# ---------------- ENV ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

client = OpenAI(api_key=OPENAI_API_KEY)

BASE = "data"
UPLOAD_DIR = os.path.join(BASE, "uploads")
VECTOR_DIR = os.path.join(BASE, "vectorstore")
HISTORY_FILE = os.path.join(BASE, "media_history.json")

os.makedirs(BASE, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)

# ---------------- FASTAPI ----------------
app = FastAPI(title="ASphere API — Final", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------
class RagIngestRequest(BaseModel):
    urls: List[str] = Field(default_factory=list)
    file_paths: List[str] = Field(default_factory=list)

class GenerateRequest(BaseModel):
    prompt: Optional[str] = ""
    use_rag: Optional[bool] = False  # Par défaut FALSE
    rag_k: Optional[int] = 5
    post_type: Optional[str] = "Texte + Image"

class RegenerateImageRequest(BaseModel):
    previous_image_id: str
    new_prompt: str

# ---------------- UTILS ----------------
def embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

def normalize_urls(urls: List[str]) -> List[str]:
    out = []
    for u in urls:
        u = (u or "").strip()
        if not u:
            continue
        if not u.startswith("http"):
            u = "https://" + u
        out.append(u)
    return out

def extract_n_posts(prompt: str, default=1, max_posts=5):
    m = re.search(r"(\d+)\s*posts?", prompt.lower())
    if m:
        return min(max(int(m.group(1)), 1), max_posts)
    return default

def load_vectorstore():
    if not os.path.exists(os.path.join(VECTOR_DIR, "index.faiss")):
        return None
    return FAISS.load_local(VECTOR_DIR, embeddings(), allow_dangerous_deserialization=True)

def upload_tmpfile(data: bytes, name: str) -> str:
    r = requests.post(
        "https://tmpfiles.org/api/v1/upload",
        files={"file": (name, data)},
        timeout=60
    )
    r.raise_for_status()
    return r.json()["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")

def load_history():
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_history(h):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(h, f, indent=2, ensure_ascii=False)

def store_media(**kwargs):
    h = load_history()
    rec = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat(),
        "forgotten": False,
        **kwargs
    }
    h.append(rec)
    save_history(h)
    return rec

# ---------------- MEDIA ----------------
def generate_image(prompt: str) -> bytes:
    img = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1
    )
    img_url = img.data[0].url
    response = requests.get(img_url)
    return response.content

# ---------------- ROUTES ----------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": path}

@app.post("/rag/ingest")
def rag_ingest(req: RagIngestRequest):
    docs = []

    for u in normalize_urls(req.urls):
        try:
            loaded = WebBaseLoader(u).load()
            docs += [d for d in loaded if d.page_content and len(d.page_content.strip()) > 200]
        except Exception as e:
            print(f"Erreur chargement {u}: {e}")

    for p in req.file_paths:
        if not os.path.exists(p):
            raise HTTPException(400, f"File not found: {p}")
        try:
            if p.lower().endswith(".pdf"):
                docs += PyPDFLoader(p).load()
            else:
                docs += TextLoader(p, encoding="utf-8").load()
        except Exception as e:
            print(f"Erreur fichier {p}: {e}")

    docs = [d for d in docs if d.page_content and len(d.page_content.strip()) > 200]
    if not docs:
        raise HTTPException(400, "Aucune source exploitable (HTML vide / PDF scanné)")

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)

    vs = load_vectorstore()
    if vs:
        vs.add_documents(chunks)
    else:
        vs = FAISS.from_documents(chunks, embeddings())
    vs.save_local(VECTOR_DIR)

    return {"chunks": len(chunks), "message": "Index RAG créé avec succès"}

@app.get("/rag/debug")
def rag_debug(query: str, k: int = 5):
    vs = load_vectorstore()
    if not vs:
        raise HTTPException(400, "RAG non initialisé")
    hits = vs.similarity_search_with_score(query, k=k)
    return [{"score": float(s), "content": d.page_content[:400]} for d, s in hits]

@app.get("/rag/status")
def rag_status():
    """Vérifie si le RAG est initialisé"""
    vs = load_vectorstore()
    return {
        "initialized": vs is not None,
        "index_exists": os.path.exists(os.path.join(VECTOR_DIR, "index.faiss"))
    }

@app.post("/generate-with-media")
def generate_with_media(req: GenerateRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(400, "Le prompt est vide")

    post_type = req.post_type if req.post_type in ["Texte + Image"] else "Texte + Image"
    n_posts = extract_n_posts(req.prompt)

    context = ""
    if req.use_rag:
        vs = load_vectorstore()
        if not vs:
            raise HTTPException(400, "RAG activé mais index absent. Veuillez d'abord indexer des sources.")
        retrieved = vs.similarity_search(req.prompt, k=req.rag_k or 5)
        context = "\n\n---\n\n".join(d.page_content for d in retrieved)

    llm_prompt = f"""
{"CONTEXTE RAG:" if context else ""}
{context}

INSTRUCTION:
Tu dois générer EXACTEMENT {n_posts} posts distincts pour les réseaux sociaux.

PROMPT UTILISATEUR:
{req.prompt}

FORMAT JSON STRICT (renvoie uniquement ce JSON, rien d'autre):
{{"posts":[{{"texte":"contenu du post 1"}},{{"texte":"contenu du post 2"}}]}}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Tu es un expert en création de contenu pour réseaux sociaux. Retourne uniquement du JSON valide."},
            {"role": "user", "content": llm_prompt}
        ]
    )

    posts = json.loads(res.choices[0].message.content).get("posts", [])[:n_posts]
    out = []

    for i, p in enumerate(posts, 1):
        text = (p.get("texte") or "").strip()
        if not text:
            continue

        img_prompt = f"Professional social media illustration, modern and clean design, no text overlay: {text[:200]}"
        
        try:
            img = generate_image(img_prompt)
            url = upload_tmpfile(img, f"post_{i}.png")
        except Exception as e:
            print(f"Erreur génération image: {e}")
            url = "https://via.placeholder.com/1024x1024?text=Image+Error"

        out.append(store_media(
            texte=text,
            media_url=url,
            media_type="image",
            media_prompt=img_prompt,
            media_description="Image générée"
        ))

    return {"n_posts_detected": n_posts, "posts": out}

@app.post("/image/regenerate")
def regenerate_image(req: RegenerateImageRequest):
    h = load_history()
    prev = next((x for x in h if x["id"] == req.previous_image_id), None)
    if not prev:
        raise HTTPException(404, "Image introuvable")

    combined_prompt = f"""
Professional social media illustration, maintain overall style and composition.
No text overlay in the image.

Previous concept:
{prev.get("media_prompt","")}

New instruction:
{req.new_prompt}
"""

    try:
        img = generate_image(combined_prompt)
        url = upload_tmpfile(img, "regen.png")
    except Exception as e:
        raise HTTPException(500, f"Erreur génération: {str(e)}")

    return store_media(
        texte=prev["texte"],
        media_url=url,
        media_type="image",
        media_prompt=req.new_prompt,
        media_description="Image régénérée",
        parent_id=prev["id"]
    )

@app.get("/history")
def get_history():
    """Récupère l'historique des médias générés"""
    return load_history()

@app.delete("/history/{media_id}")
def delete_media(media_id: str):
    """Marque un média comme oublié"""
    h = load_history()
    media = next((x for x in h if x["id"] == media_id), None)
    if not media:
        raise HTTPException(404, "Média introuvable")
    media["forgotten"] = True
    save_history(h)
    return {"message": "Média marqué comme oublié"}