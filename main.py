#########################################################
# ASphere API — FINAL STABLE VERSION (Windows Safe)
# RAG HTML + FAISS + Media (NO Playwright)
#########################################################

import os
import json
import base64
import shutil
import tempfile
from io import BytesIO
from typing import List, Literal, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from reportlab.pdfgen import canvas
from moviepy.editor import VideoClip

# LangChain
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# ======================================================
# ENV
# ======================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

client = OpenAI(api_key=OPENAI_API_KEY)

UPLOAD_DIR = "data/uploads"
VECTOR_DIR = "data/vectorstore"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"

# ======================================================
# FASTAPI
# ======================================================
app = FastAPI(title="ASphere API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# MODELS
# ======================================================
PostType = Literal["Texte + Image", "Texte + Vidéo", "Texte + PDF"]

class UploadResponse(BaseModel):
    path: str

class RagIngestRequest(BaseModel):
    urls: List[str] = Field(default_factory=list)
    file_paths: List[str] = Field(default_factory=list)

class GenerateRequest(BaseModel):
    prompt: str
    n_posts: int = 3
    post_type: PostType
    use_rag: bool = True
    rag_k: int = 5

# ======================================================
# UTILS
# ======================================================
def embeddings():
    return OpenAIEmbeddings(model=EMBED_MODEL)

def normalize_urls(urls: List[str]) -> List[str]:
    fixed = []
    for u in urls:
        u = (u or "").strip()
        if not u:
            continue
        if not u.startswith("http"):
            u = "https://" + u
        fixed.append(u)
    return fixed

def load_vectorstore() -> Optional[FAISS]:
    if not os.path.exists(f"{VECTOR_DIR}/index.faiss"):
        return None
    return FAISS.load_local(
        VECTOR_DIR,
        embeddings(),
        allow_dangerous_deserialization=True
    )

def upload_tmpfile(data: bytes, filename: str) -> str:
    r = requests.post(
        "https://tmpfiles.org/api/v1/upload",
        files={"file": (filename, data)},
        timeout=60
    )
    r.raise_for_status()
    js = r.json()
    return js["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")

# ======================================================
# MEDIA
# ======================================================
def generate_image(text: str) -> bytes:
    img = client.images.generate(
        model="gpt-image-1",
        prompt=f"Image professionnelle sans texte illustrant : {text}",
        size="1024x1024"
    )
    return base64.b64decode(img.data[0].b64_json)

def generate_pdf(text: str) -> bytes:
    buf = BytesIO()
    pdf = canvas.Canvas(buf)
    pdf.setFont("Helvetica", 12)
    y = 800
    for line in text.split("\n"):
        if y < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y = 800
        pdf.drawString(50, y, line[:120])
        y -= 18
    pdf.save()
    buf.seek(0)
    return buf.getvalue()

def generate_video(text: str) -> bytes:
    W, H = 1024, 1024
    font = ImageFont.load_default()

    def frame(t):
        img = Image.new("RGB", (W, H), "black")
        draw = ImageDraw.Draw(img)
        draw.text((60, H // 2), text[:300], fill="white", font=font)
        return np.array(img)

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/video.mp4"
        VideoClip(frame, duration=3).write_videofile(
            path, fps=24, codec="libx264", audio=False, logger=None
        )
        return open(path, "rb").read()

# ======================================================
# ROUTES
# ======================================================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    path = f"{UPLOAD_DIR}/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": path}

@app.post("/rag/ingest")
def rag_ingest(req: RagIngestRequest):
    docs = []

    # URLs (HTML classique uniquement)
    for url in normalize_urls(req.urls):
        loaded = WebBaseLoader(url).load()
        usable = [
            d for d in loaded
            if d.page_content and len(d.page_content.strip()) > 200
        ]
        docs += usable

    # Files
    for p in req.file_paths:
        if not os.path.exists(p):
            raise HTTPException(400, f"File not found: {p}")
        if p.lower().endswith(".pdf"):
            docs += PyPDFLoader(p).load()
        else:
            docs += TextLoader(p, encoding="utf-8").load()

    # FILTER EMPTY CONTENT
    docs = [
        d for d in docs
        if d.page_content and len(d.page_content.strip()) > 200
    ]

    if not docs:
        raise HTTPException(
            400,
            "Sources chargées mais aucun texte exploitable "
            "(site vitrine / HTML pauvre / PDF scanné)"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise HTTPException(400, "Découpage en chunks vide")

    vs = load_vectorstore()
    if vs:
        vs.add_documents(chunks)
    else:
        vs = FAISS.from_documents(chunks, embeddings())

    vs.save_local(VECTOR_DIR)
    return {"chunks": len(chunks)}
@app.get("/rag/debug")
def rag_debug(query: str, k: int = 5):
    vs = load_vectorstore()
    if not vs:
        raise HTTPException(400, "RAG non initialisé")

    hits = vs.similarity_search_with_score(query, k=k)

    return [
        {
            "score": float(score),
            "content": doc.page_content[:400]
        }
        for doc, score in hits
    ]



@app.post("/generate-with-media")
def generate(req: GenerateRequest):
    context = ""

    if req.use_rag:
        vs = load_vectorstore()
        if not vs:
            raise HTTPException(400, "RAG activé mais index absent")
        retrieved = vs.similarity_search(req.prompt, k=req.rag_k)
        if not retrieved:
            raise HTTPException(400, "RAG vide")
        context = "\n\n---\n\n".join(d.page_content for d in retrieved)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un générateur de posts basé sur RAG.\n"
                    "Tu DOIS utiliser exclusivement le CONTEXTE.\n"
                    "Retourne STRICTEMENT le JSON demandé."
                )
            },
            {
                "role": "user",
                "content": f"""
CONTEXTE:
{context}

PROMPT:
{req.prompt}

FORMAT:
{{ "posts": [{{"texte":"..."}},{{"texte":"..."}}] }}

NOMBRE: {req.n_posts}
"""
            }
        ]
    )

    posts = json.loads(response.choices[0].message.content)["posts"]
    result = []

    for i, p in enumerate(posts, 1):
        texte = p["texte"]

        if req.post_type == "Texte + Image":
            media = upload_tmpfile(generate_image(texte), f"post_{i}.png")
        elif req.post_type == "Texte + Vidéo":
            media = upload_tmpfile(generate_video(texte), f"post_{i}.mp4")
        else:
            media = upload_tmpfile(generate_pdf(texte), f"post_{i}.pdf")

        result.append({"texte": texte, "media_url": media})

    return {"posts": result}
