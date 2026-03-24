"""
backend.py  —  Serveur FastAPI pour le RAG Centrale Lyon
=========================================================
Relie le frontend HTML à :
  - ChromaDB (db_centrale_lyon) via langchain-chroma
  - Ollama (qwen2.5:1.5b)       via langchain-ollama
  - Embeddings multilingual-e5-base
  - CodeCarbon                   mesure CO₂ réelle par requête

Démarrage :
    pip install fastapi uvicorn codecarbon langchain-chroma \
                langchain-huggingface langchain-ollama langchain-core
    uvicorn backend:app --reload --port 8000

Le frontend doit pointer sur http://localhost:8000
"""

import os, time, logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
torch.set_num_threads(4)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from codecarbon import EmissionsTracker

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("ECL-RAG")

# ── Config ─────────────────────────────────────────────────────
PERSIST_DIR  = "db_centrale_lyon"   # répertoire ChromaDB (à côté de ce fichier)
MODEL_EMBED  = "intfloat/multilingual-e5-base"
MODEL_LLM    = "qwen2.5:1.5b"
TOP_K        = 5                    # nombre de chunks récupérés

# ── Initialisation (au démarrage, une seule fois) ──────────────
log.info("Chargement du modèle d'embeddings…")
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_EMBED,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

log.info("Connexion à ChromaDB…")
if not os.path.isdir(PERSIST_DIR):
    raise RuntimeError(
        f"Répertoire ChromaDB introuvable : '{PERSIST_DIR}'\n"
        "Lancez d'abord ingest.py pour créer la base vectorielle."
    )

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    collection_name="langchain",
    collection_metadata={"hnsw:space": "cosine"},
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

log.info("Connexion à Ollama…")
llm = OllamaLLM(model=MODEL_LLM)

# ── Prompt (identique à rag.py) ────────────────────────────────
PROMPT_TEMPLATE = """Tu es un assistant pédagogique pour les étudiants de l'École Centrale de Lyon.
Utilise les extraits de cours suivants pour répondre à la question.
Si les extraits ne contiennent pas l'information, dis-le clairement.
Indique les sources pertinentes sous la forme [SOURCE: nom_du_fichier].

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :"""

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

log.info("Backend prêt ✓")

# ── FastAPI ────────────────────────────────────────────────────
app = FastAPI(title="ECL RAG Backend", version="1.0")

# CORS : autorise le HTML servi depuis n'importe quelle origine locale
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # en production, restreindre à votre domaine
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schémas ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    history: Optional[list] = []   # [{role, content}, …]  (non utilisé pour l'instant)

class ChatResponse(BaseModel):
    response: str
    tokens_est: int
    cost_eur: float
    co2_kg: float          # mesure CodeCarbon réelle
    co2_label: str         # formaté (ex : "0.042 mgCO₂")
    co2_analogy: str       # équivalence pédagogique
    sources: list[str]     # noms de fichiers sources trouvés
    duration_s: float      # temps de réponse en secondes

class StatusResponse(BaseModel):
    status: str
    db_chunks: int
    model_embed: str
    model_llm: str

# ── Routes ────────────────────────────────────────────────────
@app.get("/status", response_model=StatusResponse)
def status():
    """Vérifie que le backend est opérationnel et renvoie des infos utiles."""
    try:
        count = vectorstore._collection.count()
    except Exception:
        count = -1
    return StatusResponse(
        status="ok",
        db_chunks=count,
        model_embed=MODEL_EMBED,
        model_llm=MODEL_LLM,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    """
    Route principale : reçoit une question, effectue le RAG complet,
    mesure le CO₂ réel avec CodeCarbon et renvoie tout au frontend.
    """
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    # Préfixe E5 pour la recherche (identique à ingest.py)
    query_for_search = f"query: {query}" if not query.startswith("query:") else query

    # ── CodeCarbon ─────────────────────────────────────────────
    tracker = EmissionsTracker(
        project_name="RAG_Centrale_Lyon",
        log_level="error",          # silencieux dans les logs
        save_to_file=False,         # pas de fichier emissions.csv
        tracking_mode="process",
    )

    t0 = time.perf_counter()
    tracker.start()

    try:
        response_text = rag_chain.invoke(query_for_search)
    except Exception as e:
        tracker.stop()
        log.error(f"Erreur RAG : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération : {e}")

    co2_kg = tracker.stop() or 0.0   # kg CO₂eq réel mesuré par CodeCarbon
    duration_s = time.perf_counter() - t0

    # ── Métriques ──────────────────────────────────────────────
    tokens_est = round((len(query) + len(response_text)) / 4)
    cost_eur   = round(tokens_est * 0.000003, 6)

    co2_label   = _format_carbon(co2_kg)
    co2_analogy = _carbon_analogy(co2_kg)

    # ── Sources extraites de la réponse ───────────────────────
    import re
    sources = list({
        m.strip()
        for m in re.findall(r"\[SOURCE:\s*([^\]]+)\]", response_text)
    })

    log.info(
        f"Query: {query[:60]}… | {tokens_est} tok | "
        f"{co2_kg*1e6:.3f} µgCO₂ | {duration_s:.2f}s"
    )

    return ChatResponse(
        response=response_text,
        tokens_est=tokens_est,
        cost_eur=cost_eur,
        co2_kg=co2_kg,
        co2_label=co2_label,
        co2_analogy=co2_analogy,
        sources=sources,
        duration_s=round(duration_s, 3),
    )


@app.get("/docs-list")
def docs_list():
    """
    Retourne la liste des documents uniques présents dans ChromaDB
    (extraits depuis les métadonnées 'source' des chunks).
    """
    try:
        results = vectorstore._collection.get(include=["metadatas"])
        sources = sorted({
            os.path.basename(m.get("source", "Inconnu"))
            for m in results["metadatas"]
            if m
        })
        return {"documents": sources, "total_chunks": len(results["metadatas"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Helpers ───────────────────────────────────────────────────
def _format_carbon(co2_kg: float) -> str:
    """Formate un poids CO₂ en unité lisible."""
    co2_g = co2_kg * 1000
    if co2_g < 0.000_001:
        return f"{co2_g * 1_000_000:.4f} µgCO₂"
    if co2_g < 0.001:
        return f"{co2_g * 1_000:.4f} mgCO₂"
    if co2_g < 1:
        return f"{co2_g * 1_000:.3f} mgCO₂"
    return f"{co2_g:.4f} gCO₂"


def _carbon_analogy(co2_kg: float) -> str:
    """
    Équivalences pédagogiques (sources ADEME / RTE).
      - 1 e-mail envoyé        ≈ 4 gCO₂
      - Charge smartphone      ≈ 8.22 gCO₂
      - 1 km voiture thermique ≈ 120 gCO₂
    """
    co2_g = co2_kg * 1000
    if co2_g == 0:
        return "non mesurable"
    email_g    = 4.0
    charge_g   = 8.22
    voiture_gm = 0.120  # g par mètre

    if co2_g < voiture_gm:
        mm = co2_g / voiture_gm * 1000
        return f"≈ {mm:.3f} mm en voiture thermique"
    if co2_g < email_g * 0.01:
        pct = co2_g / email_g * 100
        return f"≈ {pct:.4f} % d'un e-mail (ADEME)"
    if co2_g < charge_g * 0.01:
        pct = co2_g / charge_g * 100
        return f"≈ {pct:.4f} % d'une charge smartphone"
    if co2_g < email_g:
        pct = co2_g / email_g * 100
        return f"≈ {pct:.2f} % d'un e-mail envoyé (ADEME)"
    return f"≈ {co2_g / charge_g:.4f} × charge smartphone complète"
