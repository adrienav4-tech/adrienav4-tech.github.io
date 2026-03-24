import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
torch.set_num_threads(4)

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

tracker = EmissionsTracker(project_name="RAG_Centrale_Lyon")
tracker.start()

print("--- Début de l'indexation des cours ---")

# Chargement PDF
path_to_pdf = r"C:\Users\adrie\OneDrive\Documents\Centrale Lyon - 4A\ProjetInfo\interface_rag\adrienav4-tech.github.io\rag_centrale"
loader = PyPDFDirectoryLoader(path_to_pdf)
docs = loader.load()
print(f"Documents chargés : {len(docs)} pages")

# Chunking
model_id = "intfloat/multilingual-e5-base"  # Modèle plus léger
tokenizer = AutoTokenizer.from_pretrained(model_id)

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=512,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n#{1,6} ", "```\n", "\n\n", "\n", ".", " "]
)

chunks = text_splitter.split_documents(docs)
print(f"Chunks créés : {len(chunks)}")

# Test du modèle AVANT LangChain
print(f"Chargement et test du modèle : {model_id}...")
try:
    model = SentenceTransformer(model_id, device='cpu')
    _ = model.encode("passage: test")
    print("Modèle chargé avec succès")
except Exception as e:
    print(f"Erreur au chargement : {e}")
    tracker.stop()
    exit(1)

# Embeddings LangChain
embeddings = HuggingFaceEmbeddings(
    model_name=model_id,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 8}
)

# Ajout préfixe E5
for chunk in chunks:
    if not chunk.page_content.startswith("passage: "):
        chunk.page_content = f"passage: {chunk.page_content}"

# Vectorisation
print("Début de la vectorisation...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='db_centrale_lyon',
    collection_metadata={"hnsw:space": "cosine"}
)

print("Base créée dans : db_centrale_lyon")
emissions = tracker.stop()
print(f"Émissions : {emissions:.4f} kg CO2")