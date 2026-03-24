import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"   

import torch
torch.set_num_threads(4) 

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma                      
from langchain_ollama import OllamaLLM                  
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(project_name="RAG_Centrale_Lyon_phase_RAG")
tracker.start()

# 1. Config Embeddings
persist_directory = 'db_centrale_lyon'
model_id = "intfloat/multilingual-e5-base"
embeddings = HuggingFaceEmbeddings(
    model_name=model_id,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. Chargement Vectorstore
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_name="langchain"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Config LLM via Ollama
llm = OllamaLLM(model="qwen2.5:1.5b")                  

# 4. Prompt
template = """Tu es un assistant pédagogique pour les étudiants de l'École Centrale de Lyon.
Utilise les extraits de cours suivants pour répondre à la question.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :"""

prompt = PromptTemplate.from_template(template)

# 5. Chaîne RAG
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- TEST ---
query = "Qu'est-ce que la régression linéaire ?"
query_for_search = f"query: {query}"


print("\n--- GÉNÉRATION EN COURS ---")
response = rag_chain.invoke(query_for_search)
print(response)


emissions = tracker.stop()
