"""Configuración central del RAG. Un solo lugar para cambiar rutas y parámetros."""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_DOCS = RAIZ / "data" / "docs"   # cada subcarpeta es un rol: data/docs/<rol>/archivo
DIR_CHROMA = RAIZ / "chroma_db"     # base vectorial persistente (no se sube a Git)
COLECCION = "documentos"

MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_TOKENS_FRAGMENTO = 120  # margen bajo el límite de 128 del modelo
SOLAPE_ORACIONES = 1        # oraciones repetidas entre fragmentos consecutivos

# Recuperación y generación
TOP_K = 4  # fragmentos que se envían al LLM
UMBRAL_SIMILITUD = 0.40  # por debajo, el fragmento se descarta como ruido
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODELO = os.getenv("LLM_MODELO", "llama3.2:3b")
LLM_TEMPERATURA = 0.1  # baja: respuestas fieles al contexto
LLM_NUM_CTX = 4096
