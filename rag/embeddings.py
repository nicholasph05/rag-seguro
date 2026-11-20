"""Embeddings: un único modelo para documentos y preguntas."""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from rag.config import MODELO_EMBEDDINGS


@lru_cache(maxsize=1)
def obtener_modelo() -> SentenceTransformer:
    """Carga el modelo una sola vez por proceso (tarda unos segundos)."""
    return SentenceTransformer(MODELO_EMBEDDINGS)


def vectorizar(textos: list[str]) -> list[list[float]]:
    """Vectores normalizados (longitud 1): así la distancia L2 ordena igual que el coseno."""
    return obtener_modelo().encode(textos, normalize_embeddings=True).tolist()


def contar_tokens(texto: str) -> int:
    """Cuenta tokens con el mismo tokenizador del modelo (incluye tokens especiales)."""
    return len(obtener_modelo().tokenizer(texto, verbose=False)["input_ids"])
