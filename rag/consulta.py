"""Consulta RAG: pregunta -> fragmentos más cercanos -> respuesta del LLM con citas.

Uso (desde la raíz del repo): python -m rag.consulta "¿Cuántos días de vacaciones tengo?"
"""
import argparse
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from rag.config import COLECCION, DIR_CHROMA, TOP_K
from rag.embeddings import vectorizar
from rag.llm import generar

SISTEMA = (
    "Eres el asistente interno de Nébula Logística S.A.S. "
    "Responde en español, de forma breve, usando SOLO la información del contexto. "
    "Después de cada afirmación indica la fuente entre corchetes, por ejemplo [1]. "
    "Si el contexto no contiene la respuesta, di exactamente: "
    "\"No encuentro esa información en los documentos disponibles.\""
)


@dataclass
class Resultado:
    texto: str
    fuente: str
    pagina: int
    rol: str
    similitud: float


def _coleccion():
    cliente = chromadb.PersistentClient(path=str(DIR_CHROMA), settings=Settings(anonymized_telemetry=False))
    return cliente.get_collection(COLECCION)


def recuperar(pregunta: str, k: int = TOP_K) -> list[Resultado]:
    res = _coleccion().query(
        query_embeddings=vectorizar([pregunta]),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        Resultado(
            texto=doc,
            fuente=meta["fuente"],
            pagina=meta["pagina"],
            rol=meta["rol"],
            similitud=1 - dist / 2,  # vectores normalizados: L2² = 2 - 2·coseno
        )
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def construir_contexto(resultados: list[Resultado]) -> str:
    return "\n\n".join(
        f"[{i}] ({r.fuente}, pág. {r.pagina})\n{r.texto}" for i, r in enumerate(resultados, start=1)
    )


def responder(pregunta: str, k: int = TOP_K) -> tuple[str, list[Resultado]]:
    resultados = recuperar(pregunta, k)
    usuario = f"Contexto:\n{construir_contexto(resultados)}\n\nPregunta: {pregunta}"
    return generar(SISTEMA, usuario), resultados


def main() -> None:
    parser = argparse.ArgumentParser(description="Pregunta al asistente RAG")
    parser.add_argument("pregunta")
    parser.add_argument("-k", type=int, default=TOP_K, help="fragmentos a recuperar")
    args = parser.parse_args()

    respuesta, resultados = responder(args.pregunta, args.k)
    print(f"\n{respuesta}\n\nFuentes recuperadas:")
    for i, r in enumerate(resultados, start=1):
        print(f"  [{i}] {r.similitud:.3f}  [{r.rol}] {r.fuente}, pág. {r.pagina}")


if __name__ == "__main__":
    main()
