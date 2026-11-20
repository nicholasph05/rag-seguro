"""Consulta RAG: pregunta -> fragmentos más cercanos -> respuesta del LLM con citas.

Uso (desde la raíz del repo): python -m rag.consulta "¿Cuántos días de vacaciones tengo?"
"""
import argparse
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from rag.config import COLECCION, DIR_CHROMA, TOP_K, UMBRAL_SIMILITUD
from rag.embeddings import vectorizar
from rag.llm import generar

SIN_RESPUESTA = "No encuentro esa información en los documentos disponibles."

SISTEMA = f"""Eres el asistente interno de Nébula Logística S.A.S.
Reglas:
1. Responde en español usando SOLO la información del contexto.
2. Responde TODAS las partes de la pregunta.
3. Termina cada frase con su fuente entre corchetes, por ejemplo [1].
4. No comentes qué información falta ni agregues datos que no estén en el contexto.
5. Si el contexto no contiene la respuesta, responde exactamente: "{SIN_RESPUESTA}"
"""


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


def recuperar(pregunta: str, k: int = TOP_K, umbral: float = UMBRAL_SIMILITUD) -> list[Resultado]:
    res = _coleccion().query(
        query_embeddings=vectorizar([pregunta]),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    resultados = [
        Resultado(
            texto=doc,
            fuente=meta["fuente"],
            pagina=meta["pagina"],
            rol=meta["rol"],
            similitud=1 - dist / 2,  # vectores normalizados: L2² = 2 - 2·coseno
        )
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    return [r for r in resultados if r.similitud >= umbral]


def construir_contexto(resultados: list[Resultado]) -> str:
    return "\n\n".join(
        f"[{i}] ({r.fuente}, pág. {r.pagina})\n{r.texto}" for i, r in enumerate(resultados, start=1)
    )


def responder(pregunta: str, k: int = TOP_K) -> tuple[str, list[Resultado]]:
    resultados = recuperar(pregunta, k)
    if not resultados:  # nada relevante: no se llama al LLM, así no puede inventar
        return SIN_RESPUESTA, []
    usuario = (
        f"Contexto:\n{construir_contexto(resultados)}\n\n"
        f"Pregunta: {pregunta}\n\n"
        "Responde todas las partes de la pregunta y cita cada frase con [n]."
    )
    return generar(SISTEMA, usuario), resultados


def main() -> None:
    parser = argparse.ArgumentParser(description="Pregunta al asistente RAG")
    parser.add_argument("pregunta")
    parser.add_argument("-k", type=int, default=TOP_K, help="fragmentos a recuperar")
    parser.add_argument("--ver-contexto", action="store_true", help="muestra los fragmentos enviados al LLM")
    args = parser.parse_args()

    respuesta, resultados = responder(args.pregunta, args.k)
    if args.ver_contexto and resultados:
        print(f"\n--- Contexto enviado al LLM ---\n{construir_contexto(resultados)}\n-------------------------------")
    print(f"\n{respuesta}\n\nFuentes recuperadas:")
    for i, r in enumerate(resultados, start=1):
        print(f"  [{i}] {r.similitud:.3f}  [{r.rol}] {r.fuente}, pág. {r.pagina}")
    if not resultados:
        print(f"  (ningún fragmento supera el umbral de {UMBRAL_SIMILITUD})")


if __name__ == "__main__":
    main()
