"""Indexación: documentos -> fragmentos -> embeddings -> ChromaDB.

Uso (desde la raíz del repo): python -m rag.indexar
"""
from collections import Counter

import chromadb
from chromadb.config import Settings

from rag.carga import cargar_documentos
from rag.config import COLECCION, DIR_CHROMA, DIR_DOCS
from rag.embeddings import contar_tokens, vectorizar
from rag.fragmentacion import fragmentar


def main() -> None:
    paginas = cargar_documentos(DIR_DOCS)
    fragmentos = fragmentar(paginas)
    if not fragmentos:
        print(f"No hay documentos en {DIR_DOCS}")
        return

    cliente = chromadb.PersistentClient(
        path=str(DIR_CHROMA),
        settings=Settings(anonymized_telemetry=False),  # nada sale del equipo
    )
    # Reindexación completa: evita fragmentos huérfanos de documentos editados o borrados
    if COLECCION in [c if isinstance(c, str) else c.name for c in cliente.list_collections()]:
        cliente.delete_collection(COLECCION)
    coleccion = cliente.create_collection(COLECCION)

    textos = [f.texto for f in fragmentos]
    coleccion.add(
        ids=[f.id for f in fragmentos],
        documents=textos,
        embeddings=vectorizar(textos),
        metadatas=[f.metadatos for f in fragmentos],
    )

    tokens = [contar_tokens(t) for t in textos]
    print(f"Páginas cargadas: {len(paginas)}")
    print(f"Fragmentos indexados: {coleccion.count()} (tokens: mín {min(tokens)}, máx {max(tokens)})")
    for (fuente, rol), n in sorted(Counter((f.metadatos['fuente'], f.metadatos['rol']) for f in fragmentos).items()):
        print(f"  [{rol}] {fuente}: {n} fragmentos")


if __name__ == "__main__":
    main()
