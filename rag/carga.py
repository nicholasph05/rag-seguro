"""Carga de documentos: Markdown, texto y PDF, con su rol según la carpeta."""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

EXTENSIONES = {".md", ".txt", ".pdf"}


@dataclass
class Pagina:
    texto: str
    fuente: str  # ruta relativa a data/docs, p. ej. "general/manual.md"
    pagina: int
    rol: str


def cargar_documentos(dir_docs: Path) -> list[Pagina]:
    paginas: list[Pagina] = []
    for ruta in sorted(dir_docs.rglob("*")):
        if not ruta.is_file() or ruta.suffix.lower() not in EXTENSIONES:
            continue
        relativa = ruta.relative_to(dir_docs)
        if len(relativa.parts) < 2:
            print(f"[omitido] {relativa}: debe estar dentro de una carpeta de rol")
            continue
        rol, fuente = relativa.parts[0], relativa.as_posix()

        if ruta.suffix.lower() == ".pdf":
            for numero, pagina in enumerate(PdfReader(ruta).pages, start=1):
                texto = pagina.extract_text() or ""
                if texto.strip():
                    paginas.append(Pagina(texto, fuente, numero, rol))
        else:
            paginas.append(Pagina(ruta.read_text(encoding="utf-8"), fuente, 1, rol))
    return paginas
