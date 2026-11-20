"""Fragmentación por oraciones, limitada por tokens reales del modelo de embeddings."""
import re
from dataclasses import dataclass

from rag.carga import Pagina
from rag.config import MAX_TOKENS_FRAGMENTO, SOLAPE_ORACIONES
from rag.embeddings import contar_tokens

# Corta tras . ! ? solo si sigue mayúscula, número o signo de apertura: así "8:00 a. m." no se parte
_FIN_ORACION = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡#\d])")


@dataclass
class Fragmento:
    id: str
    texto: str
    metadatos: dict


def _oraciones(texto: str) -> list[str]:
    limpio = " ".join(texto.split())  # une saltos de línea y espacios repetidos
    return [o for o in _FIN_ORACION.split(limpio) if o]


def _partir_si_excede(oracion: str, max_tokens: int) -> list[str]:
    """Una oración que sola supera el límite se corta por palabras."""
    if contar_tokens(oracion) <= max_tokens:
        return [oracion]
    piezas, actual = [], []
    for palabra in oracion.split():
        if actual and contar_tokens(" ".join(actual + [palabra])) > max_tokens:
            piezas.append(" ".join(actual))
            actual = []
        actual.append(palabra)
    if actual:
        piezas.append(" ".join(actual))
    return piezas


def _agrupar(oraciones: list[str], max_tokens: int, solape: int) -> list[str]:
    grupos: list[str] = []
    actual: list[str] = []
    for oracion in oraciones:
        for pieza in _partir_si_excede(oracion, max_tokens):
            if actual and contar_tokens(" ".join(actual + [pieza])) > max_tokens:
                grupos.append(" ".join(actual))
                actual = actual[-solape:] if solape else []
                if actual and contar_tokens(" ".join(actual + [pieza])) > max_tokens:
                    actual = []
            actual.append(pieza)
    if actual:
        grupos.append(" ".join(actual))
    return grupos


def fragmentar(paginas: list[Pagina]) -> list[Fragmento]:
    """Fragmenta cada página por separado para que la cita de página sea exacta."""
    fragmentos: list[Fragmento] = []
    for pag in paginas:
        textos = _agrupar(_oraciones(pag.texto), MAX_TOKENS_FRAGMENTO, SOLAPE_ORACIONES)
        for i, texto in enumerate(textos):
            fragmentos.append(Fragmento(
                id=f"{pag.fuente}#p{pag.pagina}-f{i}",
                texto=texto,
                metadatos={"fuente": pag.fuente, "pagina": pag.pagina, "rol": pag.rol, "fragmento": i},
            ))
    return fragmentos
