"""Control de acceso: denegar por defecto; solo se recupera lo que el rol permite."""
import json

from rag.config import ARCHIVO_USUARIOS, PERMISOS_POR_ROL


class AccesoDenegado(Exception):
    pass


def roles_documento_permitidos(usuario: str) -> list[str]:
    usuarios = json.loads(ARCHIVO_USUARIOS.read_text(encoding="utf-8"))
    if usuario not in usuarios:
        raise AccesoDenegado(f"Usuario desconocido: {usuario}")
    rol = usuarios[usuario]["rol"]
    permitidos = PERMISOS_POR_ROL.get(rol, [])
    if not permitidos:
        raise AccesoDenegado(f"El rol '{rol}' no tiene permisos de lectura")
    return permitidos
