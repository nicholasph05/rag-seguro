"""LLM intercambiable: el resto del código solo llama a generar()."""
from rag.config import LLM_MODELO, LLM_NUM_CTX, LLM_PROVIDER, LLM_TEMPERATURA


def generar(sistema: str, usuario: str) -> str:
    if LLM_PROVIDER == "ollama":
        import ollama

        respuesta = ollama.chat(
            model=LLM_MODELO,
            messages=[
                {"role": "system", "content": sistema},
                {"role": "user", "content": usuario},
            ],
            options={"temperature": LLM_TEMPERATURA, "num_ctx": LLM_NUM_CTX},
        )
        return respuesta["message"]["content"]
    raise ValueError(f"LLM_PROVIDER no soportado: {LLM_PROVIDER}")
