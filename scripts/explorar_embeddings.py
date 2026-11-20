"""Experimento: cómo los embeddings capturan significado (no palabras)."""
from sentence_transformers import SentenceTransformer, util

MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

pregunta = "¿Cuántos días de vacaciones tengo al año?"
fragmentos = [
    "Los empleados disponen de 15 días hábiles de descanso remunerado por cada año trabajado.",
    "Las vacaciones del gerente en Cartagena fueron todo un éxito.",
    "El horario de atención de la oficina es de 8:00 a. m. a 5:00 p. m.",
    "Employees are entitled to 15 working days of paid leave per year.",
    "La contraseña del correo corporativo debe cambiarse cada 90 días.",
]

modelo = SentenceTransformer(MODELO)

emb_pregunta = modelo.encode(pregunta)
emb_fragmentos = modelo.encode(fragmentos)

print(f"Dimensiones del vector: {len(emb_pregunta)}")
print(f"Máximo de tokens por entrada: {modelo.max_seq_length}")
print(f"Primeros 5 números de la pregunta: {emb_pregunta[:5].round(3)}\n")

# Similitud coseno: 1 = mismo significado, 0 = sin relación
similitudes = util.cos_sim(emb_pregunta, emb_fragmentos)[0].tolist()
ranking = sorted(zip(fragmentos, similitudes), key=lambda x: x[1], reverse=True)

print(f"Pregunta: {pregunta}\n")
for texto, sim in ranking:
    print(f"{sim:.3f}  {texto}")

# Truncamiento: lo que pase del límite se descarta sin aviso
texto_largo = " ".join(fragmentos * 10)
n_tokens = len(modelo.tokenizer(texto_largo)["input_ids"])
print(f"\nTexto largo: {n_tokens} tokens -> el modelo solo usa los primeros {modelo.max_seq_length}")