# rag-seguro

Asistente de preguntas sobre documentación interna que aplica controles de seguridad en cada etapa del pipeline RAG, junto con una herramienta de pruebas ofensivas para auditarlo.

## El problema

Un sistema RAG conectado a los documentos de una empresa hereda sus riesgos. Un empleado puede recibir fragmentos de contratos que no tiene permiso de ver, el modelo puede repetir datos personales contenidos en los documentos y un texto malicioso escondido en un archivo puede alterar el comportamiento del asistente (prompt injection indirecta). La mayoría de implementaciones de ejemplo no contemplan estos casos.

Este proyecto construye el asistente con esos riesgos en mente y después lo ataca para medir qué tan bien resisten los controles.

## Controles de seguridad

| Riesgo | Control |
|---|---|
| Acceso a documentos fuera del rol | Filtro por rol dentro de la consulta a la base vectorial, antes de la búsqueda por similitud. Un fragmento no autorizado nunca llega al contexto del modelo. |
| Exposición de datos personales | Detección y enmascaramiento de cédulas, correos y teléfonos antes de enviar el contexto al modelo. |
| Respuestas sin respaldo | El modelo responde solo a partir de los fragmentos recuperados y cita el documento de origen. |
| Falta de trazabilidad | Registro de auditoría por consulta: usuario, rol, pregunta y documentos utilizados. |
| Fuga de información a terceros | Inferencia local con Ollama. Ni los documentos ni las consultas salen del equipo. |

## Arquitectura

```
Ingesta
  Documentos (PDF, DOCX, Markdown)
    -> fragmentación con metadatos (fuente, roles autorizados)
    -> embeddings -> ChromaDB

Consulta
  Usuario autenticado con rol
    -> embedding de la pregunta
    -> búsqueda en ChromaDB filtrada por rol
    -> enmascaramiento de datos personales
    -> LLM local genera la respuesta con citas
    -> registro de auditoría
```

## Auditoría ofensiva

Un escáner propio ejecuta casos de ataque contra la API, tomando como referencia el OWASP Top 10 for LLM Applications:

- Prompt injection directa e indirecta (instrucciones ocultas dentro de documentos)
- Escalamiento de permisos entre roles
- Extracción de datos personales y del prompt del sistema

Cada hallazgo se documenta, se corrige y se vuelve a medir. Los resultados se publicarán en un informe técnico.

## Decisiones de diseño

**Filtrar antes de recuperar, no después.** Si el filtro de permisos se aplica sobre los resultados ya recuperados, los primeros resultados pueden ser todos no autorizados y el usuario se queda sin respuesta. Además, el control pasa a depender de que ningún paso intermedio exponga lo que se descartó. Filtrar dentro de la consulta elimina ese riesgo desde el origen.

**Diseñado para hardware de consumo.** Todo el sistema corre en una GPU de 4 GB. Se usa un modelo de 3B parámetros (`llama3.2:3b`) con flash attention y caché KV cuantizada a 8 bits para que cargue completo en VRAM.

**Proveedor de LLM intercambiable.** El modelo se selecciona por configuración. Esto permite comparar, con el mismo set de pruebas, la calidad y la resistencia a ataques de un modelo local pequeño frente a uno de mayor tamaño.

## Stack

Python, FastAPI, ChromaDB, sentence-transformers, Ollama, Streamlit, Docker Compose, pytest, GitHub Actions.

## Estado

En desarrollo activo. Hito actual: pipeline RAG básico (ingesta, recuperación y respuesta con citas).

Hoja de ruta:

1. Pipeline RAG con citas y API
2. Control de acceso por rol y enmascaramiento de datos personales
3. Registro de auditoría, interfaz y despliegue con Docker Compose
4. Evaluación de calidad y escáner de seguridad
5. Corrección de hallazgos y medición antes y después

## Autor

Nicholas Pareja Hincapié · [GitHub](https://github.com/nicholasph05)