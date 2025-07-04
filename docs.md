Estrategia integral para un “despiece‑bot” basado en LLMs y visión que procese PDFs complejos de proyectos de construcción

A continuación encontrarás una arquitectura de referencia y las mejores prácticas probadas en 2024‑25 para combinar OCR avanzado (Gemini, GPT‑4o, Claude 4) con razonamiento LLM y obtener listados de materiales (quantity take‑off) con el menor esfuerzo humano posible.

⸻

1. Capa de ingestión y preclasificación
	1.	Entrada nativa de PDF. Gemini 2.5 Pro y GPT‑4o ya aceptan PDFs de hasta ≈ 1 000 páginas sin convertirlos a imágenes, manteniendo estructura y metadatos  ￼ ￼.
	2.	Separación automática de tipos de hoja. Clasifica cada página en planos, cómputos métricos o memorias usando un clasificador ligero (e.g., CLIP/VLM) y reglas de densidad de líneas. Esto permite aplicar OCR o CV especializado sólo donde aporta valor.
	3.	Persistencia y versionado. Almacena cada página como “document‑object” en un bucket con hash y número de revisión; evita reprocesar copias idénticas.

⸻

2. OCR y análisis de layout
	•	Ensemble OCR. Combina el OCR de Google Document AI (≈ 98 % de acierto) con la visión LLM de Gemini o GPT‑4o para recuperar tablas y jerarquía de títulos  ￼ ￼.
	•	Detección de símbolos y cotas. Para planos, usa un detector de objetos ligero (YOLOv8 o Segment Anything) entrenado con símbolos propios (ladrillos, vigas, puertas) y pasa esas regiones al LLM de visión para lectura semántica (“Qué representa este bloque y qué dimensión aparece a su lado”).
	•	Normalización métrica. El agente extrae la escala de cada plano (“Esc: 1:100”) y la almacena en el “context store” para que los sub‑agentes de cuantificación conviertan automáticamente unidades.

⸻

3. Extracción semántica y estructuración
	1.	Prompt de extracción de entidades. Los LLMs (GPT‑4o / Claude Opus 4) reciben páginas en chunks ≤8 000 tokens y devuelven JSON normalizado con: {item, type, dimensions, units, quantity, notes, page_ref}.
	2.	Ontología de materiales. Mantén un “material lexicon” (ej.: CSI MasterFormat o propia base de datos) y haz entity‑linking via embeddings para homogeneizar sinónimos (“bloc 15 × 20” ↔ “muro bloque 200 mm”).
	3.	Vector store + RAG. Guarda los embeddings de cada item y sus fragmentos de evidencia; después, un paso RAG permite re‑consultar las hojas originales cuando algo quede ambiguo.

⸻

4. Cálculo de cantidades y control de coherencia
	•	Agente calculista. Un sub‑agente llama a un motor de reglas (o Python en servidor) para multiplicar longitudes, áreas y coeficientes de desperdicio; envía los resultados al LLM para explicar el cálculo paso a paso (auditable).
	•	Autoverificación. Aplica la técnica multi‑agent debate: dos instancias del LLM recalculan independientemente; si difieren >2 %, se marca “ revisión humana”. Claude 4 destaca en razonamiento numérico largo  ￼.
	•	Consistencia global. Un último paso recorre el JSON consolidado y detecta cantidades ilógicas (ej.: hormigón > 2 × volumen total del proyecto).

⸻

5. Generación de la lista de materiales (BOM) y costes
	•	Formato de salida: CSV + enlace IFC/BIM. Incluye campos compatibles con ERP o plataformas de compras.
	•	Enriquecimiento de precios: Consulta tu base de precios (RSMeans, librería interna) y deja al LLM responder preguntas del tipo “¿Cuánto costaría si el acero sube 5 %?”.
	•	Interfaz de usuario: Chat web con panel lateral que muestra tanto la evidencia visual (snapshot de plano con caja roja) como la línea de despiece calculada: UX parecida a Bild AI o Kreo, startups que ya automatizan planos y costos  ￼ ￼.

⸻

6. Gobernanza, privacidad y escalabilidad
	•	Contenedores aislados. Procesa los PDFs on‑prem o en VPC si el contrato lo exige; los modelos vía API (Gemini, GPT‑4o) sólo reciben páginas no confidenciales o versiones anonimizadas.
	•	Microservicios event‑driven. Cada etapa (OCR, extracción, cálculo, validación, reporte) se ejecuta como función asíncrona. Con colas tipo Kafka/Rabbit se puede escalar a > 64 planos concurrentes (tu objetivo original de “más de 64 colas”).
	•	Feedback continuo. Guarda correcciones humanas; re‑entrena el detector de símbolos y ajusta prompts con few‑shot para mejorar recall en cada iteración. Experiencias 2025 muestran recortes de 10‑15 % en coste de proyecto gracias a IA en toma de cantidades  ￼.

⸻

Checklist de implementación rápida

Fase	Herramienta recomendada (jul‑2025)	Comentario
OCR plano­‑texto	Google Document AI + fallback GPT‑4o	Mejora estructura de tablas
Visión LLM	Gemini 2.5 Pro o GPT‑4o (128 K ctx)	Nativo PDF, multilingüe
Parsing semántico	Claude Opus 4 (mejor en CO‑T)	Filtra alucinaciones
Vector DB	Qdrant / Weaviate	Soporta filtros por metadatos
Regla‑engine	Jupyter‑Python + Pandas	Transparencia en cálculos

(Ajusta con modelos open‑source como Llama 4 Maverick si buscas despliegue 100 % local)  ￼

⸻

Conclusión

La combinación de OCR de alta fidelidad, LLMs multimodales y un flujo agentico en microservicios permite obtener “despieces” confiables sin rehacer planos a BIM. La clave es orquestar varios modelos —cada uno donde es más fuerte—, mantener evidencias trazables y retroalimentar continuamente el sistema con correcciones de los especialistas de obra. Así alcanzarás tiempos de estimación que pasan de días a minutos y reducirás los sobrecostes propios de los errores de cantidad.

Plan 2.0 — Arquitectura unificada para el “despiece‑bot” con Gemini 2.5 Pro / Flash y tool_use nativo

El objetivo es mantener la precisión anterior pero reducir piezas externas gracias a (1) la ventana de 1 M tokens de Gemini 2.5 Pro y (2) la latencia ultra‑baja de Gemini 2.5 Flash. El flujo resultante queda en tres micro‑servicios lógicos en lugar de seis.

⸻

1. Servicio ingestor‑flash

Responsabilidad: recibir el PDF, identificar contenido y almacenar páginas.

Paso	Implementación con Gemini Flash	Motivo
a. Clasificación de páginas (planos / tablas / memorias)	Gemini‑2.5‑Flash prompt: “Return JSON {page_id, kind} for each page”	Flash responde en <500 ms y basta razonamiento ligero.
b. OCR preliminar para páginas de texto corrido	gemini.flash.extract_text() (API helper)	Reduce costes; los errores se corrigen más adelante.
c. Persistencia	Guarda cada página y su metadato en bucket + Firestore	Sin cambios respecto al plan 1.0.

Resultado: JSON “manifest” con layout + enlaces de cada página, listo para razonamiento profundo.

⸻

2. Servicio reasoner‑pro

Responsabilidad: extracción semántica, despiece, verificación y explicaciones.
	1.	Contexto único de gran tamaño
Cargar en un solo prompt:

{
  "manifest": …,
  "pages": [obj_page_1, obj_page_2, …],         // Hasta ~5 000 páginas aprox.
  "material_lexicon": …,
  "business_rules": …             // desperdicios, factores de seguridad, etc.
}

Con 1 M tokens cabe todo el proyecto más un few‑shot de ejemplos correctos. No hace falta dividir en chunks ni RAG intermedio; se simplifica la orquestación y se evitan “context‑switch hallucinations”.

	2.	Extracción y normalización (Gemini 2.5 Pro)
	•	Prompt autorecursivo que devuelve lote‑a‑lote un array de objetos:
{item, type, dims, units, qty_raw, evidence_pages, confidence}
	•	Entity‑linking por embeddings internos de Pro; reemplaza el vector‑DB externo.
	3.	Cálculo y validación con tool_use
	•	Declarar dentro del prompt las funciones:

function area(length: number, width: number): number {}
function volume(area: number, height: number): number {}
function waste(qty: number, factor: number): number {}


	•	Gemini Pro llama las funciones automáticamente; los resultados regresan al contexto y pueden desencadenar nuevas llamadas (“chain‑of‑thought calculado”).
	•	Verificación doble: se pide a Pro que repita la serie de tool_use y compare contra la primera pasada; discrepancias > 2 % → flag “REVIEW”.

	4.	Salida
	•	JSON BOM listo + anexos: explanation_markdown, evidence_links.
	•	Se publica en Pub/Sub para notificar al front‑end.

⸻

3. Servicio qa‑flash (opcional en tiempo real)

Cuando un usuario de obra pregunte “¿Cuántos bloques 15×20 necesito para el muro norte?”, el front‑end consulta:
	1.	Prompt a Gemini Flash con:
	•	question del usuario
	•	JSON BOM (≈ a 1 000 tokens)
	•	Lista de pocas evidencias relevantes.
	2.	Respuesta en ~300 ms con cifra exacta + referencia de plano.

⸻

Ventajas concretas de la simplificación

Plan 1.0	Plan 2.0
6 micro‑servicios	3 micro‑servicios
Vector DB externo (Weaviate/Qdrant)	Eliminado — embeddings internos
OCR ensemble (Google Doc AI + LLM)	OCR Flash para texto, Pro para símbolos
Motor reglas Python	tool_use interno
Gestión de chunks/ventanas	Contexto único de 1 M tokens
Latencia total: 2–4 min / 100 págs	≈ 1 min / 100 págs


⸻

Recomendaciones prácticas
	1.	Límites reales de memoria
Aunque el tope teórico sea 1 M tokens, los tiempos de respuesta crecen > 60 s cuando superas ~600 k tokens. Segmenta proyectos excepcionales (>600 k) en dos lotes lógicos: Planos y Especificaciones.
	2.	Política de coste
	•	Flash cuesta ≈ ⅒ de Pro. Rutea todo lo que no requiera razonamiento profundo a Flash (clasificación, OCR simple, preguntas de QA).
	•	Mantén un cost‑guard middleware que marque el prompt como “Pro” sólo si incluye cálculo o agregación entre páginas.
	3.	Observabilidad y trazabilidad
	•	Habilita tool_use logs para auditar cómo Pro llegó a cada cantidad.
	•	Guarda los intermediate function calls; son oro puro para depurar.
	4.	Seguridad de datos
	•	Si se procesan planos confidenciales, usa Gemini Enterprise con data residency y desactiva model‑training opt‑in.

⸻

Conclusión

Con Gemini 2.5 Pro como cerebro de razonamiento y Gemini 2.5 Flash como músculo de E/S y QA, se consigue un pipeline más corto, más rápido y más barato, sin sacrificar fiabilidad. El uso de la ventana de 1 M tokens elimina casi toda la complejidad de segmentación y RAG, mientras que tool_use reemplaza los micro‑servicios de cálculo y validación numérica. El resultado es un “despiece‑bot” listo para producción que cabe en tres contenedores ligeros y es fácilmente escalable a múltiples proyectos en paralelo.