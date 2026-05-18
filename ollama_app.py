from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import base64
from ollama import chat
import logging
from vector_db import generar_vector_postulante, buscar_talento_vectorial

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(filename='response.log', level=logging.INFO)

def log_response(logger: logging.Logger, response_data):
    logger.info(f"Log: {response_data}")

# --- Modelos Pydantic ---
class QABase(BaseModel):
    question: str
    answer: str

class QAAnalytics(QABase):
    thought: str
    topic: str

class QuestionPayload(BaseModel):
    question: str
    encoded_image: str

class MensajeChat(BaseModel):
    mensaje: str

# --- Funciones Auxiliares ---
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# --- Aplicación FastAPI ---
app = FastAPI(title="Futurework IA API")

# Habilitar CORS para que PHP/JS pueda comunicarse con FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Endpoint 1: Pregunta sobre una imagen
@app.post("/question")
def llm_qa_response(payload: QuestionPayload):
    response = chat(
        model='dolphin-phi', 
        messages=[
            {'role': 'system', 'content': 'Eres un ayudante útil.'},
            {'role': 'user', 'content': payload.question, 'images': [payload.encoded_image]}
        ],
        format=QAAnalytics.model_json_schema()
    )
    qa_instance = QAAnalytics.model_validate_json(response.message.content)
    log_response(logger, qa_instance.model_dump())
    return qa_instance

# Endpoint 2: Vectorizar a un Postulante
@app.post("/postulantes/{id_postulante}/vectorizar")
def vectorizar_postulante(id_postulante: int):
    exito = generar_vector_postulante(id_postulante)
    if not exito:
        raise HTTPException(status_code=404, detail="Postulante no encontrado.")
    return {"mensaje": "Perfil vectorizado correctamente."}

# Endpoint 3: El motor del Chatbot RAG
@app.post("/chat/reclutador")
def chatbot_reclutador(payload: MensajeChat):
    mensaje_limpio = payload.mensaje.lower().strip()

    saludos_comunes = ["hola", "buenas", "buenos dias", "buenas tardes", "buenas noches", "qué tal", "que tal", "saludos", "hi", "hey","hola buenas tardes","hola buenas noches","hola buenos dias"]
    
    # Si el usuario solo saluda, respondemos instantáneamente sin usar la BD ni la IA
    if mensaje_limpio in saludos_comunes:
        return {
            "respuesta": "¡Hola! 👋 Soy el asistente de reclutamiento de Futurework ITT. ¿Qué habilidades o perfil técnico estás buscando hoy en un candidato?",
            "candidatos_crudos": []
        }
        
    # Si el mensaje es muy corto (ej. "a", "ok", "?"), pedimos más contexto
    if len(mensaje_limpio) < 3:
         return {
            "respuesta": "Por favor, dame un poco más de detalle sobre el talento que buscas para poder ayudarte.",
            "candidatos_crudos": []
        }
    
    # 1. Buscar en la base de datos
    candidatos = buscar_talento_vectorial(payload.mensaje, limite=3)
    
    if not candidatos:
        return {"respuesta": "Lo siento, actualmente no tenemos candidatos que coincidan con esa descripción en nuestra plataforma.", "candidatos": []}

    # 2. Armar el contexto para la IA
    contexto = "Candidatos encontrados:\n"
    for c in candidatos:
        contexto += f"- {c['nombreCompleto']} ({c['nombreCarrera']} en {c['ubicacion']}). Match: {c['porcentaje_match']}%\n"

    # 3. Prompt del sistema
    prompt_sistema = f"""
    Eres el asistente inteligente de reclutamiento de Futurework_ITT. 
    Un reclutador pide: "{payload.mensaje}"
    
    Redacta una respuesta amable presentando a los candidatos. Basa tu respuesta SÓLO en esta información:
    {contexto}
    """

    # 4. Obtener respuesta de la IA
    response = chat(
        model='dolphin-phi',
        messages=[
            {'role': 'system', 'content': prompt_sistema},
            {'role': 'user', 'content': 'Por favor, preséntame a los candidatos encontrados.'}
        ]
    )
    
    return {
        "respuesta": response.message.content,
        "candidatos_crudos": candidatos
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ollama_app:app", reload=True)