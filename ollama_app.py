from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import base64
from ollama import chat
import logging

# Importamos las funciones de nuestro nuevo archivo de base de datos
from vector_db import generar_vector_postulante, buscar_talento_vectorial

# --- 1. CONFIGURACIÓN DE LOGGING ---
logger = logging.getLogger(__name__)
logging.basicConfig(filename='response.log', level=logging.INFO)

def log_response(logger: logging.Logger, response_data):
    logger.info(f"Log: {response_data}")

# --- 2. MODELOS PYDANTIC ---
class QABase(BaseModel):
    question: str = Field(..., description="La pregunta del usuario")
    answer: str = Field(..., description="Respuesta sucinta en español latino")

class QAAnalytics(QABase):
    thought: str = Field(..., description="Proceso de reflexión de la respuesta")
    topic: str = Field(..., description="La palabra que mejor describe el tema de la pregunta")

# AQUÍ CORREGIMOS EL ERROR: Solo definimos la clase una vez y usamos "encoded_image"
class QuestionPayload(BaseModel):
    question: str
    encoded_image: str

class BusquedaEmpresa(BaseModel):
    requerimiento: str
    limite: int = 5

# --- 3. FUNCIONES AUXILIAres ---
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ollama_llm_response(question: str, encoded_image: str):
    response = chat(
        model='dolphin-phi', # Tu modelo de lenguaje
        messages=[
            {'role': 'system', 'content': 'Eres un ayudante útil.'},
            {
                'role': 'user', 
                'content': f'Responde esta pregunta: {question}',
                'images': [encoded_image]
            }
        ],
        format=QAAnalytics.model_json_schema()
    )
    return response

# --- 4. APLICACIÓN FASTAPI ---
app = FastAPI(title="Futurework IA API")

# Endpoint 1: Preguntar a la IA sobre una imagen
@app.post("/question", response_model=QABase)
def llm_qa_response(payload: QuestionPayload):
    response = ollama_llm_response(payload.question, payload.encoded_image)
    qa_instance = QAAnalytics.model_validate_json(response.message.content)
    log_response(logger, qa_instance.model_dump())
    return qa_instance

# Endpoint 2: Actualizar el vector de un postulante (Llámalo cuando alguien actualice su CV)
@app.post("/postulantes/{id_postulante}/vectorizar")
def vectorizar_postulante(id_postulante: int):
    exito = generar_vector_postulante(id_postulante)
    if not exito:
        raise HTTPException(status_code=404, detail="Postulante no encontrado o sin datos.")
    return {"mensaje": "Perfil vectorizado correctamente."}

# Endpoint 3: Buscar talento (Para que las empresas lo usen)
@app.post("/empresas/buscar-talento")
def buscar_talento(payload: BusquedaEmpresa):
    candidatos = buscar_talento_vectorial(payload.requerimiento, payload.limite)
    return {"requerimiento": payload.requerimiento, "resultados": candidatos}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ollama_app:app", reload=True)