from fastapi import FastAPI
from pydantic import BaseModel, Field
import base64
from ollama import chat
import logging

# 1. Definición de Modelos con Pydantic
class QABase(BaseModel):
    question: str = Field(..., description="La pregunta del usuario")
    answer: str = Field(..., description="Respuesta sucinta en español latino")

class QAAnalytics(QABase):
    thought: str = Field(..., description="Proceso de reflexión de la respuesta")
    topic: str = Field(..., description="La palabra que mejor describe el tema de la pregunta")

class QuestionPayload(BaseModel):
    question: str
    encoded_image: str

# 2. Funciones Auxiliares para Ollama e Imágenes
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ollama_llm_response(question: str, encoded_image: str):
    # Se utiliza el modelo descargado (ej. gemma3, llama3.2, etc.)
    response = chat(
        model='dolphin-phi', # Reemplaza con el nombre del modelo que hayas descargado
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

# 3. Configuración de Logging (Registro de actividad)
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='response.log',
    level=logging.INFO
)

def log_response(logger: logging.Logger, response: QAAnalytics):
    logger.info(f"Question: {response.question}")
    logger.info(f"Answer: {response.answer}")
    logger.info(f"Thought: {response.thought}")
    logger.info(f"Topic: {response.topic}")

# 4. Configuración de FastAPI y Endpoints
app = FastAPI()

@app.post("/question", response_model=QABase)
def llm_qa_response(payload: QuestionPayload):
    # Obtener respuesta del modelo LLM
    response = ollama_llm_response(payload.question, payload.encoded_image)
    
    # Validar y parsear la respuesta JSON usando el modelo de Pydantic
    qa_instance = QAAnalytics.model_validate_json(response.message.content)
    
    # Guardar en el log
    log_response(logger, qa_instance)
    
    # Se retorna al usuario solo la pregunta y respuesta (QABase)
    return qa_instance

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ollama_app:app", reload=True)