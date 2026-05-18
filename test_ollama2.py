from fastapi.testclient import TestClient
from ollama_app2 import app, encode_image_to_base64

# Crear cliente de pruebas
client = TestClient(app)

def test_task_question_with_image(question: str, image_path: str):
    # Codificar la imagen a base64
    encoded_image = encode_image_to_base64(image_path)
    
    # Crear payload para la solicitud
    payload = {
        "question": question,
        "encoded_image": encoded_image
    }
    
    # Realizar la solicitud POST al endpoint
    url = "/question"
    response = client.post(url, json=payload)

    # Verificar que la respuesta sea exitosa
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    print("Response JSON:", response.json())

if __name__ == "__main__":
    question = "¿Qué contiene esta imagen?"  
    image_path = "xp.jpeg" 
    test_task_question_with_image(question, image_path)