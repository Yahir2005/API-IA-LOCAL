import mariadb
import ollama
import sys

# Configuración de tu base de datos (¡Cámbiala con tus datos reales!)
DB_CONFIG = {
    "user": "root",
    "password": "tu_password",
    "host": "localhost",
    "port": 3306,
    "database": "Futurework_ITT"
}

def get_connection():
    try:
        return mariadb.connect(**DB_CONFIG)
    except mariadb.Error as e:
        print(f"Error conectando a MariaDB: {e}")
        sys.exit(1)

def generar_vector_postulante(id_postulante: int):
    """Genera el perfil semántico del postulante y lo guarda como vector."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    # 1. Obtener la información del postulante uniendo las tablas
    query_info = """
        SELECT c.nombreCarrera, p.ubicacion,
               GROUP_CONCAT(DISTINCT h.nombreHabilidad SEPARATOR ', ') AS habilidades,
               GROUP_CONCAT(DISTINCT cert.nombre SEPARATOR ', ') AS certificaciones
        FROM Postulante p
        JOIN Carrera c ON p.Carrera_idCarrera = c.idCarrera
        LEFT JOIN Postulante_Habilidades ph ON p.idPostulante = ph.Postulante_idPostulante
        LEFT JOIN Habilidades h ON ph.Habilidades_idHabilidad = h.idHabilidad
        LEFT JOIN Postulante_Certificacion pc ON p.idPostulante = pc.Postulante_idPostulante
        LEFT JOIN Certificaciones cert ON pc.Certificaciones_idCertificacion = cert.idCertificacion
        WHERE p.idPostulante = ?
        GROUP BY p.idPostulante;
    """
    cursor.execute(query_info, (id_postulante,))
    datos = cursor.fetchone()
    
    if not datos:
        conn.close()
        return False
        
    # 2. Armar el texto descriptivo
    habilidades = datos['habilidades'] if datos['habilidades'] else "Ninguna"
    certificaciones = datos['certificaciones'] if datos['certificaciones'] else "Ninguna"
    
    texto_perfil = f"Profesional en {datos['nombreCarrera']}. Ubicado en {datos['ubicacion']}. " \
                   f"Habilidades: {habilidades}. Certificaciones: {certificaciones}."
                   
    # 3. Generar el vector con el modelo nomic-embed-text
    respuesta = ollama.embeddings(model="nomic-embed-text", prompt=texto_perfil)
    vector_str = str(respuesta["embedding"])
    
    # 4. Guardar el vector
    query_update = "UPDATE Postulante SET perfil_vector = VEC_FromText(?) WHERE idPostulante = ?"
    cursor.execute(query_update, (vector_str, id_postulante))
    conn.commit()
    conn.close()
    return True

def buscar_talento_vectorial(requerimiento_empresa: str, limite: int = 5):
    """Busca los postulantes que mejor hagan match con lo que pide la empresa."""
    # Convertimos la búsqueda de la empresa en vector
    respuesta = ollama.embeddings(model="nomic-embed-text", prompt=requerimiento_empresa)
    vector_busqueda = str(respuesta["embedding"])
    
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Buscamos usando VECTOR_DISTANCE
    query_busqueda = """
        SELECT 
            p.idPostulante, 
            u.nombreCompleto, 
            c.nombreCarrera, 
            p.ubicacion,
            VECTOR_DISTANCE(p.perfil_vector, VEC_FromText(?)) AS afinidad
        FROM Postulante p
        JOIN Usuarios u ON p.Usuarios_idUsuarios = u.idUsuarios
        JOIN Carrera c ON p.Carrera_idCarrera = c.idCarrera
        WHERE p.perfil_vector IS NOT NULL
        ORDER BY afinidad ASC
        LIMIT ?
    """
    cursor.execute(query_busqueda, (vector_busqueda, limite))
    candidatos = cursor.fetchall()
    conn.close()
    
    return candidatos