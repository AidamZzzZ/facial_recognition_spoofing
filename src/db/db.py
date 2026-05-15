import os
import cv2
import numpy as np
import psycopg2
from deepface import DeepFace
from dotenv import load_dotenv

load_dotenv()

DATABASE=os.getenv('DATABASE')
USER=os.getenv('USER_DB')
PASSWORD=os.getenv('PASSWORD_DB')
HOST=os.getenv('HOST_DB')
PORT=os.getenv('PORT_DB')

def crear_connect_db():
    try:
        conn = psycopg2.connect(
            database=DATABASE,
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT
        )

        cursor = conn.cursor()
        # 3. Ejecutar comando CREATE TABLE
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            embedding vector(512)  
        )
        """
        cursor.execute(create_table_sql)
        
        # 4. Confirmar cambios
        conn.commit()
        print("Conexion exitosa!.")
        return conn
    except Exception as e:
        print(f"Error: {e}")


def registrar_usuario_db(name, img):
    conn = crear_connect_db()

    try:
        obj = DeepFace.represent(img_path=img, model_name="ArcFace", enforce_detection=True)
        embedding = obj[0]["embedding"]
        
        cursor = conn.cursor()


        cursor.execute(
            "INSERT INTO users (name, embedding) VALUES (%s, %s)",
            (name, embedding)
        )
        print(f"Usuario: {name} agregado correctamente.")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")


def buscar_por_imagen(img, enforce_detection=True):
    print("\n--- BÚSQUEDA DE USUARIO POR FOTOGRAFÍA ---")
    
    print("Analizando el rostro en la imagen...")
    try:
        # Si la imagen viene de OpenCV, convertirla a RGB
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # obtiene el embedding de la imagen
        objs = DeepFace.represent(
            img_path=img,
            model_name="ArcFace",
            enforce_detection=enforce_detection,
            detector_backend='opencv'
        )
        vector_busqueda = objs[0]["embedding"]
    except Exception as e:
        print(f"Error: No se pudo detectar un rostro claro en la imagen proporcionada. {e}")
        return None

    # 3. Conectar a PostgreSQL y buscar el más similar
    try:
        # se conecta a la base de datos
        conn = crear_connect_db()
        cur = conn.cursor()

        # Usamos la distancia Euclidiana entre embeddings (<=> en pgvector)
        query = """
            SELECT name, embedding <=> %s::vector AS distancia
            FROM users
            ORDER BY embedding <=> %s::vector
            LIMIT 1;
        """
        cur.execute(query, (vector_busqueda, vector_busqueda))
        resultados = cur.fetchall()

        UMBRAL_DISTANCIA = 0.30

        if resultados:
            nombre_match, distancia = resultados[0]

            if distancia < UMBRAL_DISTANCIA:
                    print(nombre_match)
                    return nombre_match
                
            else:
                print("Rostro desconocido.")
                return None
        else:
            print("La base de datos está vacía.")
            return None

    except Exception as e:
        print(f"Error de base de datos: {e}")
        return None
    finally:
        if 'conn' in locals() and conn:
            cur.close()
            conn.close()
