import psycopg2
import numpy as np
from deepface import DeepFace

def crear_connect_db():
    try:
        conn = psycopg2.connect(
            database='db_faces',
            user='aidam',
            password='aidam',
            host='localhost',
            port='5432'
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
        obj = DeepFace.represent(img_path=img, model_name="ArcFace")
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


def buscar_por_imagen(ruta_imagen):
    import os
    print("\n--- BÚSQUEDA DE USUARIO POR FOTOGRAFÍA ---")

    # 2. Extraer el embedding de la imagen
    print("Analizando el rostro en la imagen...")
    try:
        objs = DeepFace.represent(
            img_path=ruta_imagen, 
            model_name="ArcFace", 
            enforce_detection=True
        )
        vector_busqueda = objs[0]["embedding"]
    except Exception as e:
        print("Error: No se pudo detectar un rostro claro en la imagen proporcionada.")
        return

    # 3. Conectar a PostgreSQL y buscar el más similar
    try:
        conn = crear_connect_db()
        cur = conn.cursor()

        # Usamos la distancia Coseno (<=>)
        query = """
            SELECT name, embedding <=> %s::vector AS distancia
            FROM users
            ORDER BY embedding <=> %s::vector
            LIMIT 1;
        """
        cur.execute(query, (vector_busqueda, vector_busqueda))
        resultado = cur.fetchone()

        # 4. Evaluar el resultado con un umbral
        UMBRAL_DISTANCIA = 0.40

        if resultado:
            nombre_match, distancia = resultado
            
            print("\n--- RESULTADO DE LA BÚSQUEDA ---")
            print(f"Mejor coincidencia encontrada: {nombre_match}")
            print(f"Distancia Coseno: {distancia:.4f}")

            if distancia < UMBRAL_DISTANCIA:
                # Si es menor al umbral, es la misma persona
                print(f"✅ ¡Identidad confirmada! El rostro pertenece a {nombre_match}.")
            else:
                # Si el rostro más parecido sigue estando muy "lejos" matemáticamente
                print("❌ Rostro desconocido. Se parece un poco, pero la distancia es muy alta para confirmar.")
        else:
            print("La base de datos está vacía.")

    except Exception as e:
        print(f"Error de base de datos: {e}")
    finally:
        if 'conn' in locals() and conn:
            cur.close()
            conn.close()

new_user = buscar_por_imagen("/home/aidam/Desktop/facial_recognition_spoofing/images/imagen.jpg")
print(new_user)