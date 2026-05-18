import os
import cv2
import psycopg2
import numpy as np
from deepface import DeepFace

def ajustar_gamma(imagen, gamma=1.0):
    # Construir una tabla de búsqueda (LUT) que mapea los valores de píxeles
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # Aplicar la corrección usando la tabla
    return cv2.LUT(imagen, table)

conn = psycopg2.connect(
    host="localhost",
    user="aidam",
    password="aidam",
    port="5432",
    database="db_faces"
)

cursor = conn.cursor()

# Use OpenCV face detector to get bounding boxes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)
cap.set(cv2.CAP_PROP_BACKLIGHT, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

name = None
res = ...
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            img_gamma = ajustar_gamma(frame, gamma=1.5)
            lab = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2LAB)

            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            cl = clahe.apply(l)

            limg = cv2.merge((cl, a, b))
            img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            face_img = img_clahe[y:y+h, x:x+w]
            try:
                objs = DeepFace.represent(
                    img_path=face_img, 
                    model_name="ArcFace", 
                    enforce_detection=True,
                    align=True
                )
            except Exception as e:
                objs = None

            if objs:
                embedding = objs[0].get('embedding') if isinstance(objs, list) and len(objs) > 0 else None
                if embedding is not None:
                    try:
                        cursor.execute("""
                        SELECT * FROM (
                            SELECT name, embedding <=> %s::vector AS distance
                            FROM users
                        ) a
                        where distance < 0.6
                        ORDER BY distance ASC
                        LIMIT 1;
                        """, (embedding,))
                        res = cursor.fetchone()
                        print(res)
                        if res and res[1] is not None:
                            name = res[0]
                    except Exception as e:
                        conn.rollback()
            if res:
                color = (0, 255, 0)
                label = name
                text_color = (255, 255, 255)
            else:
                color = (150, 150, 150)
                label = "Desconocido"
                text_color = (255, 255, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        cv2.imshow("Prueba", frame)

        if cv2.waitKey(32) == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    cursor.close()
    conn.close()