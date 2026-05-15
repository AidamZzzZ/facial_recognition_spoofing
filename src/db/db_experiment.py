import os
import cv2
import psycopg2
from deepface import DeepFace

conn = psycopg2.connect(
    host="localhost",
    user="postgres",
    password="postgres",
    port="5432",
    database="db_faces"
)

cursor = conn.cursor()

# Use OpenCV face detector to get bounding boxes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)

name = None
res = None
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            try:
                objs = DeepFace.represent(img_path=face_img, model_name="ArcFace", enforce_detection=True)
            except Exception as e:
                objs = None

            if objs:
                embedding = objs[0].get('embedding') if isinstance(objs, list) and len(objs) > 0 else None
                if embedding is not None:
                    try:
                        cursor.execute("""
                        SELECT name, embedding <-> %s::vector AS distance
                        FROM users
                        LIMIT 1
                        """, (embedding,))
                        res = cursor.fetchone()
                        if res and res[1] is not None:
                            name = res[0]
                    except Exception as e:
                        conn.rollback()
                        print(f"DB error: {e}")

            if res[1] < 4.5:
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