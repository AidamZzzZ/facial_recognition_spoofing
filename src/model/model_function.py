import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import cv2
import time
from deepface import DeepFace
from deepface.modules.exceptions import ImgNotFound
import os
from deepface_antispoofing import DeepFaceAntiSpoofing

# funcion para detectar rostros en imagenes
def detectar_rostro_imagen(base_path):
    try: 
        # se lee la imagen desde la ruta
        img = cv2.imread(base_path)
        # se extraen los rostros con sus coordenadas
        result = DeepFace.extract_faces(img_path=base_path, detector_backend='retinaface', enforce_detection=False)

        # se iteran sobre todos los rostros
        for face in result:
            # si la confianza del modelo es mayor a 0.6 este mostrara el rostro
            confianza = face['confidence']
            if confianza > 0.6: 
                # se toman las coordenadas en los distintos ejes
                x = face['facial_area']['x']
                y = face['facial_area']['y']
                w = face['facial_area']['w']
                h = face['facial_area']['h']
                
                # se crea el rectangulo con los parametros en la imagen
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # se agrega una etiqueta con la cual muestra el nivel de confianza del modelo
                cv2.putText(img, f"{confianza:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        # se redimensiona la imagen para mostrarla al usuario
        imagen_redimensionada = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("resultado: ",imagen_redimensionada)
        # cuando el usuario presione 0 se cerrara la ventana
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img
    # excepcion de errores
    except cv2.error as e:
        print(f"Error: {e}")
    except ImgNotFound as e:
        print(f"Error: {e}")

# funcion para detectar rostros en vivo
def detectar_rostro_vivo():
    TEMP_FOLDER = "images/"
    # inicializando la video camara
    cap_h1 = cv2.VideoCapture(0)

    # se crea un objeto CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0)

    # inicializacion de cuadro pasado del frame
    p_time = 0

    # inicializando objeto de antispoofing
    # deepface = DeepFaceAntiSpoofing()

    # si no se abre la camara web se mostrara un error
    if not cap_h1.isOpened():
        print("Camara no pudo iniciarse")
    
    # se ejecuta la camara web en tiempo real
    while cap_h1.isOpened():
        ret, frame = cap_h1.read()
        
        if not ret:
            break
        
        # calculando los fps por segundo
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # mostrar los FPS en el live
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #deepface = DeepFaceAntiSpoofing()
        #temp_path = os.path.join(TEMP_FOLDER, "capture.jpg")
        #cv2.imwrite(temp_path, frame)

        #temp_pathresponse = deepface.analyze_deepface(temp_path)
        #print(response)

        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend="opencv", anti_spoofing=True)
        for result in results:
            x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']        # aplicando downscaling a la imagen a 1/4 parte de su tamano
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Deteccion en tiempo real ", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    # se limpia el cache y se quitan los procesos abiertos en segundo plano
    cap_h1.release()
    cv2.destroyAllWindows()

        