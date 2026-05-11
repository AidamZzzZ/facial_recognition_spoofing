import cv2
import time
import face_recognition
from deepface import DeepFace
from deepface.modules.exceptions import ImgNotFound


def detectar_rostro_imagen(base_path):
    try: 
        img = cv2.imread(base_path)
        result = DeepFace.extract_faces(img_path=base_path, detector_backend='retinaface', enforce_detection=False)

        for face in result:
            confianza = face['confidence']
            if confianza > 0.6: 
                x = face['facial_area']['x']
                y = face['facial_area']['y']
                w = face['facial_area']['w']
                h = face['facial_area']['h']
                
            
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{confianza:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        imagen_redimensionada = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("resultado: ",imagen_redimensionada)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img
    except cv2.error as e:
        print(f"Error: {e}")
    except ImgNotFound as e:
        print(f"Error: {e}")
        
def detectar_rostro_vivo():
    # inicializando la video camara
    cap = cv2.VideoCapture(0)

    # se crea un objeto CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0)

    # inicializacion de cuadro pasado del frame
    p_time = 0

    # si no se abre la camara web se mostrara un error
    if not cap.isOpened():
        print("Camara no pudo iniciarse")
    
    # se ejecuta la camara web en tiempo real
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # calculando los fps por segundo
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # mostrar los FPS en el live
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # aplicando downscaling a la imagen a 1/4 parte de su tamano
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        lab = cv2.cvtColor(small_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)

        limg = cv2.merge((l_clahe, a, b))

        final_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # el modelo busca los rostros en la imagen
        face_location = face_recognition.face_locations(final_frame, model='arcface')

        # se crea un bounding box que se incrusta en la imagen
        for (top, right, bottom, left) in face_location:
            # escalando coordenadas por el factor inversa a la imagen original
            # top *= 2
            # right *= 2
            # bottom *= 2
            # left *= 2
            cv2.rectangle(final_frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # se muestra la imagen en vivo
        cv2.imshow("Deteccion facial en vivo", frame)
        cv2.imshow("frame con clahe", final_frame)
        # al presionar q se apaga la camara
        if cv2.waitKey(1) == ord('q'):
            break
        
    # se limpia el cache y se quitan los procesos abiertos en segundo plano
    cap.release()
    cv2.destroyAllWindows()
