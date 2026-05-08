import cv2
from src.tools.clean_screen import limpiar_pantalla
from src.model.model_function import detectar_rostro_imagen

def main():
    limpiar_pantalla()
    entrada = int(input("""
Seleccione una opcion
1. Analizar imagen local
2. Analizar imagen en vivo
R: """))
    if entrada == 1:
        ruta = input("Ingrese la ruta de la imagen: ")
        print("Analizando imagen...")
        img = detectar_rostro_imagen(ruta)
        limpiar_pantalla()
    else:
        print("Opcion no valida.")
    
if __name__ == '__main__':
    main()