from packaging._parser import Value
import time
from src.tools.clean_screen import limpiar_pantalla
from src.model.model_function import detectar_rostro_imagen, detectar_rostro_vivo

def main():
    # se limpia la pantalla de la terminal antes de comenzar el programa
    limpiar_pantalla()
    while True:
        try:
            # se le pide al usuario ingresar una opcion
            entrada = int(input("""
Seleccione una opcion
1. Analizar imagen local
2. Analizar imagen en vivo
3. Salir
R: """))
            # si la entrada es igual a 1 es para analizar una imagen
            if entrada == 1:
                # se le pide la ruta de la imagen al usuario
                ruta = input("Ingrese la ruta de la imagen: ")
                time.sleep(1)
                print("Analizando imagen...")
                # se llama a la funcion para detectar rostros en una imagen
                img = detectar_rostro_imagen(ruta)
                time.sleep(2)
                # se limpia la pantalla nuevamente
                limpiar_pantalla()
            # si la entrada es igual a 2 es para hacer un envivo
            elif entrada == 2:
                # se llama a la funcion para detectar rostros en vivo
                live = detectar_rostro_vivo()
            elif entrada == 4:
                ...
            elif entrada == 5:
                print("Cerrando programa...")
                break
            # si no la respuesta es incorrecta
            else:
                print("Opcion no valida.")
        except ValueError as e:
            print(f'Error: {e}')
    
if __name__ == '__main__':
    main()