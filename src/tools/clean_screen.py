import os 

# funcion para limpiar pantalla en la terminal 
def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')