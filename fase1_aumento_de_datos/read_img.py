# Lectura de una imagen RGB desde un archivo y conversión a una imagen es escala de grises
from PIL import Image
import numpy as np

# leemos la imagen canal por canal
def read_image_as_grayscale(image_path):
    # Abrir la imagen 
    img = Image.open(image_path)

    # Convertir a un arreglo numpy
    img_array = np.array(img)

    img.close()

    # Extraer los canales RGB
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2] 

    # Convertir a escala de grises usando la formula de luminosidad
    # ITU‑R BT.601 
    # Y = 0.2126 R + 0.7152 G + 0.0722 B
    Y = 0.299 * r + 0.587 * g + 0.114 * b

    # Convertir de vuelta a uint8
    Y = Y.astype(np.uint8)
    
    return Y

def read_image_as_rgb(image_path):
    # Abrir la imagen 
    img = Image.open(image_path)

    # Convertir a un arreglo numpy
    img_array = np.array(img)

    img.close()

    return img_array