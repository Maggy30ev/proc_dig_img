# Lectura de una imagen RGB desde un archivo y conversión a una imagen es escala de grises
from PIL import Image
import numpy as np

# leemos la imagen canal por canal
def read_image_as_grayscale(image_path, mode='normal'):
    # Abrir la imagen 
    img = Image.open(image_path)

    # Convertir a un arreglo numpy
    img_array = np.array(img)

    img.close()

    # Extraer los canales RGB
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    # Coeficientes por modo
    weights_map = {
        "normal": np.array([0.299, 0.587, 0.114], dtype=np.float16),
        "custom": np.array([0.85, 0.10, 0.05], dtype=np.float16),
    }
    
    if mode not in weights_map:
        raise ValueError(f"Modo no soportado: {mode}. Usa 'normal' o 'custom'.")
    
    weights = weights_map[mode]
    
    # Calculo vectorizado de gris en una sola operacion
    # (equivalente a r*w0 + g*w1 + b*w2)
    Y = np.tensordot(img_array[:, :, :3].astype(np.float16), weights, axes=([-1], [0]))
    
    if mode == "custom":
        y_min = Y.min()
        y_range = Y.max() - y_min
        if y_range > 0:
            Y = (Y - y_min) * (255.0 / y_range)
    
    # Y = np.clip(Y, 0, 255).astype(np.uint8)
    
    return Y.astype(np.uint8)

def read_image_as_rgb(image_path):
    # Abrir la imagen 
    img = Image.open(image_path)

    # Convertir a un arreglo numpy
    img_array = np.array(img)

    img.close()

    return img_array