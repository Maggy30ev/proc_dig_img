from PIL import Image
import numpy as np

def rgb_to_gris(R, G, B):
    # Coeficientes para modo normal
    w0, w1, w2 = 0.85, 0.10, 0.05

    # Calculo de gris
    Y = R * w0 + G * w1 + B * w2

    y_min = Y.min()
    y_range = Y.max() - y_min
    if y_range > 0:
        Y = (Y - y_min) * (255.0 / y_range)

    return Y.astype(np.uint8)

def read_image_by_channels(image_path):
    # Abrir la imagen 
    img = Image.open(image_path)

    # Convertir a un arreglo numpy
    img_array = np.array(img) / 255.0

    img.close()

    img_array = np.clip(img_array, 0, 1)

    # Extraer los canales RGB
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2] 

    return R, G, B

def rgb_a_cmy(R, G, B):
    # Convertir a CMY
    C = 1 - R
    M = 1 - G
    Y = 1 - B

    return C, M, Y

def rgb_a_cmyk(R, G, B):

    K = 1 - np.maximum.reduce([R, G, B])

    # evitar división por cero donde K=1 (negro puro)
    denominador = np.where(K < 1, 1 - K, 1)

    C = (1 - R - K) / denominador
    M = (1 - G - K) / denominador
    Y = (1 - B - K) / denominador

    C = np.where(K < 1, C, 0)
    M = np.where(K < 1, M, 0)
    Y = np.where(K < 1, Y, 0)

    return C, M, Y, K

def rgb_a_hsi(R, G, B):

    eps = 1e-8
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + eps
    theta = np.arccos(np.clip(num / den, -1, 1))

    H = np.where(B <= G, theta, 2*np.pi - theta) / (2*np.pi)
    I = (R + G + B) / 3

    min_rgb = np.minimum.reduce([R, G, B])
    S = np.where(I > 0, 1 - min_rgb / (I + eps), 0)

    return H, S, I