'''Este archivo contiene una serie de funciones (implementadas desde cero) para aumentar datos de 
   imágenes utilizando varias técnicas. 

    Las funciones incluyen las siguientes tecnicas:

        a. Volteado (flipping)
        b. Rotación aplicando interpolación bilineal para definir la intensidad de la imagen resultante.
        c. Traslación
        d. Escalamiento aplicando interpolación bilineal para definir la intensidad de la imagen resultante.
        e. Borrado aleatorio (Random erase)
        f. Mezclado de regiones (cutmix)

'''
import numpy as np
from typing import Optional
import random

# Clase para aumentar datos de imágenes
class DataAugmentation:

    def __init__(self, seed: Optional[int] = None):
        """
        Inicializa el generador de aumento de datos.
        
        Args:
            seed: Semilla para reproducibilidad de resultados aleatorios
        """
        if seed is not None:
            np.random.seed(seed)

    # Interpolación bilineal vectorizada (función auxiliar)
    @staticmethod
    def _bilinear_interpolation(image, x_coords, y_coords):
        """
        Realiza interpolación bilineal vectorizada para un conjunto de coordenadas.
        
        Fórmula:
            I(x,y) = (1-a_h)(1-a_v)·I(x0,y0) + a_h(1-a_v)·I(x1,y0) 
                   + (1-a_h)a_v·I(x0,y1) + a_h·a_v·I(x1,y1)
        
        donde a_h = x - x0 y a_v = y - y0
        
        Args:
            image: Imagen de entrada como array.
            x_coords: Array de coordenadas x.
            y_coords: Array de coordenadas y.
        
        Returns:
            tuple: (valores_interpolados, mascara_validos)
                - valores_interpolados: Array con los valores interpolados.
                - mascara_validos: Array booleano indicando píxeles válidos.
        """
        orig_h, orig_w = image.shape
        
        # Coordenadas de los 4 píxeles vecinos (esquinas del cuadrado)
        x0 = np.floor(x_coords).astype(int)
        y0 = np.floor(y_coords).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Crear máscara para píxeles válidos (dentro de los límites)
        valid_mask = (x0 >= 0) & (x1 < orig_w) & (y0 >= 0) & (y1 < orig_h)
        
        # Limitar coordenadas a los bordes para evitar errores de índice
        x0_safe = np.clip(x0, 0, orig_w - 1)
        x1_safe = np.clip(x1, 0, orig_w - 1)
        y0_safe = np.clip(y0, 0, orig_h - 1)
        y1_safe = np.clip(y1, 0, orig_h - 1)
        
        # Calcular los pesos para interpolación bilineal
        alpha_h = x_coords - x0  # Peso horizontal (distancia a x0)
        alpha_v = y_coords - y0  # Peso vertical (distancia a y0)
        beta_h = 1 - alpha_h     # Peso complementario horizontal
        beta_v = 1 - alpha_v     # Peso complementario vertical
        
        # Obtener valores de los 4 píxeles vecinos
        I_00 = image[y0_safe, x0_safe]  # Esquina superior izquierda
        I_01 = image[y0_safe, x1_safe]  # Esquina superior derecha
        I_10 = image[y1_safe, x0_safe]  # Esquina inferior izquierda
        I_11 = image[y1_safe, x1_safe]  # Esquina inferior derecha
        
        # Interpolación bilineal: promedio ponderado de los 4 vecinos
        interpolated = (beta_h * beta_v * I_00 +
                        alpha_h * beta_v * I_01 +
                        beta_h * alpha_v * I_10 +
                        alpha_h * alpha_v * I_11)
        
        return interpolated, valid_mask


    # a. Volteo (flipping)
    @staticmethod
    def flip(image, mode='horizontal'):
        """
        Voltea una imagen horizontal o verticalmente.

        Args:
            image: Imagen de entrada como array.
            mode: Modo de volteo ('horizontal' o 'vertical').
        Returns:
            Imagen volteada como array.
        """
        # Verificamos que la imagen sea un array numpy
        if not isinstance(image, np.ndarray):
            raise ValueError("La imagen debe ser un array numpy.")
        if mode == 'horizontal':
            img_flipped = image[:, ::-1].copy()
        elif mode == 'vertical':
            img_flipped = image[::-1, :].copy()
        else:
            raise ValueError("El modo debe ser 'horizontal' o 'vertical'.")
        return img_flipped

    # b. Rotación con interpolación bilineal (OPTIMIZADO)
    @staticmethod
    def rotate(image, angle):
        """
        Rota una imagen por un ángulo dado usando interpolación bilineal,
        para definir la intensidad de los píxeles en la imagen rotada.
        
        Args:
            image: Imagen de entrada como array (escala de grises).
            angle: Ángulo de rotación en grados (positivo = antihorario).
        
        Returns:
            Imagen rotada como array.
        """
        # Obtener dimensiones de la imagen original
        orig_h, orig_w = image.shape

        # Centro de la imagen
        center_y, center_x = orig_h / 2, orig_w / 2

        # Convertir ángulo a radianes
        theta = np.deg2rad(angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Crear matrices de coordenadas para TODOS los píxeles de salida
        j_coords, i_coords = np.meshgrid(np.arange(orig_w), np.arange(orig_h))
        
        # Coordenadas relativas al centro
        x_rel = j_coords - center_x
        y_rel = i_coords - center_y
        
        # Aplicar rotación inversa (mapeo hacia atrás)
        # Para encontrar de dónde viene cada píxel de la imagen de salida
        x_prima = cos_theta * x_rel + sin_theta * y_rel + center_x
        y_prima = -sin_theta * x_rel + cos_theta * y_rel + center_y
        
        # Aplicamos la interpolación bilineal
        interpolated, valid_mask = DataAugmentation._bilinear_interpolation(
            image, x_prima, y_prima
        )
        
        # Aplicar máscara: píxeles fuera de límites quedan en 0
        rotated_image = np.where(valid_mask, interpolated, 0)
        
        return rotated_image.astype(image.dtype)

    # c. Traslación
    @staticmethod
    def translate(image, x_0, y_0):
        """
        Desplaza una imagen hacia una posición dada (x_0, y_0).
        
        Args:
            image: Imagen de entrada como array (escala de grises).
            x_0: Desplazamiento en el eje x.
            y_0: Desplazamiento en el eje y.
        
        Returns:
            Imagen trasladada como array.
        """

        # Obtener dimensiones de la imagen original
        orig_x, orig_y = image.shape

        # Creamos la imagen de salida con las mismas dimensiones
        trans_image = np.zeros((orig_x, orig_y), dtype=image.dtype)

        # Centro de la imagen
        center_x, center_y = orig_x / 2, orig_y / 2

        # Definirmos la matriz de traslación
        translation_matrix = np.array([[1, 0, x_0],
                                       [0, 1, y_0],
                                       [0, 0, 1]])
        
        for i in range(orig_x):
            for j in range(orig_y):
                # Vector [x, y, 1] relativo al centro
                coord = np.array([j - center_y, i - center_x, 1])

                # Aplicamos la transformación matricial
                # el @ es el operador de producto matricial en numpy
                transformed = translation_matrix @ coord

                # Coordenadas en la imagen original
                x_prima = transformed[0] + center_y
                y_prima = transformed[1] + center_x

                # Verificamos si las coordenadas están dentro de los límites
                if 0 <= x_prima < orig_y and 0 <= y_prima < orig_x:
                    trans_image[i, j] = image[int(y_prima), int(x_prima)]
                    
        return trans_image
        

    # d. Escalamiento con interpolación bilineal
    @staticmethod
    def scale(image, x_factor, y_factor):
        """
        Realiza el escalamiento de una imagen por factores dados en x e y,
        utilizando interpolación bilineal para definir la intensidad de los píxeles.
        
        Args:
            image: Imagen de entrada como array (escala de grises).
            x_factor: Factor de escala en el eje x.
            y_factor: Factor de escala en el eje y.
        
        Returns:
            Imagen escalada como array.
        """
        # Obtener dimensiones de la imagen original
        orig_h, orig_w = image.shape

        # Centro de la imagen
        center_x, center_y = orig_w / 2, orig_h / 2

        # Creamos matrices de coordenadas para TODOS los píxeles de salida
        j_coords, i_coords = np.meshgrid(np.arange(orig_w), np.arange(orig_h))

        # Mapeamos las coordenadas de la imagen de salida a la original (mapeo inverso)
        # Escalamos respecto al centro de la imagen
        x_prima = (j_coords - center_x) / x_factor + center_x
        y_prima = (i_coords - center_y) / y_factor + center_y
        
        
        # Aplicamos la interpolación bilineal
        interpolated, valid_mask = DataAugmentation._bilinear_interpolation(
            image, x_prima, y_prima
        )
        
        # Aplicar máscara: píxeles fuera de límites quedan en 0
        scaled_image = np.where(valid_mask, interpolated, 0)
        
        return scaled_image.astype(image.dtype)

    # e. Borrado aleatorio (Random erase)
    @staticmethod
    def random_erase(image, p=0.5, s_l=0.02, s_h=0.4, r1=0.3, r2=3.3, fill_mode='random'):
        """
        Implementa el algoritmo de Random Erasing para aumento de datos.
        
        Selecciona aleatoriamente una región rectangular de la imagen y la 
        reemplaza con valores según el modo de relleno especificado.
        
        Args:
            image: Imagen de entrada como array.
            p: Probabilidad de aplicar el borrado aleatorio (default: 0.5).
            s_l: Límite inferior del rango de área de borrado (proporción del área total).
            s_h: Límite superior del rango de área de borrado (proporción del área total).
            r1: Límite inferior del rango de aspect ratio del rectángulo.
            r2: Límite superior del rango de aspect ratio del rectángulo.
            fill_mode: Modo de relleno para la región borrada (default: 'random').
                - 'random': Rellena con valores aleatorios entre 0 y 255.
                - 'black': Rellena con color negro (valor 0).
                - 'white': Rellena con color blanco (valor 255).
        
        Returns:
            Imagen con región borrada aleatoriamente (o imagen original si no se aplica).
        """
        # Validar el modo de relleno
        valid_modes = ['random', 'black', 'white']
        if fill_mode not in valid_modes:
            raise ValueError(f"fill_mode debe ser uno de: {valid_modes}")
        
        # Creamos una copia de la imagen
        I_star = image.copy()
        
        # Obtener dimensiones de la imagen
        H, W = image.shape

        # Área total de la imagen
        S = W * H
        
        # Generamos un p1 aleatorio entre 0 y 1
        p1 = np.random.uniform(0, 1)
        
        # Si p1 >= p, regresamos la imagen sin cambios
        if p1 >= p:
            return I_star
        
        # Encontramos una región válida
        while True:
            # Generamos el área de la región de borrado S_e
            S_e = np.random.uniform(s_l, s_h) * S
            
            # Generamos el aspect ratio r_e
            r_e = np.random.uniform(r1, r2)
            
            # Calculamos dimensiones del rectángulo de borrado
            H_e = int(np.sqrt(S_e * r_e))
            W_e = int(np.sqrt(S_e / r_e))
            
            # Generamos posición aleatoria (x_e, y_e)
            x_e = np.random.randint(0, W)
            y_e = np.random.randint(0, H)
            
            # Verificamos si el rectángulo cabe dentro de la imagen
            if x_e + W_e <= W and y_e + H_e <= H:
                # Definimos la región I_e = (x_e, y_e, x_e + W_e, y_e + H_e)
                # Asignamos valores según el modo de relleno
                if fill_mode == 'random':
                    # Relleno con valores aleatorios
                    fill_value = np.random.randint(0, 256, size=(H_e, W_e), dtype=image.dtype)
                elif fill_mode == 'black':
                    # Relleno con negro (0)
                    fill_value = np.zeros((H_e, W_e), dtype=image.dtype)
                else:  # fill_mode == 'white'
                    # Relleno con blanco (255)
                    fill_value = np.full((H_e, W_e), 255, dtype=image.dtype)
                
                I_star[y_e:y_e + H_e, x_e:x_e + W_e] = fill_value
                
                return I_star

    # f. Mezclado de regiones (cutmix)
    @staticmethod
    def cutmix(image_A, image_B, alpha=1.0):
        """
        Implementa el algoritmo CutMix para aumento de datos.
        
        Combina dos imágenes reemplazando una región rectangular de image_A
        con la misma región de image_B. La fórmula es:
            x̃ = M ⊙ x_A + (1 - M) ⊙ x_B
        donde M es una máscara binaria (1 fuera del bounding box, 0 dentro).
        
        Args:
            image_A: Primera imagen de entrada como array (escala de grises).
            image_B: Segunda imagen de entrada como array (escala de grises).
            alpha: Parámetro de la distribución Beta(a, a) para muestrear lamda.
                   Si alpha=1.0, lamda se muestrea uniformemente de (0, 1).
        
        Returns:
            tuple: (imagen_mezclada, lamda_value)
                - imagen_mezclada: Imagen resultante de la combinación.
                - lamda_value: Valor de lamda usado (proporción de image_A en el resultado).
        """
        # Verificar que las imágenes tengan las mismas dimensiones
        if image_A.shape != image_B.shape:
            raise ValueError("Las imágenes deben tener las mismas dimensiones.")
        
        # Obtener dimensiones de la imagen
        H, W = image_A.shape
        
        # Muestreamos lamda de la distribución Beta(a, a)
        # Con a=1, esto es equivalente a Uniforme(0, 1)
        lam = np.random.beta(alpha, alpha)
        
        # Calculamos las dimensiones del bounding box B
        # r_w = W * sqrt(1 - lamda), r_h = H * sqrt(1 - lamda)
        # Esto hace que el área recortada sea (r_w * r_h) / (W * H) = 1 - lamda
        cut_ratio = np.sqrt(1 - lam)
        r_w = int(W * cut_ratio)
        r_h = int(H * cut_ratio)
        
        # Muestreamos la posición del centro del bounding box
        # r_x ~ Uniforme(0, W), r_y ~ Uniforme(0, H)
        r_x = np.random.randint(0, W)
        r_y = np.random.randint(0, H)
        
        # Calculamos las coordenadas del bounding box B = (x1, y1, x2, y2)
        # Asegurar que el box esté dentro de los límites de la imagen
        x1 = np.clip(r_x - r_w // 2, 0, W)
        y1 = np.clip(r_y - r_h // 2, 0, H)
        x2 = np.clip(r_x + r_w // 2, 0, W)
        y2 = np.clip(r_y + r_h // 2, 0, H)
        
        # Creamos la máscara binaria M
        # M = 1 fuera del bounding box B (conserva image_A)
        # M = 0 dentro del bounding box B (toma de image_B)
        M = np.ones((H, W), dtype=np.float32)
        M[y1:y2, x1:x2] = 0
        
        # Aplicamos la operación CutMix
        image_mixed = (M * image_A + (1 - M) * image_B).astype(image_A.dtype)
        
        # Ajustamos lamda al área real recortada
        # 1 - (área del box) / (área total)
        lam_adjusted = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)
        
        return image_mixed, lam_adjusted