import numpy as np
from typing import Optional
from scipy.ndimage import convolve

class Filtrado_Espacio:
    def __init__(self, seed: Optional[int] = None):
        """
        Inicializa el generador de aumento de datos.
        
        Args:
            seed: Semilla para reproducibilidad de resultados aleatorios
        """
        if seed is not None:
            np.random.seed(seed)

    # Calculo del histograma
    @staticmethod
    def _histograma(image):
        """Calcula el histograma de una imagen.

        Parámetros:
        image : Imagen de entrada en escala de grises

        Retorna:
        hist : Histograma de la imagen, con 256 bins para valores de intensidad.
        """
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

        return hist
    
    @staticmethod
    def clip_limit_value(image, alpha_min=1.2, alpha_max=4.5, L=256):
        """Calcula el clip limit para CLAHE basado en la entropía de la imagen.

        Lógica:
        - Entropía baja  → imagen homogénea → se necesita más realce → cliplimit alto
        - Entropía alta  → imagen con mucho detalle → menos realce → cliplimit bajo

        Parámetros:
        image     : Imagen de entrada en escala de grises (uint8 o float)
        alpha_min : Factor mínimo sobre la media por bin (entropía alta)
        alpha_max : Factor máximo sobre la media por bin (entropía baja)
        L         : Número de niveles de intensidad (256 para 8 bits)

        Retorna:
        clip_limit : int, valor entero del clip limit en conteos de histograma
        """
        # Asegurar tipo entero para bincount
        img_u8 = np.clip(image, 0, 255).astype(np.uint8)

        # Histograma normalizado como distribución de probabilidad
        hist = np.bincount(img_u8.ravel(), minlength=L).astype(np.float64)
        p = hist / hist.sum()
        p = p[p > 0]  # descartar bins vacíos para el log

        # Entropía de Shannon (bits)
        H = -(p * np.log2(p)).sum()

        # Normalizar al rango [0, 1]  (H_max = log2(L) ≈ 8 para 256 niveles)
        Hn = H / np.log2(L)

        # Mapeo inverso: menor entropía → alpha más alto → más realce
        alpha = alpha_min + (1.0 - Hn) * (alpha_max - alpha_min)

        # Media esperada de conteos por bin
        media_por_bin = image.size / L

        clip_limit = alpha * media_por_bin
        return max(1, int(round(clip_limit)))
    

    # Función de filtrado
    @staticmethod
    def _filtro_suavizantes(img, kernel, mode_padding='constant'):
        """Aplica un filtro suavizante a una imagen utilizando un kernel dado.

        Parámetros:
        img : Imagen de entrada en escala de grises
        kernel : Kernel de convolución para el filtro suavizante
        mode_padding : Manejo de los bordes. Por defecto 'constant', que rellena con ceros. 
        Otras opciones incluyen 'reflect' que refleja los bordes, 'nearest' que repite el valor del borde,
        'mirror' que refleja sin repetir el borde, y 'wrap' que envuelve la imagen.

        Retorna:
        img_suavizada : Imagen resultante después de aplicar el filtro suavizante.
        """
        img = img.astype(float)

        # Convolución
        conv = convolve(img, kernel, mode=mode_padding)
        suma_kernel = np.sum(kernel)

        img_suavizada = conv / suma_kernel

        return img_suavizada.astype(np.uint8)
    
    # 1) ECUALIZACIÓN DE HISTOGRAMA
    @staticmethod
    def ecualizacion(image):
        """ Ecualización del histograma a una imagen.

            Parámetros:
            image : Imagen de entrada en escala de grises

            Retorna:
            img_ecualizada : Imagen resultante después de la ecualización.
        """
        # Calcular el histograma
        hist = Filtrado_Espacio._histograma(image)

        # Calcular la función de distribución acumulativa (CDF)
        cdf = hist.cumsum()

        # Normalizar la CDF
        cdf_normalizada = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

        # Mapear los valores de intensidad originales a los nuevos valores ecualizados
        img_ecualizada = np.interp(image.flatten(), np.arange(256), cdf_normalizada)

        return img_ecualizada.reshape(image.shape).astype(np.uint8)
    
    # 2) CLAHE
    @staticmethod
    def clahe(image, cliplimit=2.0):
        """Aplica el algoritmo CLAHE (Contrast Limited Adaptive Histogram Equalization) a una imagen.

        Parámetros:
        image : Imagen de entrada en escala de grises
        cliplimit : Límite de contraste para la ecualización adaptativa. Por defecto 2.0.
        Retorna:
        img_clahe : Imagen resultante después de aplicar CLAHE.
        """
        image = image.astype(np.uint8)
        cliplimit = max(1, int(round(cliplimit)))

        M, N = image.shape

        tam1 = M // 2
        tam2 = N // 2

        p1 = image[:tam1, :tam2]
        p2 = image[tam1:, :tam2]
        p3 = image[:tam1, tam2:]
        p4 = image[tam1:, tam2:]

        tiles = [p1, p2, p3, p4]

        L = 256
        luts = []


        for tile in tiles:

            frq = Filtrado_Espacio._histograma(tile).astype(np.int64)

            frq_rec = np.minimum(frq, cliplimit)

            exceso = frq - frq_rec
            tot_exc = int(exceso.sum())

            suma_exc = tot_exc // L

            rep = frq_rec + suma_exc

            resto = int(tot_exc - suma_exc * L)
            if resto > 0:
                rep[:resto] += 1

            cdf = np.cumsum(rep)
            cdf = cdf / cdf[-1]

            lut = np.floor((L-1) * cdf).astype(np.uint8)

            luts.append(lut)

        lut1, lut2, lut3, lut4 = luts

        y = np.arange(M)
        x = np.arange(N)

        Y, X = np.meshgrid(y, x, indexing="ij")

        dy = Y / tam1
        dx = X / tam2

        dy = np.clip(dy, 0, 1)
        dx = np.clip(dx, 0, 1)

        val = image

        f11 = lut1[val]
        f12 = lut3[val]
        f21 = lut2[val]
        f22 = lut4[val]


        img_clahe = (
            f11 * (1-dx) * (1-dy) +
            f12 * dx * (1-dy) +
            f21 * (1-dx) * dy +
            f22 * dx * dy
        )

        return img_clahe.astype(np.uint8)

    # 5) HIGHBOOSTING
    @staticmethod
    def highboost(img, kernel, k=1.5):
        """Aplica un filtro highboost a una imagen.

        Parámetros:
        img : Imagen de entrada en escala de grises
        kernel : Kernel de convolución para el filtro suavizante (usado para obtener la imagen suavizada)
        k : Factor de realce. Por defecto 1.5. Un valor mayor que 1 realza más los detalles.

        Retorna:
        img_realzada : Imagen resultante después de aplicar el filtro high-boost.
        """
        img = img.astype(float)

        # Obtener la imagen suavizada
        img_suavizada = Filtrado_Espacio._filtro_suavizantes(img, kernel)

        # Calcular la máscara
        mascara = img - img_suavizada

        # Aplicar el filtro high-boost
        img_realzada = img + k * mascara

        # Normalizar a rango [0, 255]
        img_realzada = np.clip(img_realzada, 0, 255)

        return img_realzada.astype(np.uint8)

    # 4) GRADIENTE - LAPLACIANO
    @staticmethod
    def filtro_gradiente_laplaciano(img_entrada):
        """
        Aplica el filtro combinado Gradiente-Laplaciano para realzar bordes.

        El proceso es:
        1. Aplica filtro Laplaciano para detectar bordes finos.
        2. Realza la imagen con el Laplaciano (sharpening).
        3. Calcula el gradiente de Sobel (magnitud).
        4. Suaviza la magnitud con filtro de caja (box filter).
        5. Multiplica la imagen realzada por la magnitud suavizada.
        6. Suma el resultado a la imagen original.
        7. Aplica corrección gamma para comprimir el rango dinámico.

        Parámetros
        img_entrada : numpy.ndarray
            Imagen de entrada (puede ser uint8 o float).
            Corresponde a la imagen con ruido de Poisson (JP en el contexto).

        Retorna
        img_gamma : numpy.ndarray (uint8, rango 0-255)
            Imagen final con bordes realzados y corrección gamma aplicada.
        """

        #'Convertir a double
        img_d = img_entrada.astype(np.float64)

        #LAPLACIANO
        #Máscara de 4 vecinos (Laplaciano estándar)
        masc_lap = np.array([[0,  1,  0],
                            [1, -4,  1],
                            [0,  1,  0]], dtype=np.float64)

        #Aplicar filtro Laplaciano con relleno de ceros en los bordes
        #mode='constant' cval=0
        img_filtrada = convolve(img_d, masc_lap, mode='constant', cval=0.0)

        # Coeficiente c=-1 para realzar (sharpening con Laplaciano)
        c = -1
        img_realzada = img_d + c * img_filtrada  # imagen con bordes realzados

        #GRADIENTE
        # Máscaras en X e Y
        masc_grad_x = np.array([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=np.float64)

        masc_grad_y = np.array([[-1,  0,  1],
                                [-2,  0,  2],
                                [-1,  0,  1]], dtype=np.float64)

        #Aplicar gradiente en X e Y
        img_grad_x = convolve(img_d, masc_grad_x, mode='constant', cval=0.0)
        img_grad_y = convolve(img_d, masc_grad_y, mode='constant', cval=0.0)

        #Calcular la magnitud del gradiente
        mag_img_fil = np.sqrt(img_grad_x**2 + img_grad_y**2)

        #SUAVIZADO DE LA MAGNITUD
        #Filtro de caja (box filter) 3x3 normalizado
        filt_caja = (1.0 / 9.0) * np.ones((3, 3), dtype=np.float64)
        mag_img_fil_suav = convolve(mag_img_fil, filt_caja, mode='constant', cval=0.0)

        #COMBINACIÓN GRADIENTE-LAPLACIANO
        #Multiplicar la imagen realzada por la magnitud suavizada
        masc_grad_lap = img_realzada * mag_img_fil_suav

        #Sumar la máscara a la imagen original
        G_img = img_d + masc_grad_lap

        #CORRECCIÓN GAMMA
        # Tomar valor absoluto para eliminar negativos
        G_img_abs = np.abs(G_img)

        # Aplicar gamma = 0.4 (compresión de rango dinámico)
        G_img_gamma = G_img_abs ** 0.4

        #Rescalar al rango [0, 255] y convertir a uint8
        G_min, G_max = G_img_gamma.min(), G_img_gamma.max()
        if G_max - G_min > 0:
            G_norm = (G_img_gamma - G_min) / (G_max - G_min)
        else:
            G_norm = np.zeros_like(G_img_gamma)

        img_gamma = (G_norm * 255).astype(np.uint8)

        return img_gamma


    # 5) FILTRADO ADAPTATIVO LOCAL
    @staticmethod
    def filtro_adaptativo_local(img_con_ruido):
        """
        Aplica el Filtro Adaptativo Local para reducir ruido gaussiano preservando detalles.

        Lógica:
        - Si la varianza local <= varianza global ent usar la media local
            (zona homogénea, reemplazar con el promedio del vecindario).
        - Si la varianza local > varianza global ent aplicar la fórmula
            adaptativa que pondera entre el píxel original y la media local.

        Parámetros:
        img_con_ruido : numpy.ndarray
            Imagen con ruido gaussiano, tipo uint8 o float.

        Retorna:
        Img_filt_Adap : numpy.ndarray (float64)
            Imagen filtrada. Para visualizar, convertir a uint8 con:
            np.clip(Img_filt_Adap, 0, 255).astype(np.uint8)
        """

        #Convertir a double
        J_d = img_con_ruido.astype(np.float64)
        M, N = J_d.shape

        #Estadísticas globales de la imagen completa
        media_global = np.mean(J_d)
        std_global   = np.std(J_d)
        var_global   = std_global ** 2  # varianza global

        print(f'Media global: {media_global:.4f}')
        print(f'Varianza global: {var_global:.4f}')

        #Imagen de salida inicializada en ceros
        Img_filt_Adap = np.zeros((M, N), dtype=np.float64)

        #Aumentar la imagen con borde de ceros (padding de 1 píxel por lado)
        Img1_Au = np.zeros((M + 2, N + 2), dtype=np.float64)
        Img1_Au[1:M+1, 1:N+1] = J_d  # copiar imagen original en el centro

        #Recorrer cada píxel de la imagen original
        for x in range(1, M + 1):       #índices del 1 al M (en el array aumentado)
            for y in range(1, N + 1):   #índices del 1 al N

                #Extraer vecindad 3x3 alrededor del píxel actual
                pixVeci = Img1_Au[x-1:x+2, y-1:y+2]  #ventana 3x3

                #Estadisticas locales de la vecindad
                media_local = np.mean(pixVeci)
                std_local   = np.std(pixVeci)
                var_local   = std_local ** 2

                if var_local <= var_global:
                    #reemplazar con la media local
                    Img_filt_Adap[x-1, y-1] = media_local
                else:
                    #fórmula adaptativa
                    #Pondera entre el píxel original y la media local
                    Img_filt_Adap[x-1, y-1] = (J_d[x-1, y-1]
                                                - (var_global / var_local)
                                                * (J_d[x-1, y-1] - media_local))

        return Img_filt_Adap


    # 6) FILTRO ADAPTATIVO DE MEDIANA (AMF)
    @staticmethod
    def filtro_adaptativo_mediana(img_con_ruido, S_max=7):
        """
        Aplica el Filtro Adaptativo de Mediana (Adaptive Median Filter - AMF).

        Diseñado para eliminar ruido Salt & Pepper preservando los detalles.
        A diferencia del filtro de mediana estándar, este aumenta dinámicamente
        el tamaño de la ventana si la mediana local también es ruido.

        Algoritmo (por cada píxel):
        Fase A
            - Si z_min < z_med < z_max ent la mediana NO es ruido ent ir a Fase B
            - Si no ent aumentar la ventana (hasta S_max)
        Fase B - ¿Es el píxel actual un impulso?
            - Si z_min < z_xy < z_max ent el píxel NO es ruido ent conservar z_xy
            - Si no ent reemplazar con z_med

        Parámetros:
        img_con_ruido : numpy.ndarray
            Imagen con ruido Salt & Pepper (J en el contexto), tipo uint8 o float.
        S_max : int, opcional
            Tamaño máximo de ventana permitido (debe ser impar). Por defecto 7.

        Retorna:
        ImgAMF : numpy.ndarray (float64)
            Imagen con ruido Salt & Pepper reducido.
            Para visualizar: np.clip(ImgAMF, 0, 255).astype(np.uint8)
        """

        #Convertir a double
        J_SP = img_con_ruido.astype(np.float64)
        M, N = J_SP.shape

        #Calcular el aumento máximo de borde (radio de la ventana más grande)
        aum_max = S_max // 2  # para 7x7 → aum_max = 3

        #Crear imagen aumentada con ceros (padding para los bordes)
        J_aumAMF = np.zeros((M + 2 * aum_max, N + 2 * aum_max), dtype=np.float64)
        #Copiar la imagen original en el centro
        J_aumAMF[aum_max:aum_max+M, aum_max:aum_max+N] = J_SP

        #Imagen de salida
        ImgAMF = np.zeros((M, N), dtype=np.float64)

        #Recorrer cada píxel
        for x in range(M):
            for y in range(N):

                s = 3        #ventana inicial 3x3
                salir = False

                while not salir:
                    act = s // 2  #radio de la ventana actual

                    # Coordenadas en la imagen aumentada
                    xi = x + aum_max - act
                    xf = x + aum_max + act + 1 # +1 porque el límite superior no se incluye
                    yi = y + aum_max - act
                    yf = y + aum_max + act + 1

                    # Extraer vecindad cuadrada de tamaño s x s
                    pixVeci = J_aumAMF[xi:xf, yi:yf]

                    z_min = pixVeci.min()    # mínimo local
                    z_max = pixVeci.max()    # máximo local
                    z_med = np.median(pixVeci)  # mediana local
                    z_xy  = J_SP[x, y]      # valor del píxel actual

                    # FASE A
                    A = z_med - z_min  # > 0 si z_med no es el mínimo
                    B = z_med - z_max  # < 0 si z_med no es el máximo

                    if A > 0 and B < 0:
                        C = z_xy - z_min
                        D = z_xy - z_max

                        #FASE B
                        if C > 0 and D < 0:
                            # El píxel NO es ruido ent conservar su valor
                            ImgAMF[x, y] = z_xy
                        else:
                            # El píxel es ruido ent reemplazar con la mediana local
                            ImgAMF[x, y] = z_med

                        salir = True  # salir del while

                    else:
                        # La mediana podría ser ruido ent aumentar ventana
                        s += 2  # siguiente tamaño impar (3,5,7)

                        if s > S_max:
                            #Se alcanzó el tamaño máximo ent usar mediana de todos modos
                            ImgAMF[x, y] = z_med
                            salir = True

        return ImgAMF
