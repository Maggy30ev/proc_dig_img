import numpy as np
from typing import Optional
from scipy.ndimage import convolve, uniform_filter
from scipy.ndimage import minimum_filter, maximum_filter, median_filter

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
        """Calcula el clip limit para CLAHE basado en la varianza de la imagen."""

        # asegurar rango válido
        img = np.clip(image, 0, 255).astype(np.float64)

        # calcular varianza
        var = img.var()

        # varianza máxima teórica
        var_max = ((L-1)**2) / 4

        # normalizar
        var_n = var / var_max
        var_n = np.clip(var_n, 0, 1)

        # mapeo inverso
        alpha = alpha_min + (1 - var_n) * (alpha_max - alpha_min)

        # media esperada por bin
        mean_per_bin = img.size / L

        clip_limit = alpha * mean_per_bin

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
    def adaptive_local_noise_filter(image, noise_variance, kernel_size=7):
        """
        Filtro adaptativo local de reducción de ruido (Gonzalez & Woods, Ec. 5-32).
        
        Parámetros
        ----------
        image : np.ndarray
            Imagen de entrada (escala de grises o color).
            Acepta uint8 [0,255], float [0,1] o float [0,255].
        noise_variance : float
            Varianza del ruido (σ²η). Si la imagen se normalizó a [0,1]
            internamente, la varianza se reescala automáticamente.
        kernel_size : int
            Tamaño de la vecindad Sxy (debe ser impar). Típicamente 7.
        
        Retorna
        -------
        np.ndarray
            Imagen filtrada con el mismo dtype y rango que la entrada.
        """
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size debe ser un entero impar mayor o igual a 3.")
        if noise_variance < 0:
            raise ValueError("noise_variance no puede ser negativa.")

        # Manejo de imágenes a color: aplicar canal por canal
        if image.ndim == 3:
            canales = [
                Filtrado_Espacio.adaptive_local_noise_filter(
                    image[..., c], noise_variance, kernel_size
                )
                for c in range(image.shape[2])
            ]
            return np.stack(canales, axis=-1)

        original_dtype = image.dtype

        # --- Normalización de entrada ---
        # Siempre trabajamos internamente en float64 [0, 255].
        # Detectamos si la imagen viene en [0, 1] para reescalar
        # tanto la imagen como la varianza (que depende del rango).
        if np.issubdtype(original_dtype, np.floating):
            if image.max() <= 1.0:
                entrada_es_01 = True
                # Reescalamos imagen a [0, 255]
                g = image.astype(np.float64) * 255.0
                # La varianza escala con el cuadrado del factor (255²)
                noise_variance_interna = noise_variance * (255.0 ** 2)
            else:
                entrada_es_01 = False
                g = image.astype(np.float64)
                noise_variance_interna = float(noise_variance)
        else:
            entrada_es_01 = False
            g = image.astype(np.float64)
            noise_variance_interna = float(noise_variance)

        # --- Caso trivial: ruido cero ---
        if noise_variance_interna == 0:
            return image.copy()

        # --- 1) Media local (vectorizado con filtro separable en C) ---
        media_local = uniform_filter(g, size=kernel_size, mode='reflect')

        # --- 2) Varianza local: Var(X) = E[X²] - (E[X])² ---
        media_g2 = uniform_filter(g * g, size=kernel_size, mode='reflect')
        var_local = media_g2 - media_local ** 2

        # Corregir errores de redondeo en zonas perfectamente planas
        np.maximum(var_local, 0, out=var_local)

        # --- 3) Ratio σ²η / σ²Sxy, recortado a 1.0 como indica el libro ---
        ratio = np.where(
            var_local > noise_variance_interna,
            noise_variance_interna / np.maximum(var_local, 1e-12),
            1.0
        )

        # --- 4) Ecuación 5-32 ---
        salida = g - ratio * (g - media_local)

        # --- Desnormalización: devolvemos en el mismo rango y dtype de entrada ---
        if entrada_es_01:
            # Volvemos a [0, 1]
            salida = salida / 255.0
            salida = np.clip(salida, 0.0, 1.0)
            return salida.astype(original_dtype)

        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            salida = np.clip(salida, info.min, info.max)

        return salida.astype(original_dtype)

    # 6) FILTRO ADAPTATIVO DE MEDIANA (AMF)
    @staticmethod
    def adaptive_median_filter(image, S_max=7):
        """
        Filtro de mediana adaptativo (Gonzalez & Woods, Sec. 5.3).
        
        Parámetros
        ----------
        image : np.ndarray
            Imagen de entrada (escala de grises o color).
        S_max : int
            Tamaño máximo permitido de la ventana Sxy (debe ser impar e > 1).
        
        Retorna
        -------
        np.ndarray
            Imagen filtrada con el mismo dtype que la entrada.
        """
        if S_max < 3 or S_max % 2 == 0:
            raise ValueError("S_max debe ser un entero impar mayor o igual a 3.")

        # Manejo de imágenes a color: aplicar canal por canal
        if image.ndim == 3:
            canales = [
                Filtrado_Espacio.adaptive_median_filter(image[..., c], S_max)
                for c in range(image.shape[2])
            ]
            return np.stack(canales, axis=-1)

        original_dtype = image.dtype
        # Trabajamos en float para evitar problemas con uint8 en las comparaciones
        g = image.astype(np.float32)
        
        # Salida inicializada con la imagen original; se irá sobrescribiendo
        salida = g.copy()
        
        # Máscara de píxeles que aún no han sido resueltos (True = pendiente)
        pendientes = np.ones(g.shape, dtype=bool)

        # Iteramos por tamaños de ventana crecientes: 3, 5, 7, ..., S_max
        for S in range(3, S_max + 1, 2):
            # Calculamos min, max y mediana sobre TODA la imagen para esta ventana.
            # Esto es mucho más rápido que iterar pixel por pixel porque
            # scipy.ndimage usa código C optimizado internamente.
            z_min = minimum_filter(g, size=S, mode='reflect')
            z_max = maximum_filter(g, size=S, mode='reflect')
            z_med = median_filter(g, size=S, mode='reflect')

            # --- NIVEL A ---
            # Condición: z_min < z_med < z_max  -> la mediana NO es ruido impulsivo,
            # entonces pasamos al Nivel B para este píxel.
            nivel_A_ok = (z_med > z_min) & (z_med < z_max)
            
            # Píxeles que pasan a Nivel B en esta iteración (y que aún estaban pendientes)
            a_procesar_B = pendientes & nivel_A_ok

            if np.any(a_procesar_B):
                # --- NIVEL B ---
                # Si z_min < z_xy < z_max, la salida es z_xy (preserva detalle).
                # En caso contrario, la salida es z_med (probable píxel de ruido).
                nivel_B_ok = (g > z_min) & (g < z_max)
                
                # Donde Nivel B se cumple: dejamos g(x,y) (ya está en 'salida' por la copia inicial,
                # pero lo asignamos explícitamente para mayor claridad)
                usar_gxy = a_procesar_B & nivel_B_ok
                usar_med = a_procesar_B & ~nivel_B_ok
                
                salida[usar_gxy] = g[usar_gxy]
                salida[usar_med] = z_med[usar_med]
                
                # Marcamos estos píxeles como ya resueltos
                pendientes &= ~a_procesar_B
            
            # Los píxeles donde el Nivel A NO se cumplió siguen pendientes
            # y se reintentarán con una ventana más grande en la próxima iteración.
            
            if not np.any(pendientes):
                break  # Ya terminamos todos los píxeles, salimos antes

        # Para los píxeles que llegaron a S_max sin resolverse, salida = z_med
        # (corresponde al "Else, output z_med" del Nivel A cuando Sxy > S_max)
        if np.any(pendientes):
            salida[pendientes] = z_med[pendientes]

        # Restauramos el tipo original (recortando si es entero)
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            salida = np.clip(salida, info.min, info.max)
        
        return salida.astype(original_dtype)
    
    @staticmethod
    def filtro_gradiente_laplaciano_2(I, gamma=0.4):
        """
        Filtro Gradiente-Laplaciano
        I: imagen en escala de grises (numpy array)
        gamma: parámetro de corrección gamma
        Retorna: imagen filtrada (numpy array uint8)
        """
        I = I.astype(np.float64)

        # Laplaciano
        wl = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        L = convolve(I, wl, mode='nearest')
        R = I + L

        # Gradiente
        wx = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])
        wy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        dx = convolve(I, wx, mode='nearest')
        dy = convolve(I, wy, mode='nearest')

        mag = np.sqrt(dx**2 + dy**2)

        # Suavizado de la magnitud del gradiente
        ws = np.ones((3,3))
        m_suav = Filtrado_Espacio._filtro_suavizantes(mag, ws, mode_padding='nearest')

        # Mascara combinada (operación a nivel de píxel )
        mask = R * m_suav

        # Imagen final
        G = I + mask

        # Normalización y corrección gamma
        norm = (G - G.min()) / (G.max() - G.min() + 1e-8)  # normalizar a [0,1]
        G_gamma = norm ** gamma  # aplicar corrección gamma

        return G_gamma
    
    import numpy as np


    @staticmethod
    def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8), n_bins=256):
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Parámetros
        ----------
        image : np.ndarray
            Imagen de entrada (escala de grises o color, uint8).
        clip_limit : float
            Umbral de recorte del histograma. Valores típicos: 2.0 - 4.0.
            Es un multiplicador respecto al promedio uniforme del histograma.
        tile_grid_size : tuple(int, int)
            Número de tiles (filas, columnas) en que se divide la imagen.
        n_bins : int
            Número de bins del histograma (256 para uint8).
        
        Retorna
        -------
        np.ndarray
            Imagen con contraste mejorado, mismo dtype que la entrada.
        """
        # Manejo de imágenes a color: aplicar solo al canal de luminancia.
        # Lo correcto en CLAHE para color es trabajar en un espacio donde
        # luminancia y croma estén separados (ej. YCrCb o LAB), no aplicarlo
        # canal por canal en RGB porque desplaza los colores.
        if image.ndim == 3:
            # Conversión RGB -> YCrCb manual (sin OpenCV) usando ITU-R BT.601
            rgb = image.astype(np.float32)
            Y  =  0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            Cr =  0.5   * rgb[..., 0] - 0.4187 * rgb[..., 1] - 0.0813 * rgb[..., 2] + 128
            Cb = -0.1687 * rgb[..., 0] - 0.3313 * rgb[..., 1] + 0.5   * rgb[..., 2] + 128
            
            Y_eq = Filtrado_Espacio.clahe(
                np.clip(Y, 0, 255).astype(np.uint8),
                clip_limit, tile_grid_size, n_bins
            ).astype(np.float32)
            
            # YCrCb -> RGB
            R = Y_eq + 1.402 * (Cr - 128)
            G = Y_eq - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
            B = Y_eq + 1.772 * (Cb - 128)
            out = np.stack([R, G, B], axis=-1)
            return np.clip(out, 0, 255).astype(image.dtype)

        
        original_dtype = image.dtype
    
        # Normalización de entrada: siempre trabajamos internamente con uint8
        if np.issubdtype(original_dtype, np.floating):
            entrada_es_01 = image.max() <= 1.0
            if entrada_es_01:
                img = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            else:
                img = np.clip(image, 0, 255).astype(np.uint8)
        else:
            entrada_es_01 = False
            img = image.astype(np.uint8)

        H, W = img.shape
        n_tiles_y, n_tiles_x = tile_grid_size

        # 1) Padding para que la imagen sea divisible exactamente entre los tiles.
        # Replicamos los bordes para no introducir valores artificiales en los histogramas.
        pad_y = (n_tiles_y - H % n_tiles_y) % n_tiles_y
        pad_x = (n_tiles_x - W % n_tiles_x) % n_tiles_x
        img_pad = np.pad(img, ((0, pad_y), (0, pad_x)), mode='edge')
        Hp, Wp = img_pad.shape
        tile_h = Hp // n_tiles_y
        tile_w = Wp // n_tiles_x

        # 2) Calcular el LUT (lookup table) de cada tile.
        # luts[i, j] es un arreglo de n_bins valores: el mapeo de intensidad para el tile (i, j).
        luts = np.zeros((n_tiles_y, n_tiles_x, n_bins), dtype=np.float32)
        
        # Clip limit absoluto: clip_limit es un factor respecto al valor uniforme.
        # Un histograma uniforme tendría (tile_h * tile_w / n_bins) píxeles por bin.
        pixels_por_tile = tile_h * tile_w
        clip_abs = max(1, int(clip_limit * pixels_por_tile / n_bins))

        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                tile = img_pad[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
                
                # Histograma del tile
                hist, _ = np.histogram(tile, bins=n_bins, range=(0, n_bins))
                
                # --- Recorte del histograma ---
                # Tomamos el exceso por encima del clip_limit y lo redistribuimos
                # uniformemente entre todos los bins. Esto se hace iterativamente
                # porque al redistribuir podríamos volver a exceder el límite,
                # pero en la práctica una sola redistribución es suficientemente buena.
                exceso = np.maximum(hist - clip_abs, 0).sum()
                hist = np.minimum(hist, clip_abs)
                hist += exceso // n_bins  # redistribución uniforme
                
                # El residuo (exceso % n_bins) se reparte uno por uno; para mantener
                # la implementación simple y rápida, lo sumamos al primer bin.
                # En la práctica esto es despreciable visualmente.
                hist[0] += exceso % n_bins
                
                # --- CDF y construcción del LUT ---
                cdf = hist.cumsum().astype(np.float32)
                cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
                denom = cdf[-1] - cdf_min
                if denom > 0:
                    lut = (cdf - cdf_min) / denom * (n_bins - 1)
                else:
                    lut = np.zeros(n_bins, dtype=np.float32)
                
                luts[i, j] = lut

        # 3) Aplicar los LUTs con interpolación bilineal entre tiles.
        # Para cada píxel calculamos: a qué tiles "pertenece" y con qué peso.
        # Los centros de los tiles están en (tile_h/2 + i*tile_h, tile_w/2 + j*tile_w).
        # Cada píxel se interpola entre los 4 tiles vecinos cuyos centros lo rodean.
        
        # Coordenadas de cada píxel respecto a la grilla de centros de tiles
        ys = np.arange(Hp, dtype=np.float32)
        xs = np.arange(Wp, dtype=np.float32)
        
        # "Posición" en unidades de tile (0 = centro del primer tile)
        fy = (ys - tile_h / 2) / tile_h
        fx = (xs - tile_w / 2) / tile_w
        
        # Índices de tiles vecinos (superior/inferior, izquierdo/derecho)
        iy0 = np.clip(np.floor(fy).astype(np.int32), 0, n_tiles_y - 1)
        iy1 = np.clip(iy0 + 1, 0, n_tiles_y - 1)
        ix0 = np.clip(np.floor(fx).astype(np.int32), 0, n_tiles_x - 1)
        ix1 = np.clip(ix0 + 1, 0, n_tiles_x - 1)
        
        # Pesos de interpolación (recortados a [0, 1] para los bordes)
        wy = np.clip(fy - np.floor(fy), 0, 1)
        wx = np.clip(fx - np.floor(fx), 0, 1)
        
        # En los bordes (antes del primer centro o después del último) forzamos
        # que el peso vaya completo al tile más cercano, para no extrapolar.
        wy[fy < 0] = 0
        wy[fy > n_tiles_y - 1] = 1
        wx[fx < 0] = 0
        wx[fx > n_tiles_x - 1] = 1

        # Para vectorizar, expandimos los índices a 2D
        IY0, IX0 = np.meshgrid(iy0, ix0, indexing='ij')
        IY1, IX1 = np.meshgrid(iy1, ix1, indexing='ij')
        WY,  WX  = np.meshgrid(wy,  wx,  indexing='ij')

        # Valor del píxel original (índice dentro del LUT)
        vals = img_pad  # shape (Hp, Wp), valores 0..n_bins-1
        
        # Consultamos los 4 LUTs vecinos para cada píxel (vectorizado)
        lut_tl = luts[IY0, IX0, vals]  # top-left
        lut_tr = luts[IY0, IX1, vals]  # top-right
        lut_bl = luts[IY1, IX0, vals]  # bottom-left
        lut_br = luts[IY1, IX1, vals]  # bottom-right
        
        # Interpolación bilineal
        top    = lut_tl * (1 - WX) + lut_tr * WX
        bottom = lut_bl * (1 - WX) + lut_br * WX
        out    = top    * (1 - WY) + bottom * WY

        # Quitamos el padding y convertimos al dtype original
        out = out[:H, :W]
        out = np.clip(np.round(out), 0, n_bins - 1)
    
        if np.issubdtype(original_dtype, np.floating):
            if entrada_es_01:
                return (out / 255.0).astype(original_dtype)
            return out.astype(original_dtype)
        return out.astype(original_dtype)