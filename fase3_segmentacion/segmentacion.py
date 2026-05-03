import numpy as np
from scipy.ndimage import convolve




class Segmentacion:

    #SUBFUNCIÓN COMPARTIDA (Otsu)
    @staticmethod
    def _calcular_estadisticas(img, L=256):
        """Calcula probabilidades y medias acumuladas del histograma.

        Parámetros:
        img : Imagen de entrada en escala de grises
        L   : Número de niveles de intensidad (256 para 8 bits)

        Retorna:
        pk    : Probabilidad de cada nivel de gris (array L)
        omega : Probabilidad acumulada hasta cada nivel (array L)
        mu    : Media acumulada hasta cada nivel (array L)
        mG    : Media global de la imagen
        """
        total   = img.size
        niveles = np.arange(L, dtype=np.float64)
        hist    = np.bincount(img.ravel().astype(np.uint8), minlength=L).astype(np.float64)
        pk      = hist / total
        omega   = np.cumsum(pk)
        mu      = np.cumsum(niveles * pk)
        mG      = mu[-1]
        return pk, omega, mu, mG

    #OTSU SIMPLE
    @staticmethod
    def otsu_simple(img, L=256):
        """Umbralización de Otsu con un único umbral óptimo (2 clases).

        Divide la imagen en fondo y objeto maximizando la varianza entre clases.

        Parámetros:
        img : Imagen de entrada en escala de grises (uint8)
        L   : Número de niveles de intensidad. Por defecto 256.

        Retorna:
        img_umbralizada : Imagen binaria resultante (uint8, 0 o 255)
        T               : Umbral óptimo encontrado
        """
        img = img.astype(np.uint8)
        _, omega, mu, mG = Segmentacion._calcular_estadisticas(img, L)

        with np.errstate(divide='ignore', invalid='ignore'):
            sigma_cuad = (omega * mG - mu) ** 2 / (omega * (1 - omega))
            sigma_cuad = np.nan_to_num(sigma_cuad, nan=0.0, posinf=0.0)

        T               = int(np.argmax(sigma_cuad))
        img_umbralizada = np.where(img <= T, 255, 0).astype(np.uint8)
        return img_umbralizada, T

    #OTSU DOBLE
    @staticmethod
    def otsu_doble(img, L=256):
        """Umbralización de Otsu con dos umbrales óptimos (3 clases).

        Divide la imagen en fondo, gris medio y objeto maximizando
        la varianza entre clases sobre todos los pares (T1, T2).

        Parámetros:
        img : Imagen de entrada en escala de grises (uint8)
        L   : Número de niveles de intensidad. Por defecto 256.

        Retorna:
        img_seg : Imagen segmentada en 3 clases (uint8, valores 0 / 128 / 255)
        T1      : Umbral inferior óptimo
        T2      : Umbral superior óptimo
        """
        img = img.astype(np.uint8)
        _, omega, mu, mG = Segmentacion._calcular_estadisticas(img, L)

        matriz_var = np.zeros((L, L), dtype=np.float64)

        for T1 in range(L - 2):
            for T2 in range(T1 + 1, L - 1):
                w1 = omega[T1]
                m1 = mu[T1] / w1 if w1 > 0 else 0.0

                w2 = omega[T2] - omega[T1]
                m2 = (mu[T2] - mu[T1]) / w2 if w2 > 0 else 0.0

                w3 = 1.0 - omega[T2]
                m3 = (mG - mu[T2]) / w3 if w3 > 0 else 0.0

                matriz_var[T1, T2] = (w1 * (m1 - mG) ** 2
                                    + w2 * (m2 - mG) ** 2
                                    + w3 * (m3 - mG) ** 2)

        idx     = np.argmax(matriz_var)
        T1, T2  = np.unravel_index(idx, (L, L))
        T1, T2  = int(T1), int(T2)

        img_seg = np.zeros_like(img, dtype=np.uint8)
        img_seg[img < T1]                    = 0
        img_seg[(img >= T1) & (img < T2)]   = 128
        img_seg[img >= T2]                   = 255
        return img_seg, T1, T2

    #CANNY
    @staticmethod
    def canny(img, t_high_ratio=0.10, t_low_ratio=0.03):
        """Aplica el detector de bordes Canny a una imagen.

        Parámetros:
        img          : Imagen de entrada en escala de grises
        t_high_ratio : Fracción del máximo para el umbral alto. Por defecto 0.90
        t_low_ratio  : Fracción del máximo para el umbral bajo. Por defecto 0.10

        Retorna:
        borde : Imagen binaria con los bordes detectados (uint8, 0 o 1)
        """
        I    = img.astype(np.float64)
        M, N = I.shape

        mdx = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float64)
        mdy = np.array([[-1,  0,  1],
                         [-2,  0,  2],
                         [-1,  0,  1]], dtype=np.float64)

        dx      = convolve(I, mdx, mode='constant', cval=0.0)
        dy      = convolve(I, mdy, mode='constant', cval=0.0)
        mag     = np.sqrt(dx**2 + dy**2)
        dirgrad = np.degrees(np.arctan2(dx, dy))

        Ib = np.zeros((M, N))
        for x in range(M):
            for y in range(N):
                actual = dirgrad[x, y] % 180
                if actual <= 22.5 or actual >= 157.5:
                    Ib[x, y] = 90
                elif actual < 67.5:
                    Ib[x, y] = -45
                elif actual <= 112.5:
                    Ib[x, y] = 0
                else:
                    Ib[x, y] = 45

        Mag               = np.zeros((M + 2, N + 2))
        Mag[1:M+1, 1:N+1] = mag
        Gn                = np.zeros((M, N))

        for x in range(1, M + 1):
            for y in range(1, N + 1):
                alpha  = Ib[x-1, y-1]
                actual = Mag[x, y]
                if alpha == 0:
                    v1, v2 = Mag[x, y-1], Mag[x, y+1]
                elif alpha == 45:
                    v1, v2 = Mag[x-1, y+1], Mag[x+1, y-1]
                elif alpha == 90:
                    v1, v2 = Mag[x-1, y], Mag[x+1, y]
                else:
                    v1, v2 = Mag[x-1, y-1], Mag[x+1, y+1]
                Gn[x-1, y-1] = actual if (actual >= v1 and actual >= v2) else 0

        maxM = Gn.max()
        T_H  = maxM * t_high_ratio
        T_L  = maxM * t_low_ratio

        G_NH  = (Gn >= T_H).astype(np.uint8)
        G_NL  = (Gn >= T_L).astype(np.uint8)
        G_NLI = G_NL & ~G_NH

        Gnli               = np.zeros((M + 2, N + 2), dtype=np.uint8)
        Gnli[1:M+1, 1:N+1] = G_NLI
        borde               = np.zeros((M + 2, N + 2), dtype=np.uint8)
        borde[1:M+1, 1:N+1] = G_NH

        cambio = True
        while cambio:
            cambio = False
            for x in range(1, M + 1):
                for y in range(1, N + 1):
                    if Gnli[x, y] == 1 and borde[x, y] == 0:
                        if borde[x-1:x+2, y-1:y+2].any():
                            borde[x, y] = 1
                            cambio = True

        return borde[1:M+1, 1:N+1]
