import sys
import os

# Obtener la ruta del directorio actual del archivo
actual_dir = os.path.dirname(os.path.abspath(__file__))

# Ir un nivel arriba (a proc_dig_img/)
base_path = os.path.dirname(actual_dir)

if base_path not in sys.path:
    sys.path.append(base_path)

import paths
import pandas as pd
import matplotlib.pyplot as plt
import modos_color
import numpy as np


## Rutas del proyecto
data_dir = paths.data_dir()
avance3_base = paths.fase3_segmentacion_dir()

# imagenes seleccionadas para cambio de color
sel_img_path = os.path.join(avance3_base, "imagenes_prueba.csv")
sel_img_df = pd.read_csv(sel_img_path)

# Ruta para gauardar las imagenes
cambio_color_dir = os.path.join(avance3_base, "data_color")

# Para cada imagen del dataframe
for idx, row in sel_img_df.iterrows():
    img_name = row["ruta_imagen"]
    img_num = row["dificultad"]
    img_path = os.path.join(data_dir, img_name)
    
    # Leer la imagen
    R, G, B = modos_color.read_image_by_channels(img_path)
    
    # Para cada modo de color, convertir y guardar la imagen
    for mode in ["cmy", "cmyk", "hsi"]:
        if mode == "cmy":
            C, M, Y = modos_color.rgb_a_cmy(R, G, B)
            img_mode = np.stack([C, M, Y], axis=-1)
            path_img = os.path.join(cambio_color_dir, "cmy", f"{img_num}_{mode}.png")
        elif mode == "cmyk":
            C, M, Y, K = modos_color.rgb_a_cmyk(R, G, B)
            img_mode = np.stack([C, M, Y, K], axis=-1)
            path_img = os.path.join(cambio_color_dir, "cmyk", f"{img_num}_{mode}.png")
        elif mode == "hsi":
            H, S, I = modos_color.rgb_a_hsi(R, G, B)
            img_mode = np.stack([H, S, I], axis=-1)
            path_img = os.path.join(cambio_color_dir, "hsi", f"{img_num}_{mode}.png")
        
        # Guardar la imagen convertida
        plt.imsave(path_img, img_mode)
