"""
Lee un directorio con imagenes en subcarpetas:
Base/
    Bosque/
        Fuego/
        Humo/
    NoBosque/
        Fuego/
        Humo/

Y crea un archivo CSV con la siguiente estructura, donde la columna 'nombre'
contiene el nombre completo de la imagen y el tipo corresponde con el contenido
de la imagen segun las subcarpetas donde se encuentre:
nombre,tipo
imagen1.jpg, BF
imagen2.jpg, BH
imagen3.jpg, NBF
imagen4.jpg, NBH
"""
import pandas as pd
import os
def crear_csv(directorio, archivo_csv):
    datos = []
    for subdir, dirs, files in os.walk(directorio):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                ruta_imagen = os.path.relpath(os.path.join(subdir, file), directorio)
                partes = ruta_imagen.split(os.sep)
                if len(partes) >= 3:
                    categoria = partes[0]
                    tipo = partes[1]
                    if categoria == 'Bosque':
                        if tipo == 'Fuego':
                            etiqueta = 'BF'
                        elif tipo == 'Humo':
                            etiqueta = 'BH'
                    elif categoria == 'NoBosque':
                        if tipo == 'Fuego':
                            etiqueta = 'NBF'
                        elif tipo == 'Humo':
                            etiqueta = 'NBH'
                    datos.append({'nombre': ruta_imagen, 'tipo': etiqueta})
    df = pd.DataFrame(datos)
    df.to_csv(archivo_csv, index=False)
if __name__ == "__main__":
    directorio = 'PDI_FOREST'
    archivo_csv = 'imagenes.csv'
    crear_csv(directorio, archivo_csv)
