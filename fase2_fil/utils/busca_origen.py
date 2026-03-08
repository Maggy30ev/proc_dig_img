"""
Buscar la ruta orinal de cada imagen segun donde se encuentren 
dentro de la carpeta "data" y genera un nuevo archivo .csv con la estructura: 
ruta_imagen,tipo
/test/images/1392_png.rf.0cff93c4891cf1412de96ef971310873.jpg,BH
/train/images/1212_png.rf.a92f0931457738e80f3a5a1f84552015.jpg,BH
/valid/images/370_png.rf.673d7f6e0b494b45d56e42fccaea5a6e.jpg,NBH

En este caso la ruta original debe buscarse en un directorio llamado "data" con la siguiente estructura:
data/
    test/
        images/
        labels/
    train/
        images/
        labels/
    valid/
        images/
        labels/
"""
import pandas as pd
import os
def buscar_ruta_original(directorio, archivo_csv, nuevo_archivo_csv):
    df = pd.read_csv(archivo_csv)
    datos = []
    for index, row in df.iterrows():
        nombre_imagen = row['nombre']
        tipo = row['tipo']
        ruta_encontrada = None
        for subdir, dirs, files in os.walk(directorio):
            if nombre_imagen in files:
                ruta_encontrada = os.path.join(subdir, nombre_imagen)
                break
        if ruta_encontrada:
            datos.append({'ruta_imagen': ruta_encontrada, 'tipo': tipo})
    nuevo_df = pd.DataFrame(datos)
    nuevo_df.to_csv(nuevo_archivo_csv, index=False)
if __name__ == "__main__":
    directorio = 'data'
    archivo_csv = 'imagenes.csv'
    nuevo_archivo_csv = 'imagenes_con_ruta.csv'
    buscar_ruta_original(directorio, archivo_csv, nuevo_archivo_csv)
