import matplotlib.pyplot as plt

def mostrar_imagenes(imagenes_config, figsize=(15, 5), cmap='gray'):
    """
    Muestra imágenes en una cuadrícula horizontal.
    
    Parámetros:
    -----------
    imagenes_config : list[dict]
        Lista de diccionarios con la configuración de cada imagen:
        - 'imagen': ndarray, la imagen a mostrar
        - 'titulo': str, título de la imagen
        - 'texto': str (opcional), texto adicional debajo de la imagen
        - 'texto_pos': tuple (opcional), posición (x, y) del texto, por defecto (0.5, -0.1)
    figsize : tuple
        Tamaño de la figura (ancho, alto)
    cmap : str
        Mapa de colores para las imágenes
    
    Ejemplo:
    --------
    config = [
        {'imagen': img1, 'titulo': 'Original'}, 
        {'imagen': img2, 'titulo': 'Horizontal', 'texto': "flip(img, mode='horizontal')"},
        {'imagen': img3, 'titulo': 'Vertical', 'texto': "flip(img, mode='vertical')", 'texto_pos': (0.5, -0.15)}
    ]
    mostrar_imagenes(config)
    """
    n_imgs = len(imagenes_config)
    fig, axes = plt.subplots(1, n_imgs, figsize=figsize)
    
    # Si solo hay una imagen, axes no es una lista
    if n_imgs == 1:
        axes = [axes]
    
    for i, config in enumerate(imagenes_config):
        axes[i].imshow(config['imagen'], cmap=cmap)
        axes[i].set_title(config.get('titulo', ''), fontsize=12, fontweight='bold')
        axes[i].axis('off')
        
        # Agregar texto si está especificado
        if 'texto' in config:
            pos = config.get('texto_pos', (0.5, -0.1))
            axes[i].text(pos[0], pos[1], config['texto'], fontsize=10,
                        ha='center', va='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    plt.show()