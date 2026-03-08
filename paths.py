"""Modulo para la gestion de rutas en el proyecto."""

from pyprojroot import here
from pathlib import Path
from typing import (
    Union,
    Callable,
    Iterable,
)


def make_dir_function(dir_name: Union[str, Iterable[str]]) -> Callable[..., Path]:
    """Función que convierte un string o iterable de strings 
    en una ruta relativa del proyecto.

    Args:
        dirname: Nombre de los subdirectorios para extender la ruta del proyecto principal.
            Si se pasa un iterable de strings como argumento, entonces se colapsa en
            una sola cadena con anclajes dependientes del sistema operativo.

    Returns:
        Una función que devuelve la ruta relativa a un directorio que puede
        recibir `n` número de argumentos para expansión.
    """

    def dir_path(*args) -> Path:
        if isinstance(dir_name, str):
            return here().joinpath(dir_name, *args)
        else:
            return here().joinpath(*dir_name, *args)

    return dir_path


"""Definición del directorio raíz del proyecto y subdirectorios importantes."""
project_dir = make_dir_function("")

for dir_type in [
    ["data"],
    ["fase1_aumento_de_datos"],
    ["fase2_filtrado_espacio"]
]:
    dir_var = "_".join(dir_type) + "_dir"
    exec(f"{dir_var} = make_dir_function({dir_type})")