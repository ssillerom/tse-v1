# Partición train/validation mediante hash de contenido

Cuando la fuente no proporciona validación, cada documento válido se asigna mediante un hash
estable de su texto y una semilla, en lugar de usar el hash de Python o una decisión aleatoria
dependiente del orden. La partición puede reproducirse aunque cambie la ejecución y mantiene
duplicados exactos en el mismo split, a cambio de que la proporción solicitada sea aproximada en
muestras pequeñas.
