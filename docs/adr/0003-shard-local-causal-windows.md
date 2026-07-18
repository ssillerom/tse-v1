# Ventanas causales locales al shard mediante memory mapping

El dataset abre los shards con `numpy.memmap` y presenta ventanas contiguas de `seq_len + 1`
tokens como pares desplazados de tensores `torch.long`. Las ventanas no cruzan fronteras de
shard: esto mantiene la indexación simple, permite acceso aleatorio sin cargar el corpus en RAM
y evita unir ficheros en cada worker, a cambio de descartar la cola que no complete una ventana
en cada shard.
