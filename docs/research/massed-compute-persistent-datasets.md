# Massed Compute: datasets persistentes para entrenamiento

Fecha de revisión: 6 de agosto de 2026.

## Conclusión

Massed Compute documenta dos vías para no descargar un dataset en cada VM GPU:

1. **Block volume:** su guía de Ollama indica que montar un volumen antes de
   descargar conserva los artefactos tras recrear la VM. Es la opción más
   directa si el panel permite crear y adjuntar ese volumen a la VM GPU.
2. **VM CPU persistente + NFS:** es la alternativa completamente documentada
   paso a paso para almacenamiento persistente. Se mantiene una VM CPU con los
   datos y se monta su directorio en cada GPU efímera. Así se sigue pagando una
   VM CPU, pero no una GPU entre sesiones.

La segunda vía es la recomendación práctica mientras no se conozcan por escrito
el precio, rendimiento y ciclo de vida de los block volumes. Massed dice
explícitamente que este es el patrón para tener almacenamiento persistente sin
pagar continuamente una instancia GPU. La terminación destruye los datos de la
instancia; por ello no se debe terminar la VM CPU que aloja el NFS sin tener
una copia externa.

Fuentes primarias:

- [Guía de Massed para conservar modelos con block volumes](https://massedcompute.com/deploy-llm-ollama-gpu-cloud/)
- [Tutorial oficial de NFS entre instancias de Massed](https://vm-docs.massedcompute.com/docs/advanced-tutorials/nfs-mount/)
- [API de VM de Massed: crear y terminar instancias](https://vm-docs.massedcompute.com/api/v1)

## Diseño recomendado para este repositorio

Crear un block volume llamado, por ejemplo, `llm-from-scratch-v1`, si está
disponible en la consola. Si no, crear una VM CPU de almacenamiento y exportar
`/home/Ubuntu/data` por NFS, como indica Massed. En cualquiera de los dos
casos, conservar en el almacenamiento persistente:

```text
<VOLUME_ROOT>/
  data/pretrain-v1/          # manifests v3 y shards uint16 ya tokenizados
  checkpoints/               # checkpoints y best_validation.pt
  .cache/huggingface/        # descargas/cache de Hugging Face
  .cache/uv/                 # wheel y entornos cacheados por uv
  wandb/                     # logs locales de W&B
```

El dataset de la receta V1 ocupa aproximadamente 25 GB: son unos 12B tokens
en `uint16` (dos bytes por token), más manifests. Para tres checkpoints,
cachés y margen operativo, empezar con **100 GB**; usar **150 GB** si se
conservan varios ensayos o caches de fuentes. La estimación concuerda con la
guía operativa existente del repositorio para la misma receta
([`runpod-v1.md`](../training/runpod-v1.md)).

No es necesario conservar los Parquet de origen para entrenar de nuevo: la
salida reproducible del proyecto son los shards y sus `manifest.json` con
SHA-256. La cache de Hugging Face es conveniente para preparar una variante,
pero no sustituye a los shards tokenizados.

## Flujo operativo con NFS (documentado por Massed)

1. Lanzar una VM CPU y conservarla como servidor NFS. En ella, instalar
   `nfs-kernel-server`, crear `/home/Ubuntu/data`, exportarlo sólo a la IP de
   la GPU cliente con `rw,sync,no_subtree_check`, aplicar `exportfs -a` y
   permitir NFS desde esa IP en UFW. Estos son los pasos literales del tutorial
   oficial; no expongas el export a Internet ni uses un rango abierto.
2. Al lanzar una GPU, añadir su IP a `/etc/exports` y a la regla UFW del
   servidor CPU. En la GPU, instalar `nfs-common`, crear
   `/home/Ubuntu/storage` y montar:

   ```text
   <IP_SERVIDOR>:/home/Ubuntu/data  ->  /home/Ubuntu/storage
   ```

   El tutorial recomienda persistir el montaje en `fstab` con `nofail` y
   `nconnect=16`. Al terminar una GPU y crear otra, Massed indica retirar la
   IP antigua del export, autorizar la nueva y repetir el montaje en el nuevo
   cliente.
3. Preparar los datos una única vez con el NFS montado. Desde un clone efímero
   del repositorio, enlazar los artefactos a la raíz NFS y exportar las caches:

   ```bash
   export MC_STORAGE_ROOT=/home/Ubuntu/storage
   mkdir -p "$MC_STORAGE_ROOT"/{data,checkpoints,.cache/huggingface,.cache/uv,wandb}
   ln -s "$MC_STORAGE_ROOT/data" data
   ln -s "$MC_STORAGE_ROOT/checkpoints" checkpoints
   export HF_HOME="$MC_STORAGE_ROOT/.cache/huggingface"
   export UV_CACHE_DIR="$MC_STORAGE_ROOT/.cache/uv"
   export WANDB_DIR="$MC_STORAGE_ROOT/wandb"
   make prepare-data
   make validate-recipe
   ```

   Ejecutar `make prepare-data` sólo para la preparación inicial o para fuentes
   nuevas: el escritor rechaza deliberadamente sobrescribir shards publicados.
4. Para cada GPU nueva, montar el mismo NFS, crear un clone nuevo del commit de
   entrenamiento y repetir sólo los enlaces/exportaciones anteriores. Validar
   antes de gastar GPU:

   ```bash
   make validate-recipe
   make h100-train MAX_LEARNING_RATE=0.0006
   ```

5. Al terminar, destruir la VM GPU; conservar la VM CPU y copiar los
   checkpoints importantes a un segundo almacenamiento duradero independiente
   (por ejemplo, object storage). Un volumen persistente evita cold starts,
   pero una copia independiente protege contra borrado, errores de cuenta o
   cambios de proveedor.

Para un run largo, medir primero la velocidad con el ensayo A100: los shards
se abren mediante `numpy.memmap` y NFS puede convertirse en el cuello de
botella. Si ocurre, mantener los shards canónicos en NFS, copiarlos al NVMe
local al arrancar la GPU y comparar el tiempo de copia con el ahorro de
throughput. No hay especificaciones públicas de IOPS o throughput de este
camino.

## Flujo con block volume (si se ofrece en la consola)

La guía oficial confirma que un block volume montado antes de descargar
artefactos conserva esos archivos tras recrear la VM. Usar el mismo diseño de
directorios anterior en el punto de montaje real del volumen y repetir los
enlaces, sustituyendo `MC_STORAGE_ROOT` por esa ruta. La documentación pública
revisada no explica cómo crear, adjuntar ni montar el volumen para un dataset,
por lo que esos pasos deben seguir el panel o confirmarse con soporte antes de
subir datos grandes.

## Límites que hay que confirmar antes de cargar datos grandes

La documentación pública revisada no especifica precio por GB/mes, rendimiento
o IOPS, snapshots, cifrado, retención, límites de capacidad, política de
adjuntar/desadjuntar, ni la afinidad regional del block volume. Tampoco se debe
confundir la capacidad de `storage` que aparece en cada SKU con un volumen
persistente: la API la enumera como parte de la configuración de la instancia.

Confirmar con Massed estos puntos concretos:

- precio mensual del volumen, incluso desadjuntado;
- región del volumen y cuáles GPUs se pueden lanzar junto a él;
- ruta de montaje, permisos y si se selecciona antes de arrancar la VM;
- lectura secuencial y aleatoria sostenida para los `memmap` de PyTorch;
- si se puede adjuntar a más de una VM y el modo seguro de acceso;
- snapshots, backup y política de borrado.

La FAQ de Massed también aconseja object storage para datasets y checkpoints,
pero avisa que sus respuestas están generadas por IA y pueden contener errores;
no se ha usado como fuente contractual para este diseño.
