# Datasets tokenizados como artefactos de entrenamiento

Fecha de revisión: 11 de agosto de 2026.

## Conclusión

Sí: conviene preparar y tokenizar una vez en una máquina CPU barata o persistente,
publicar el resultado como un artefacto inmutable y hacer que cada trabajo GPU lo
descargue o lo cachee en disco local antes de entrenar. Hugging Face Hub sirve para
esto, especialmente para versionado y reproducibilidad, pero el `streaming=True` de
🤗 Datasets no equivale a hacer `numpy.memmap` remoto de cualquier fichero binario.

Para este repositorio, el patrón más sencillo es:

```text
fuente HF -> preparación CPU -> manifest.json + shards uint16
                              -> HF Hub / S3 / volumen persistente
GPU efímera -> bootstrap -> cache o NVMe local -> entrenamiento con memmap
```

El entrenamiento no debería depender de tokenizar ni de leer cada muestra por HTTP.
El trabajo remoto puede entregar shards bajo demanda, pero debe descargarlos a una
cache local de tamaño controlado; este es el modelo de Mosaic Streaming.

## Qué permite Hugging Face

- Los repositorios de datasets del Hub están versionados y admiten revisiones. El Hub
  usa Xet para ficheros binarios grandes y también ofrece Storage Buckets para
  almacenamiento tipo S3 sin historial Git:
  <https://huggingface.co/docs/hub/en/repositories>
- Las librerías de Hugging Face pueden leer ficheros remotos y hacer streaming desde
  formatos soportados. Parquet permite leer por grupos de filas y el Hub documenta
  también peticiones HTTP de rango:
  <https://huggingface.co/docs/hub/en/datasets-streaming>
- `snapshot_download()` permite fijar un commit, descargar un repositorio de dataset
  y dejarlo en una carpeta local; el sistema mantiene una cache direccionada por
  contenido y no vuelve a descargar ficheros sin cambios:
  <https://huggingface.co/docs/huggingface_hub/en/guides/download>
  <https://huggingface.co/docs/huggingface_hub/en/local-cache>
- Para repositorios grandes, Hugging Face recomienda dividir ficheros, mantener menos
  de 100.000 archivos y usar `upload_folder()`/`hf upload` para subidas reanudables:
  <https://huggingface.co/docs/hub/storage-limits>
  <https://huggingface.co/docs/huggingface_hub/guides/upload>

## Qué hacen los stacks de entrenamiento

- GPT-NeoX/Megatron preprocesan el texto fuera del loop de entrenamiento y generan
  ficheros binarios indexados (`.bin` y `.idx`); su implementación ofrece una variante
  `mmap` para leerlos durante el entrenamiento:
  <https://github.com/EleutherAI/gpt-neox#datasets>
- OLMo-core exige pre-tokenizar a arrays NumPy de IDs y después ofrece dataloaders para
  esos ficheros. El README de OLMo documenta que sus configuraciones pueden leer por
  HTTP, pero recomienda descargar los ficheros localmente para reproducir a gran escala:
  <https://olmo-core.readthedocs.io/en/latest/guides/data_loading.html>
  <https://github.com/allenai/OLMo#pretraining>
- Mosaic Streaming representa el patrón explícito de object storage más cache local:
  `remote` contiene los shards, `local` es el directorio de cache y los shards se
  descargan cuando hacen falta. Su FAQ aclara que no es un flujo continuo de bytes,
  sino descarga de ficheros shard; `cache_limit` permite expulsar shards antiguos:
  <https://docs.mosaicml.com/projects/streaming/en/stable/dataset_configuration/shard_retrieval.html>
  <https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/faqs_and_tips.html>

La pauta común es, por tanto, **preprocesamiento offline + formato de lectura eficiente
en disco + almacenamiento remoto versionado + staging/cache local**. El streaming puro
por HTTP se usa para prototipos o para ocultar el arranque, no como sustituto general de
un filesystem local de alto rendimiento.

## Aplicación al repositorio

Los shards actuales son `uint16` y contienen 100 millones de tokens en la receta
operativa: aproximadamente 200 MB por shard sin comprimir. Un corpus de 12B tokens
ocupa unos 24 GB de bytes de tokens, más manifest y directorios de splits. El lector
actual valida que los shards existan junto al manifest, verifica su SHA-256 y los abre
con `numpy.memmap`; por diseño, necesita rutas locales.

Por eso hay tres niveles de solución:

1. **Solución inmediata:** subir cada dataset preparado completo (`manifest.json`,
   `train/` y `validation/`) a un repositorio de dataset privado o público, fijar el
   commit y ejecutar `snapshot_download()` al arrancar la GPU hacia un volumen/NVMe
   local. El entrenamiento sigue usando exactamente el lector actual.
2. **Solución económica y repetible:** conservar la misma copia en un volumen
   persistente o en object storage cercano a la región de las GPUs y descargarla sólo
   cuando la cache local esté vacía. Si se reutiliza la máquina o el volumen, el coste de
   arranque desaparece casi por completo.
3. **Solución avanzada:** añadir un `RemoteShardCache` que descargue un shard completo
   cuando vaya a usarse, lo valide por SHA-256, lo conserve en cache local y lo expulse
   con LRU. No conviene hacer una petición HTTP por muestra. Este cambio debe preservar
   la semántica del manifest y la reanudación determinista.

No es recomendable convertir directamente los shards binarios actuales a filas Parquet
si el objetivo principal es el throughput del entrenamiento: Parquet facilita el
streaming tabular, pero añade decodificación y estructura de filas. Puede ser una buena
salida interoperable para compartir documentos tokenizados, mientras que los shards
`uint16`/memmap siguen siendo el formato de ejecución.

