# LLM from scratch

Proyecto educativo para construir y entrenar un transformer decoder-only desde primeros
principios con PyTorch. El repositorio ya cubre el camino desde texto crudo hasta batches
causales, una arquitectura V1 probada y un loop de entrenamiento reanudable.

## Estado

El pipeline de datos ya permite:

- leer datasets de Hugging Face en streaming;
- extraer y filtrar registros anidados, como los archivos Python de Stack v3;
- tokenizar documentos con una codificación de `tiktoken` compatible con `uint16`;
- limitar una preparación por documentos y tokens;
- separar train y validation mediante un hash de contenido reproducible;
- validar y publicar shards y manifest de forma transaccional, con SHA-256 por shard;
- cargar shards mediante `numpy.memmap`;
- producir pares `(input_ids, targets)` para causal language modeling;
- inspeccionar las ventanas como IDs y texto decodificado.

El modelo V1 ya incluye embeddings, bloques Pre-Norm con atención MHA o GQA + RoPE, RMSNorm,
SwiGLU, pesos compartidos con el `lm_head` y causal cross-entropy. La generación incremental
conserva una KV cache compacta por capa. Un smoke test pequeño verifica formas, causalidad,
gradientes y que el modelo puede sobreajustar un batch dependiente del contexto.

El módulo de entrenamiento añade grupos AdamW con weight decay selectivo, acumulación de
gradientes ponderada por tokens, gradient clipping, warmup lineal con cosine decay o WSD,
evaluación periódica por fuente y checkpoints atómicos que restauran modelo, optimizador,
configuración,
estado aleatorio y posición exacta en los datos. Una receta versionada puede mezclar varias
fuentes y fases con un sampler determinista. El smoke test E2E recorre documentos locales,
shards, mezcla de datos, entrenamiento, evaluación y reanudación sin depender de la red.

## Instalación

`make setup` instala (si faltan) la versión fijada de [uv](https://docs.astral.sh/uv/), la
GitHub CLI y las dependencias bloqueadas del proyecto. También comprueba la sesión de GitHub:
usa `GH_TOKEN`/`GITHUB_TOKEN` si están definidos o inicia el flujo interactivo en un terminal.

```bash
make setup
export PATH="${UV_INSTALL_DIR:-$HOME/.local/bin}:$PATH"
```

Las recetas de Make ya anteponen automáticamente `UV_INSTALL_DIR`; el `export` sólo hace que
los comandos `uv` escritos directamente en este terminal también estén disponibles.

Si no quieres iniciar sesión en GitHub en ese equipo (la CLI se instalará igualmente):

```bash
make setup GH_AUTH=skip
export PATH="${UV_INSTALL_DIR:-$HOME/.local/bin}:$PATH"
```

## Prueba con 50 documentos de FineWeb-Edu

La preparación usa streaming: `sample-10BT` identifica la configuración del dataset, pero el
comando se detiene después de leer 50 documentos.

```bash
uv run prepare-data prepare \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --name sample-10BT \
  --split train \
  --text-field text \
  --output-dir data/fineweb-edu-50 \
  --num-tokens 10000000 \
  --shard-size 100000 \
  --workers 2 \
  --source-workers 2 \
  --max-docs 50 \
  --validation-ratio 0.1 \
  --split-seed 42 \
  --encoding gpt2
```

La salida tiene este contrato:

```text
data/fineweb-edu-50/
├── manifest.json
├── train/
│   └── shard_*.bin
└── validation/
    └── shard_*.bin
```

`--validation-ratio` es determinista pero aproximado: con 50 documentos no garantiza
exactamente 45/5. Los duplicados exactos se asignan siempre al mismo split. Cada documento
termina en EOT incluso cuando `--num-tokens` obliga a truncarlo: el último token de contenido
aceptado se sustituye por EOT.

`--workers` paraleliza la tokenización por lotes sin cambiar el orden recibido.
`--source-workers` reparte los shards remotos entre procesos para solapar descarga,
descompresión y decodificación. Para una preparación reproducible hay que fijar ambos valores:
el número de procesos de origen queda registrado en el manifest porque puede cambiar el orden
en que los shards de Hugging Face alcanzan el límite global de tokens.

Stack v3 usa `--source-reader duckdb` porque sus filas contienen listas anidadas de archivos
que pueden exceder la capacidad del lector Parquet de PyArrow. Este lector descarga
anticipadamente hasta `--source-workers` Parquet, los procesa en orden de nombre y elimina
cada copia temporal al terminarla. DuckDB entrega un archivo de código por fila; la
tokenización y el formato de salida son los mismos que con el lector predeterminado.

Durante `prepare`, `--log-every-docs` controla cada cuántos documentos se imprime una línea
compacta; las barras de progreso de las librerías de origen permanecen ocultas:

```text
Progress | tokens=148,784,753/700,000,000 (21.3%) | saved=100,000,000 | buffered=48,784,753 | shards=1
```

Al terminar se muestran tokens totales y por split, shards, contadores de documentos, tiempo
y throughput. Durante el progreso, `tokens` siempre equivale a `saved + buffered`: `saved` y
`shards` cuentan shards completos escritos en staging, mientras `buffered` son tokens todavía
en RAM. `Completed` solo aparece después de publicar atómicamente los shards y el manifest en
el directorio de destino.

La preparación nueva publica manifest v3. Cada entrada de `shards` incluye el SHA-256 del
fichero y `load_manifest()` lo verifica además del tamaño antes de abrir los datos. Los
manifest v2 existentes siguen siendo legibles, pero no ofrecen esta comprobación porque
nunca almacenaron checksums.

## Inspeccionar el slicing

```bash
uv run inspect-data-slices \
  --manifest data/fineweb-edu-50/manifest.json \
  --split train \
  --seq-len 128 \
  --num-examples 3
```

El comando muestra la ventana de `seq_len + 1` tokens, el input, el target desplazado, sus IDs y
la continuidad entre ventanas consecutivas.

## Entrenar y evaluar V1 con W&B

Autentica W&B una sola vez; la API key no debe guardarse en el repositorio:

```bash
uv run wandb login
```

El smoke training predeterminado usa el modelo educativo de 4 capas y evalúa sobre el split
`validation` del mismo manifest. Registra loss, perplexity, learning rate, gradient norm y una
tabla periódica con seis continuaciones greedy en inglés. También mide tokens procesados,
tiempo por step y tokens por segundo sin incluir el coste de evaluación:

```bash
uv run train-v1 \
  --manifest data/fineweb-edu-50/manifest.json \
  --device auto \
  --precision auto \
  --max-steps 500 \
  --eval-interval 50 \
  --sample-interval 100 \
  --checkpoint-interval 100 \
  --keep-last-checkpoints 3 \
  --wandb-project llm-from-scratch \
  --wandb-entity your-team \
  --wandb-name v1-smoke-001
```

La atención usa MHA cuando se omite `--n-kv-heads`. La receta de entrenamiento fija
`--n-heads 16 --n-kv-heads 4`: cada head K/V es compartido por cuatro heads Q (relación 4:1).
En general, `--n-kv-heads` debe dividir a `--n-heads`; el valor `1` selecciona MQA. La KV cache
compacta guarda sólo las cabezas K/V, y las muestras greedy y la generación del evaluation
harness la reconstruyen con los últimos tokens cuando alcanzan el límite de contexto.

Los prompts son fijos durante toda la ejecución para poder comparar checkpoints sin introducir
azar de sampling. La evaluación frecuente usa como máximo `--eval-batches 20`, seleccionados
de forma determinista y equidistante a lo largo de todo el split; no evalúa sólo su comienzo.
También evalúa el último step de cada segmento aunque no coincida con el intervalo. La
evaluación preserva el RNG global, por lo que observar el modelo no cambia futuros masks de
dropout ni seeds de workers. `--precision auto`
selecciona autocast BF16 cuando CUDA y la GPU lo soportan, y FP32 en los demás casos. El modelo
y los estados de AdamW se mantienen en FP32; BF16 reduce el coste de las operaciones del
forward y backward.

Con una recipe multifuente, W&B registra loss, perplexity y tokens evaluados para cada fuente,
además de un agregado ponderado con las proporciones completas de la recipe. `torch.compile`
está disponible en CUDA mediante `--compile`; los targets de entrenamiento de Make lo activan
por defecto con `--compile --compile-mode default`. Para una ejecución eager explícita, usa
`TORCH_COMPILE_ARGS=`; la elección debe conservarse idéntica al reanudar.

Por defecto se conservan sólo los tres checkpoints numerados con mayor step. Además,
`best_validation.pt` conserva siempre las ponderaciones con menor `validation/loss`; no está
sujeto a esa retención. Para continuar el mismo entrenamiento y la misma ejecución de W&B, usa
el directorio y la configuración originales:

```bash
uv run train-v1 \
  --manifest data/fineweb-edu-50/manifest.json \
  --checkpoint-dir checkpoints/v1 \
  --resume latest \
  --wandb-project llm-from-scratch
```

`--resume latest` prueba los checkpoints desde el step más reciente y, si encuentra uno
ilegible o incompatible, avisa y retrocede al anterior. También se puede pasar una ruta exacta
a `--resume`. El checkpoint restaura pesos, AdamW, step, configuración, RNG de CPU/CUDA/MPS y
la identidad completa de W&B (entity, proyecto e identificador). Al reanudar exige que ese run
ya exista, en vez de crear silenciosamente uno nuevo. Antes de continuar también valida el
contrato de datos (manifest o receta), seed, batch size y los hiperparámetros de AdamW que
determinan la posición de datos y la siguiente actualización. La
evaluación periódica y las muestras realizadas al abrir el run preservan el RNG restaurado.
Los checkpoints v7 guardan tanto su propia loss de validación como la mejor observada, además
de la configuración de heads K/V, de modo
que una reanudación no puede reemplazar `best_validation.pt` por un modelo peor. Como el
learning-rate schedule es una función del step y de la configuración guardada, no necesita un
objeto de scheduler separado.

En runs con receta, el checkpoint también conserva el presupuesto configurado y los tokens
realmente consumidos por cada fuente; la trazabilidad de la mezcla no depende de W&B.

Los checkpoints antiguos, incluido el formato v1, todavía se pueden abrir. Los formatos v1–v6
no guardaban `n_kv_heads` y se interpretan como MHA, por lo que conservan las formas de sus
pesos originales. Como v1 no guardaba el hash del manifest, el batch size ni la configuración
exacta del optimizador, la carga avisa de que no puede verificar esos datos antes de continuar.

Cada experimento debe usar un `--checkpoint-dir` exclusivo: la retención se aplica a todos los
archivos `step_*.pt` de ese directorio y nunca a `best_validation.pt`. Esto evita mezclar
estados pertenecientes a runs distintos.

Para probar el flujo completo sin iniciar sesión ni acceder a la red:

```bash
uv run train-v1 \
  --manifest data/fineweb-edu-50/manifest.json \
  --wandb-mode disabled
```

La receta propuesta para el run completo usa contexto 2.048, conserva un batch global de
524.288 tokens y, junto con los gates local, A100 y H100 SXM, está en
[V1 English 12B training run](docs/training/v1-english-12b.md). La ejecución reproducible en
RunPod está detallada en [Running the V1 recipe on RunPod](docs/training/runpod-v1.md) y
automatizada con `make help`.

La evaluación post-hoc mediante LM Evaluation Harness, las métricas por dominio y el baseline
FineWeb-only están descritos en [Evaluation protocol](docs/training/evaluation.md). El harness
se instala como extra opcional con `uv sync --extra eval`.

## Documentación

- [Receta V1 English 12B](docs/training/v1-english-12b.md): arquitectura, mezcla de datos,
  gates y entrenamiento con contexto 2.048.
- [Ejecución en RunPod](docs/training/runpod-v1.md): preparación reproducible, rehearsal en
  A100 y ejecución final en H100 SXM.
- [Protocolo de evaluación](docs/training/evaluation.md): validación durante el pretraining,
  baseline controlado de 300M y benchmarks post-hoc.
- [Recetas públicas de entrenamiento](docs/research/open-llm-training-recipes.md): referencias
  para comparar transparencia y reproducibilidad.

## Créditos

Los scripts de preparación, inspección, entrenamiento y evaluación son obra de Sergio Sillero.

## Desarrollo

```bash
uv run pytest -q
uv run ruff format --check src
uv run ruff check src
uv run mypy
```

Los tests de red no forman parte de la suite. Las pruebas sustituyen la fuente remota por
documentos pequeños y escriben shards reales en directorios temporales.

## Arquitectura de datos

- [Lenguaje del dominio](CONTEXT.md)
- [ADR 0001: shards binarios y manifest versionado](docs/adr/0001-versioned-binary-token-shards.md)
- [ADR 0002: partición determinista por contenido](docs/adr/0002-content-hash-data-partitioning.md)
- [ADR 0003: ventanas causales locales al shard](docs/adr/0003-shard-local-causal-windows.md)

## Limitaciones actuales

- Solo se admiten tokenizadores de `tiktoken` cuyos IDs quepan en `uint16`.
- La preparación paralela conserva el orden por lote, pero todavía no puede reanudarse a mitad
  de un dataset.
- Las secuencias no cruzan fronteras de shard; la cola incompleta de cada shard se descarta.
- El entrenamiento actual es de un solo dispositivo; CUDA usa BF16 con
  `--precision auto` cuando el hardware lo soporta.
- El sampler multifuente incluido es determinista y reanudable, pero todavía no existe un
  sampler distribuido para entrenamiento multi-GPU.
- FineWeb-Edu sirve para validar el pipeline en inglés, no como corpus bilingüe final.

Los datasets preparados, checkpoints y logs son artefactos locales y no deben añadirse a Git.
