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
- validar y publicar shards y manifest de forma transaccional;
- cargar shards mediante `numpy.memmap`;
- producir pares `(input_ids, targets)` para causal language modeling;
- inspeccionar las ventanas como IDs y texto decodificado.

El modelo V1 ya incluye embeddings, bloques Pre-Norm con MHA + RoPE, RMSNorm, SwiGLU, pesos
compartidos con el `lm_head` y causal cross-entropy. Un smoke test pequeño verifica formas,
causalidad, gradientes y que el modelo puede sobreajustar un batch dependiente del contexto.

El módulo de entrenamiento añade grupos AdamW con weight decay selectivo, acumulación de
gradientes ponderada por tokens, gradient clipping, warmup lineal con cosine decay o WSD,
evaluación periódica y checkpoints atómicos que restauran modelo, optimizador, configuración,
estado aleatorio y posición exacta en los datos. Una receta versionada puede mezclar varias
fuentes y fases con un sampler determinista. El smoke test E2E recorre documentos locales,
shards, mezcla de datos, entrenamiento, evaluación y reanudación sin depender de la red.

## Instalación

Requiere Python 3.13 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
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
exactamente 45/5. Los duplicados exactos se asignan siempre al mismo split.

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

Los prompts son fijos durante toda la ejecución para poder comparar checkpoints sin introducir
azar de sampling. La evaluación frecuente usa como máximo `--eval-batches 20`; el checkpoint
final se guarda incluso si el último step no coincide con el intervalo. `--precision auto`
selecciona autocast BF16 cuando CUDA y la GPU lo soportan, y FP32 en los demás casos. El modelo
y los estados de AdamW se mantienen en FP32; BF16 reduce el coste de las operaciones del
forward y backward.

Por defecto se conservan sólo los tres checkpoints con mayor step. Para continuar el mismo
entrenamiento y la misma ejecución de W&B, usa el directorio y la configuración originales:

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
evaluación y las muestras realizadas al abrir el run preservan el RNG restaurado. Como el
learning-rate schedule es una función del step y de la configuración guardada, no necesita un
objeto de scheduler separado.

Los checkpoints antiguos de formato v1 todavía se pueden abrir. Como no guardaban el hash del
manifest, el batch size ni la configuración exacta del optimizador, la carga avisa de que no
puede verificar esos datos antes de continuar.

Cada experimento debe usar un `--checkpoint-dir` exclusivo: la retención se aplica a todos los
archivos `step_*.pt` de ese directorio. Esto evita mezclar estados pertenecientes a runs
distintos.

Para probar el flujo completo sin iniciar sesión ni acceder a la red:

```bash
uv run train-v1 \
  --manifest data/fineweb-edu-50/manifest.json \
  --wandb-mode disabled
```

La receta propuesta para el run completo, junto con los gates local, A100 y H100 SXM, está en
[V1 English 12B training run](docs/training/v1-english-12b.md). La ejecución reproducible en
RunPod está detallada en [Running the V1 recipe on RunPod](docs/training/runpod-v1.md) y
automatizada con `make help`.

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
- Los shards no incluyen todavía checksums ni reanudación de una preparación interrumpida.
- La tokenización es de un solo proceso.
- Las secuencias no cruzan fronteras de shard; la cola incompleta de cada shard se descarta.
- El entrenamiento actual es de un solo dispositivo; CUDA usa BF16 con
  `--precision auto` cuando el hardware lo soporta.
- El sampler multifuente incluido es determinista y reanudable, pero todavía no existe un
  sampler distribuido para entrenamiento multi-GPU.
- FineWeb-Edu sirve para validar el pipeline en inglés, no como corpus bilingüe final.

Los datasets preparados, checkpoints y logs son artefactos locales y no deben añadirse a Git.
