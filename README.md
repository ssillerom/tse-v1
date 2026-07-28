# LLM from scratch

Proyecto educativo para construir y entrenar un transformer decoder-only desde primeros
principios con PyTorch. El repositorio ya cubre el camino desde texto crudo hasta batches
causales, una arquitectura V1 probada y un loop de entrenamiento reanudable.

## Estado

El pipeline de datos ya permite:

- leer datasets de Hugging Face en streaming;
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

El módulo de entrenamiento añade AdamW configurable por el llamador, acumulación de gradientes
ponderada por tokens, gradient clipping, warmup lineal con cosine decay, evaluación periódica y
checkpoints atómicos que restauran modelo, optimizador, configuración y estado aleatorio. El
smoke test E2E recorre documentos locales, shards, dataset, entrenamiento, evaluación y
reanudación sin depender de la red.

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
- El entrenamiento actual es de un solo dispositivo y no usa mixed precision.
- La reproducción exacta del orden de datos al reanudar requiere un iterable determinista que
  pueda reiniciarse; todavía no se guarda el estado de samplers aleatorios o distribuidos.
- FineWeb-Edu sirve para validar el pipeline en inglés, no como corpus bilingüe final.

Los datasets preparados, checkpoints y logs son artefactos locales y no deben añadirse a Git.
