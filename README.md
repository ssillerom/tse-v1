# LLM from scratch

Proyecto educativo para construir y entrenar un transformer decoder-only desde primeros
principios con PyTorch. El objetivo actual es completar primero un camino fiable desde texto
crudo hasta batches causales antes de implementar el modelo y el trainer.

## Estado

El pipeline de datos ya permite:

- leer datasets de Hugging Face en streaming;
- tokenizar documentos con una codificación de `tiktoken` compatible con `uint16`;
- limitar una preparación por documentos y tokens;
- separar train y validation mediante un hash de contenido reproducible;
- publicar shards y manifest de forma transaccional;
- cargar shards mediante `numpy.memmap`;
- producir pares `(input_ids, targets)` para causal language modeling;
- inspeccionar las ventanas como IDs y texto decodificado.

El modelo transformer, el loop de entrenamiento y los checkpoints todavía están fuera del
camino ejecutable actual.

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
uv run pytest -q src/test/data
uv run ruff format --check src/data src/scripts src/test/data
uv run ruff check src/data src/scripts src/test/data
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
- FineWeb-Edu sirve para validar el pipeline en inglés, no como corpus bilingüe final.

Los datasets preparados, checkpoints y logs son artefactos locales y no deben añadirse a Git.
