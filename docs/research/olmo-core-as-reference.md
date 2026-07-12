# OLMo-core como referencia para `llm-from-scratch`

Fecha de consulta: 2026-07-12. Revisión analizada de OLMo-core: [`fa6c501`](https://github.com/allenai/OLMo-core/tree/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d).

## Conclusión

Sí: **OLMo-core es probablemente una referencia más adecuada que Megatron-LM para este proyecto**, siempre que se copie su arquitectura conceptual y no su volumen de infraestructura.

La idea central que conviene adoptar es esta:

```text
receta de experimento
        │
        ├── construye modelo
        ├── construye módulo de entrenamiento
        ├── construye dataset + dataloader
        └── construye trainer + callbacks
                         │
                         └── fit()
```

OLMo-core se presenta como una biblioteca de *building blocks* estables para modelado y entrenamiento, centrada en un `Trainer` flexible, un transformer optimizado y un sistema de lanzamiento separado ([introducción oficial](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/docs/source/overview/introduction.rst)). No es solo «un repositorio con un `train.py`»: es una biblioteca reusable más recetas ejecutables.

Para `llm-from-scratch`, la traducción adecuada sería una versión mucho menor:

- `nn/` contiene las matemáticas del transformer.
- `data/` produce batches reproducibles y guarda su posición.
- `train/` contiene el loop, el estado y el checkpointing.
- `scripts/` contiene recetas concretas y legibles: debug, tiny, small y 350M.
- `eval/` y `generate/` usan el modelo sin depender del trainer.
- `tests/` verifica matemáticas, resume y convergencia.

No conviene copiar todavía su soporte para Beaker, almacenamiento remoto, MoE, TP/CP/PP/EP, FP8, múltiples backends Flash, callbacks empresariales ni compatibilidad histórica de checkpoints.

## Qué es realmente OLMo-core

El árbol público actual separa cinco capas principales:

```text
src/
├── olmo_core/
│   ├── config.py
│   ├── nn/
│   │   ├── attention/
│   │   ├── transformer/
│   │   ├── rope.py
│   │   ├── layer_norm.py
│   │   ├── feed_forward.py
│   │   └── lm_head.py
│   ├── data/
│   │   ├── numpy_dataset.py
│   │   ├── data_loader.py
│   │   ├── tokenizer.py
│   │   ├── source_mixture.py
│   │   └── composable/
│   ├── optim/
│   ├── distributed/
│   │   ├── parallel/
│   │   └── checkpoint/
│   ├── train/
│   │   ├── trainer.py
│   │   ├── train_module/
│   │   ├── checkpoint.py
│   │   └── callbacks/
│   ├── eval/
│   ├── generate/
│   ├── launch/
│   └── testing/
├── examples/
├── scripts/
│   ├── official/
│   ├── train/
│   └── data/
├── test/
└── integration_tests/
```

Este árbol puede comprobarse en el [paquete oficial](https://github.com/allenai/OLMo-core/tree/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core), los [scripts](https://github.com/allenai/OLMo-core/tree/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/scripts), los [tests unitarios](https://github.com/allenai/OLMo-core/tree/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/test) y los [tests de integración](https://github.com/allenai/OLMo-core/tree/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/integration_tests).

Una observación importante: OLMo-core **no usa una carpeta central de YAML como mecanismo principal para las recetas**. Las recetas oficiales y los ejemplos son scripts Python que componen dataclasses de configuración y luego aceptan overrides por línea de comandos. El README muestra, por ejemplo, `--train_module.optim.lr=6e-3` sobre un script oficial ([README oficial](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/README.md)).

## Configuración y recetas

### Patrón de OLMo-core

`Config` es una base para dataclasses serializables. Permite:

- convertir configs anidadas en diccionarios JSON-safe;
- reconstruirlas desde JSON/YAML;
- validar y copiar;
- aplicar overrides con rutas punteadas, como `trainer.max_duration` o `train_module.optim.lr`.

La implementación está en [`olmo_core/config.py`](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/config.py), con tests específicos de serialización, configs anidadas y merge en [`config_test.py`](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/test/config_test.py).

El ejemplo docente define una `ExperimentConfig` compuesta por:

```python
@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    init_seed: int
```

Luego construye los componentes de forma explícita:

```python
model = config.model.build(init_device="meta")
train_module = config.train_module.build(model)
dataset = config.dataset.build()
data_loader = config.data_loader.build(
    dataset,
    dp_process_group=train_module.dp_process_group,
)
trainer = config.trainer.build(train_module, data_loader)
trainer.fit()
```

Este flujo está en el [ejemplo oficial de entrenamiento LLM](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/examples/llm/train.py) y se explica en la [guía all-in-one](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/docs/source/guides/all_in_one_for_researchers.md).

Las configuraciones de modelo ofrecen factories con nombres y tamaños conocidos. OLMo-core incluye peldaños de 30M, 60M, 100M, 190M y 370M, entre otros, en [`TransformerConfig`](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/nn/transformer/config.py). Para un proyecto que aspira a 350M, esta escalera es una referencia especialmente útil.

### Qué copiar

- Una `ExperimentConfig` raíz con subconfigs tipadas.
- Factories Python pequeñas: `debug()`, `tiny()`, `small()` y `model_350m()`.
- Una receta ejecutable por experimento importante.
- Overrides CLI con dot notation, si se pueden implementar sin oscurecer el código.
- Guardar siempre la configuración resuelta junto al checkpoint y las métricas.
- Un modo `--dry-run` que construya todo, imprima la config y valide incompatibilidades.

### Qué cambiar respecto a la propuesta anterior

La propuesta previa colocaba gran parte de la configuración en `configs/model/*.yaml`, `configs/data/*.yaml`, `configs/runtime/*.yaml` y `configs/experiments/*.yaml`. Una estructura más fiel a OLMo-core —y más pedagógica para este caso— sería:

- dataclasses en el paquete;
- factories de tamaños en `model/config.py`;
- recetas Python completas en `scripts/train/`;
- opcionalmente YAML solo para manifests de datos o ejecuciones externas.

Esto evita construir demasiado pronto un sistema de composición YAML y mantiene visibles las decisiones del experimento en código navegable y tipado.

## Modelo y operaciones matemáticas

OLMo-core separa sus primitivas en `nn/`: attention, RoPE, normas, feed-forward, LM head y bloques transformer. `TransformerConfig.build()` crea el módulo, mientras el constructor de `Transformer` ensambla embeddings, bloques y LM head ([configuración](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/nn/transformer/config.py), [modelo](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/nn/transformer/model.py)).

La atención distingue la proyección/modelado de QKV del backend que ejecuta SDPA. El enum de backends puede construir implementación Torch, Flash Attention o Transformer Engine, y expone explícitamente las capacidades soportadas ([backend oficial](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/nn/attention/backend.py)). Sus tests comparan backends optimizados contra Torch SDPA y cubren dtypes, MHA/MQA/GQA, RoPE, QK norm, ventanas y máscaras ([tests de attention](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/test/nn/attention/attention_test.py)).

### Qué copiar

- Un módulo por concepto matemático.
- `ModelConfig` sin opciones de logging, datasets o launch.
- Un backend de atención seleccionable.
- Presets de tamaño como factories, no diccionarios mágicos dispersos.
- Tests de equivalencia entre atención de referencia y SDPA.

### Qué simplificar

Mantener solo dos backends:

1. `reference`: implementación propia, clara y lenta.
2. `sdpa`: `torch.nn.functional.scaled_dot_product_attention`.

OLMo-core ya está orientado a producción y no conserva una atención escolar completamente ingenua como oráculo. En este proyecto sí merece la pena conservarla porque aprender y detectar causal leakage es parte del objetivo.

Tampoco hace falta reproducir su sistema genérico de `SequenceMixer`, múltiples tipos de bloque, sliding-window attention, MoE, recurrent layers, kernels propios ni conversores Hugging Face.

## `TrainModule`: la frontera más interesante

La versión actual de OLMo-core no entrega directamente `model + optimizer` al trainer. Introduce `TrainModule`, una interfaz que encapsula:

- forward/backward de un batch;
- microbatching y gradient accumulation;
- cálculo de loss;
- optimizer step y zero grad;
- estado de modelo y optimizador;
- preparación distribuida del modelo;
- evaluación de batches;
- estimación de FLOPs.

La interfaz puede verse en [`train_module.py`](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/train/train_module/train_module.py). La variante transformer recibe optimizer, scheduler, precisiones, gradient clipping, compilación y configs de DP/TP/CP/EP/PP/activation checkpointing en [`TransformerTrainModuleConfig`](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/train/train_module/transformer/config.py). Es el módulo de entrenamiento, no la receta, quien aplica el paralelismo y construye el optimizador.

Esta frontera mantiene al `Trainer` relativamente agnóstico respecto al modelo. El trainer coordina pasos, estado, callbacks, métricas, cancelación y checkpoints; el train module sabe entrenar una unidad concreta.

### Recomendación para este proyecto

Copiar el concepto, con un nombre y contrato mínimos, cuando comience el entrenamiento real:

```python
class PretrainModule:
    model: Transformer
    optimizer: Optimizer

    def train_batch(self, batch) -> StepMetrics: ...
    def eval_batch(self, batch) -> EvalOutput: ...
    def optimizer_step(self) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
```

No crear todavía una jerarquía de clases ni una interfaz abstracta general para modalidades o modelos inexistentes. Una clase concreta basta. Si más adelante aparecen SFT, multimodal o pipeline parallel, el seam ya existe y puede generalizarse con evidencia.

## Datos

OLMo-core pretokeniza los datos como arrays NumPy 1D de token IDs. Sobre ellos ofrece datasets FSL/VSL, concatenación y chunking, padding, packing, mezclas y una API componible. La guía oficial explica el formato y exige que un dataloader custom sea distribuido, determinista y stateful ([guía de data loading](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/docs/source/guides/data_loading.rst)). La mezcla de fuentes puede definirse por ratios, manifests catalogados o simples globs ([guía oficial de data mixing](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/docs/source/guides/data_mixing.rst)).

El detalle profesional más valioso está en el estado. El dataloader guarda fingerprint y versión del dataset, batches y tokens procesados, seed y epoch; al cargar valida que los datos no hayan cambiado y restaura el orden ([implementación oficial](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/data/data_loader.py)).

### Qué copiar

- Shards pretokenizados y memory-mapped.
- Manifest versionado con paths, token count, dtype, tokenizer id/hash y checksum o fingerprint.
- Batch size expresado en tokens globales.
- Dataloader determinista con `state_dict()` y `load_state_dict()`.
- Posición exacta, epoch, seed/shuffle y fingerprint dentro del checkpoint.
- Separar preparación/tokenización offline de lectura durante entrenamiento.

### Qué simplificar

- Empezar solo con fixed sequence length y concatenate-and-chunk.
- Un único corpus o lista de shards; no ratios ni catálogo de mixes al principio.
- `uint16` si el vocabulario cabe, o `uint32`; documentar la elección.
- Sin VSL, OBFD packing, intra-document masking ni composable sources.

## Entrenamiento, callbacks y observabilidad

`Trainer.fit()` se ocupa de cargar automáticamente el checkpoint más reciente, instalar handlers de señales, ejecutar callbacks, hacer un dry run para detectar OOM/errores tempranos, recorrer epochs y cerrar recursos de forma coordinada ([trainer oficial](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/train/trainer.py)).

Su callback API cubre consola, velocidad, memoria, checkpoints, profiler, W&B/Comet, evaluadores, estabilidad, configuración, garbage collection y otras preocupaciones ([directorio de callbacks](https://github.com/allenai/OLMo-core/tree/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/train/callbacks)).

### Qué copiar

- El loop coordina; la matemática y el I/O viven en otros módulos.
- `global_step` y `tokens_seen` como identidades distintas.
- Métricas por paso: loss, learning rate, grad norm, tokens/s, tiempo de datos y memoria.
- Dry run antes de iniciar una ejecución cara.
- Hooks pequeños o callbacks para logging y checkpoints, si evitan condicionales dentro del loop.
- Señal de terminación que permita un checkpoint final seguro en single-node.

### Qué simplificar

No crear veinte callbacks. Inicialmente bastan tres observadores:

- `ConsoleLogger` / JSONL;
- `Checkpointer`;
- `Evaluator`.

W&B puede ser un adaptador opcional. Garbage collection manual, Slack, Comet, GAP monitoring, model merging, HF conversion y bookkeeping asíncrono deben esperar.

## Distribución

OLMo-core modela DP, TP, CP, PP y EP como ejes de un `DeviceMesh`, con validación de grados y compatibilidades ([malla distribuida](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/distributed/parallel/__init__.py)). `TransformerTrainModule` construye esa malla y aplica las transformaciones al modelo, separando la receta de la mecánica distribuida ([implementación](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/train/train_module/transformer/train_module.py)).

Conviene matizar que el modelo de OLMo-core no es un módulo «puro» completamente ajeno a distribución: su clase `Transformer` incluye métodos y estado para TP/FSDP/PP. Sin embargo, el constructor single-device y la selección de arquitectura están separados de la decisión de paralelizar, que se activa desde el train module.

### Recomendación para 350M

- Fase 1: single device.
- Fase 2: DDP, solo si se usan varias GPUs para throughput.
- FSDP únicamente si memoria de parámetros/optimizer/activaciones lo exige en el hardware elegido.
- TP, PP y CP no son necesarios para un modelo de 350M y harían más difícil aprender y depurar.

El principio que sí debe copiarse desde el principio es que el código de arquitectura no lea `RANK`, no inicialice process groups y no decida su topología.

## Checkpointing y reanudación

OLMo-core separa dos estados:

1. `TrainModule`: modelo y optimizador mediante PyTorch Distributed Checkpoint.
2. `Trainer`: step, tokens vistos, FLOPs, max steps, epoch, dataloader, world size, RNG y estado de callbacks.

El estado exacto del trainer está definido en [`TrainerStateDict`](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/train/trainer.py). `Checkpointer` guarda estado de trainer por rank, modelo/optimizer aparte y metadata solo al terminar correctamente; también soporta guardado async y storage remoto ([implementación oficial](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/train/checkpoint.py)). Los tests ejercitan guardado/carga local, distribuido y async ([tests oficiales](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/test/train/checkpoint_test.py)).

### Qué copiar desde el primer entrenamiento serio

```text
checkpoint/
├── metadata.json
├── config.json
├── model.pt
├── optimizer.pt
└── trainer.pt
```

`trainer.pt` debería contener como mínimo:

```text
step
tokens_seen
epoch / shard / offset
dataloader state + dataset fingerprint
scheduler state
Python RNG
PyTorch CPU RNG
CUDA RNG por dispositivo
```

El guardado debe escribirse en un directorio temporal y hacerse visible mediante rename/marker al completarse. El loader solo debe considerar checkpoints completos.

### Qué posponer

- PyTorch Distributed Checkpoint si se entrena single-GPU.
- Shards de checkpoint por rank.
- Guardado asíncrono.
- Carga con world size distinto.
- S3/GCS y pre-download.
- Reshard/unshard y key mappings históricos.

La semántica exacta de resume no debe posponerse: un test tiene que demostrar que una ejecución interrumpida continúa con el mismo batch siguiente y el mismo resultado que una ejecución sin interrupción.

## Evaluación y generación

OLMo-core mantiene `eval/` separado de `train/`. Un `Evaluator` controla batches y métricas, y `LMEvaluator` calcula CE y perplexity por dataset; los callbacks lo insertan dentro del loop ([evaluator base](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/eval/evaluator.py), [LM evaluator](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/eval/lm_evaluator.py)). `generate/` es otro paquete con sampling y KV cache, separado del trainer ([generación oficial](https://github.com/allenai/OLMo-core/tree/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/olmo_core/generate)).

Para este proyecto:

- implementar primero validation loss/perplexity determinista;
- mantener `generate` como entrypoint separado;
- añadir temperature y top-k antes que un framework de benchmarks;
- posponer evals downstream, harnesses externos y KV cache hasta que el pretraining funcione.

## Tests

OLMo-core coloca tests junto al layout del paquete (`src/test/...`) y mantiene integración aparte. La suite cubre config, datos, distribución, modelo, kernels, optimizadores, trainer, callbacks, checkpointing, conversión y generación. Su integración entrena un modelo OLMo3 de 30M durante unos pasos y verifica el checkpoint resultante ([test de integración oficial](https://github.com/allenai/OLMo-core/blob/fa6c5014c9f6e9ee789da2d9c20d5126fee8df0d/src/integration_tests/test_train_small_model.py)).

La jerarquía educativa recomendada sigue siendo algo más explícita:

```text
tests/
├── unit/
│   ├── nn/
│   ├── data/
│   └── train/
├── integration/
│   ├── test_overfit_one_batch.py
│   ├── test_checkpoint_resume.py
│   └── test_train_smoke.py
└── convergence/
    └── test_tiny_reference_run.py
```

No basta con copiar los tests operativos de OLMo-core. Por el objetivo pedagógico deben mantenerse pruebas propias como:

- no future leak;
- fórmula directa de RMSNorm;
- RoPE preserva normas;
- atención ingenua y SDPA coinciden en forward y backward;
- weight tying comparte storage;
- loss inicial próxima a `log(vocab_size)`;
- overfit de un batch;
- resume bitwise o numéricamente determinista;
- curva tiny que alcanza una loss de referencia.

## Estructura propuesta, inspirada en OLMo-core

```text
llm-from-scratch/
├── README.md
├── MISSION.md
├── pyproject.toml
├── uv.lock
│
├── src/
│   └── llmfs/
│       ├── config.py
│       ├── nn/
│       │   ├── attention.py
│       │   ├── attention_backend.py
│       │   ├── rope.py
│       │   ├── rms_norm.py
│       │   ├── feed_forward.py
│       │   ├── block.py
│       │   └── transformer.py
│       ├── data/
│       │   ├── tokenizer.py
│       │   ├── manifest.py
│       │   ├── dataset.py
│       │   ├── data_loader.py
│       │   └── prepare.py
│       ├── optim/
│       │   ├── adamw.py
│       │   └── scheduler.py
│       ├── train/
│       │   ├── config.py
│       │   ├── pretrain_module.py
│       │   ├── trainer.py
│       │   ├── state.py
│       │   ├── checkpoint.py
│       │   └── callbacks.py
│       ├── distributed/
│       │   └── ddp.py
│       ├── eval/
│       │   └── language_modeling.py
│       ├── generate/
│       │   ├── sampling.py
│       │   └── generate.py
│       └── testing/
│           └── assertions.py
│
├── scripts/
│   ├── train/
│   │   ├── debug.py
│   │   ├── tiny.py
│   │   ├── small.py
│   │   └── model_350m.py
│   ├── data/
│   │   ├── train_tokenizer.py
│   │   └── prepare_shards.py
│   └── inspect/
│       ├── checkpoint.py
│       └── data_shard.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── convergence/
│   └── fixtures/
│
├── docs/
│   ├── decisions/
│   ├── research/
│   └── training-runs/
│
└── runs/                 # gitignored
```

### Reglas de dependencias

```text
scripts ───────────────┐
                      v
trainer ──> pretrain_module ──> nn
   │             │
   ├────────────> data
   ├────────────> eval
   └────────────> checkpoint

generate ────────────────────> nn
```

- `nn` no importa `train`, `data`, W&B ni CLI.
- `data` no importa el trainer.
- `generate` no importa el trainer.
- `checkpoint` serializa interfaces/estado; no decide cuándo guardar.
- los scripts componen componentes, pero no implementan sus algoritmos.

## Qué implementar ahora, después y nunca por defecto

| Ahora | Al llegar al entrenamiento 10–30M | Solo al escalar a 350M | Posponer indefinidamente |
|---|---|---|---|
| `nn/` legible | `PretrainModule` | bf16 + SDPA + compile | TP/PP/CP/EP |
| configs tipadas | trainer stateful | DDP si hay varias GPUs | MoE y FP8 |
| receta `debug.py` | shards mmap | checkpoint robusto | Beaker/Slurm abstractions |
| attention reference | validation periódica | profiling/throughput | storage remoto |
| tests matemáticos | resume determinista | recipe `model_350m.py` | registries genéricos |
| overfit one batch | JSONL/W&B opcional | smoke test en GPU objetivo | compatibilidad de formatos histórica |

## Decisión recomendada

Adoptar una **“OLMo-core educativa”**, con dos ajustes respecto a la propuesta previa:

1. Usar recetas Python tipadas como fuente de verdad, en lugar de empezar con una jerarquía amplia de YAML.
2. Introducir una clase concreta `PretrainModule` entre `Trainer` y `Transformer`, pero no una framework abstracta de módulos de entrenamiento.

El resultado conserva lo mejor de OLMo-core —composición explícita, estado reproducible, fronteras model/data/train, recetas versionadas y tests por subsistema— sin esconder las matemáticas bajo una infraestructura de frontier lab que un modelo de 350M no necesita.
