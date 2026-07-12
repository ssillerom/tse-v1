# Cómo estructuran públicamente el entrenamiento LLM a escala

Fecha de consulta: 2026-07-12.

## Alcance y cautela

Este documento compara **código y documentación públicos y oficiales**. No describe necesariamente los repositorios internos con los que Meta, NVIDIA, Google, Hugging Face u OpenAI entrenan sus modelos frontier. TorchTitan, Megatron-LM/Megatron Core, MaxText y Nanotron sí son implementaciones públicas para entrenamiento a escala; de OpenAI no hay evidencia pública equivalente de su stack interno de entrenamiento.

Los enlaces a GitHub están fijados a los commits consultados cuando resulta útil, para que la evidencia no cambie bajo nuestros pies.

## Resumen ejecutivo

El patrón común no es una estructura de carpetas idéntica, sino unas fronteras claras:

1. La definición matemática del modelo vive separada de cómo se paraleliza.
2. Las recetas/configuraciones son artefactos versionados y seleccionables, no argumentos dispersos en el código.
3. Datos, checkpoints, métricas y evaluación tienen interfaces propias y estado explícito.
4. El bucle de entrenamiento coordina componentes; no contiene todas sus implementaciones.
5. La corrección distribuida se prueba por capas: unit tests, integración multi-GPU, convergencia y benchmarks.
6. Los experimentos nuevos empiezan fuera del núcleo estable y migran al core cuando maduran.

La lección para un proyecto educativo no es replicar cada abstracción. Es conservar esas fronteras con una implementación mucho menor.

## Meta / PyTorch: TorchTitan

### Hechos públicos

TorchTitan se presenta como una plataforma PyTorch nativa para experimentación y entrenamiento a gran escala, con énfasis en una base mínima, legible y extensible. Su propio README señala como puntos de entrada el bucle principal, el modelo Llama, la aplicación de paralelismo, pipeline parallel y checkpointing ([README oficial](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/README.md)).

La estructura principal separa:

- `torchtitan/models/<modelo>/`: `model.py`, registro de configuraciones, sharding/paralelización y adaptador de `state_dict`.
- `torchtitan/components/`: dataloader, loss, optimizador, scheduler, métricas, validación y checkpoints.
- `torchtitan/distributed/`: primitives e infraestructura distribuida.
- `torchtitan/config/`: sistema de configuración tipado.
- `torchtitan/observability/`: logging estructurado.
- `torchtitan/experiments/`: ideas que todavía no pertenecen al núcleo.
- `tests/unit_tests`, `tests/integration_tests` y `benchmarks/`.

La guía oficial para añadir modelos pide que `model.py` sea código de modelo de un solo dispositivo y que `parallelize.py` aplique, por separado, sharding, activation checkpointing, compilación y FSDP/HSDP. También asigna un archivo específico a la interoperabilidad de checkpoints y recomienda comenzar nuevos modelos en `experiments/` ([guía oficial de modelos](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/torchtitan/models/README.md)).

Las recetas son funciones Python versionadas. Por ejemplo, el registro de Llama devuelve una configuración compuesta del trainer con modelo, loss, optimizador, scheduler, dataloader, paralelismo, checkpoint, profiler, métricas y validador ([registro oficial de Llama 3](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/torchtitan/models/llama3/config_registry.py)). Se seleccionan mediante módulo y nombre de configuración, y el CLI puede sobrescribir campos.

En datos, TorchTitan documenta dataloaders distintos para pretraining, SFT y multimodal; soporta mezclas de fuentes y guarda el estado del interleaver y de cada fuente para reanudación reproducible ([documentación oficial de datasets](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/docs/datasets.md)).

El checkpoint manager usa PyTorch Distributed Checkpoint, puede guardar estado completo para reanudar o solo modelo para exportar, excluir estados concretos al cargar, convertir formatos e interoperar con Hugging Face ([documentación oficial de checkpoints](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/docs/checkpoint.md)).

La observabilidad incluye loss, memoria, throughput, TFLOPs/MFU y tiempos de carga; soporta TensorBoard o Weights & Biases ([documentación oficial de métricas](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/docs/metrics.md)). Los tests separan unitarios e integración, incluyen assets pequeños y permiten probar configuraciones y grados de GPU específicos ([README oficial de tests](https://github.com/pytorch/torchtitan/blob/51c197c86d7c703da96f666d5a7dbd5432b4afbf/tests/README.md)).

### Inferencia útil

Para un proyecto educativo en PyTorch, TorchTitan es probablemente la referencia estructural más cercana: conserva un modelo legible de un dispositivo y trata la distribución como una transformación posterior. No implica que los repos internos de Meta tengan exactamente esta forma.

## NVIDIA: Megatron-LM y Megatron Core

### Hechos públicos

NVIDIA distingue explícitamente dos capas: **Megatron Core**, una biblioteca componible de bloques optimizados, y **Megatron-LM**, una implementación de referencia con scripts de entrenamiento preconfigurados ([README oficial](https://github.com/NVIDIA/Megatron-LM/blob/48a887fecd01674346f724dcf44f417f455989f0/README.md)).

Su estructura refleja esa división:

- `megatron/core/models` y `megatron/core/transformer`: arquitecturas y bloques.
- `megatron/core/tensor_parallel`, `pipeline_parallel`, `distributed`: TP, PP y DP/FSDP.
- `megatron/core/datasets`, `optimizer`, `dist_checkpointing`, `inference`, `export`.
- `megatron/training/`: composición del entrenamiento, configuración, logging y checkpoint orchestration.
- `examples/`: recetas ejecutables por familia de modelo y caso de uso.
- `tools/`: conversión de checkpoints, preparación de datos y otras utilidades.
- `tests/unit_tests`, `tests/functional_tests` y `tests/performance_tests`.

El paralelismo es una capacidad de primera clase: TP, PP, DP, expert parallel y context parallel, combinables según la topología ([guía oficial de paralelismo](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)). La configuración reciente dispone de contenedores/configs tipados en `megatron/training/config`, mientras las recetas concretas permanecen en ejemplos y scripts versionados ([directorio oficial de configuración](https://github.com/NVIDIA/Megatron-LM/tree/48a887fecd01674346f724dcf44f417f455989f0/megatron/training/config)).

Los datasets tienen su propio subsistema de formatos indexados, datasets GPT/BERT/T5, construcción de mezclas y configuración del blend ([código oficial de datasets](https://github.com/NVIDIA/Megatron-LM/tree/48a887fecd01674346f724dcf44f417f455989f0/megatron/core/datasets)). Distributed checkpointing también es un paquete independiente, con mapping, serialización, validación y soporte del estado del optimizador ([código oficial de checkpointing distribuido](https://github.com/NVIDIA/Megatron-LM/tree/48a887fecd01674346f724dcf44f417f455989f0/megatron/core/dist_checkpointing)).

La suite distingue tests unitarios, funcionales y de rendimiento, incluyendo áreas específicas para determinismo, checkpointing distribuido, paralelismo, datasets, optimizador, inferencia y modelos ([árbol oficial de tests](https://github.com/NVIDIA/Megatron-LM/tree/48a887fecd01674346f724dcf44f417f455989f0/tests)). El entrenamiento integra TensorBoard y W&B; el código incluye callbacks de W&B alrededor del guardado/carga de checkpoints y artefactos ([integración oficial W&B](https://github.com/NVIDIA/Megatron-LM/blob/48a887fecd01674346f724dcf44f417f455989f0/megatron/training/wandb_utils.py)).

### Inferencia útil

Megatron demuestra la separación más fuerte entre una biblioteca reusable de primitives y una aplicación/receta de entrenamiento. Para un repositorio de una sola persona, copiar toda esa profundidad sería prematuro; sí conviene mantener la dirección de dependencias: `train` depende del core, pero el core no depende del script concreto de entrenamiento.

## Google: MaxText

### Hechos públicos

MaxText es una implementación JAX/Flax para entrenamiento LLM escalable en TPU y GPU. Declara usar Flax para redes, Optax para optimización, Orbax para checkpoints y Grain para datos ([README oficial](https://github.com/AI-Hypercomputer/maxtext/blob/c67a6c3788162ce386bb07dcda296fd18e8d4c20/README.md)).

El repositorio usa layout `src/` y divide el paquete en:

- `src/maxtext/models`, `layers`, `kernels`, `optimizers`.
- `src/maxtext/input_pipeline`.
- `src/maxtext/common`, que incluye checkpointing, métricas, profiling y utilidades de infraestructura.
- `src/maxtext/trainers` para pretraining y post-training.
- `src/maxtext/eval` e `inference`.
- `src/maxtext/configs`, con base común, variantes de modelos, hardware y post-training.
- `tests/unit`, `tests/integration`, `tests/end_to_end`, con divisiones GPU/TPU.
- `benchmarks/`, incluyendo convergencia, disrupciones, MMLU, recetas y servidores.
- `tools/` para datos, orchestration, setup e inspección de pesos.

Su configuración se construye por composición: una base YAML extensa y overrides específicos de modelo/topología. La base versiona juntos parámetros de arquitectura, datos, sharding, optimizador, logging y checkpointing ([config base oficial](https://github.com/AI-Hypercomputer/maxtext/blob/c67a6c3788162ce386bb07dcda296fd18e8d4c20/src/maxtext/configs/base.yml)).

El subsistema de datos está separado por backend/adaptador (`grain`, Hugging Face, TFDS, synthetic, multimodal), visible en el [directorio oficial de input pipeline](https://github.com/AI-Hypercomputer/maxtext/tree/c67a6c3788162ce386bb07dcda296fd18e8d4c20/src/maxtext/input_pipeline). Checkpointing está centralizado y basado en Orbax, con periodos, retención y variantes para checkpoints locales/de emergencia ([implementación oficial](https://github.com/AI-Hypercomputer/maxtext/blob/c67a6c3788162ce386bb07dcda296fd18e8d4c20/src/maxtext/common/checkpointing.py)). La métrica se escribe a TensorBoard/Vertex TensorBoard y los benchmarks guardan artefactos y resultados reproducibles ([directorio oficial de monitoring](https://github.com/AI-Hypercomputer/maxtext/tree/c67a6c3788162ce386bb07dcda296fd18e8d4c20/docs/guides/monitoring_and_debugging)).

### Inferencia útil

MaxText hace visible una preocupación que los proyectos pequeños suelen ignorar: la receta no es solo “modelo + learning rate”, sino modelo × datos × topología/hardware × resiliencia. En un proyecto pequeño puede bastar con overlays YAML sencillos, sin reproducir su matriz completa de hardware.

## Hugging Face: Nanotron

### Hechos públicos

Nanotron es una biblioteca PyTorch de pretraining con énfasis en simplicidad, rendimiento y escalabilidad. Publica soporte para paralelismo 3D (DP+TP+PP), expert parallelism, ZeRO-1, acumulación FP32 y checkpointing de módulos ([README oficial](https://github.com/huggingface/nanotron/blob/2411b022a75fb7f7561a1bb4166706da5e1b76de/README.md)).

Su layout `src/` separa:

- `src/nanotron/models` y `nn`.
- `parallel`, con estrategias y primitives distribuidas.
- `data`, `optim`, `serialize`, `eval`, `generation`, `logging`, `fp8`.
- `config`, con dataclasses/esquemas por dominio.
- `examples/`, que contiene recetas y extensiones como dataloader custom, DoReMi, MoE, Mamba y µTransfer.
- `tests/`, más tests locales dentro de algunos ejemplos.

Una receta YAML contiene secciones explícitas para checkpoints, etapas de datos, metadatos generales, logging, modelo, optimizador, paralelismo, profiler, tokenizer y presupuesto de tokens ([config tiny Llama oficial](https://github.com/huggingface/nanotron/blob/2411b022a75fb7f7561a1bb4166706da5e1b76de/examples/config_tiny_llama.yaml)). La configuración también admite varias etapas de datos y evaluación con LightEval.

El guardado está encapsulado en `serialize/`, separado en pesos, optimizador, RNG y metadata ([código oficial de serialización](https://github.com/huggingface/nanotron/tree/2411b022a75fb7f7561a1bb4166706da5e1b76de/src/nanotron/serialize)). La evaluación tiene su propio runner y soporte para subir resultados a W&B ([código oficial de evaluación](https://github.com/huggingface/nanotron/tree/2411b022a75fb7f7561a1bb4166706da5e1b76de/src/nanotron/eval)). Los tests cubren modelado, tied weights, TP, PP, DP, ZeRO, serialización, checkpoints, RNG, datasets y kernels ([árbol oficial de tests](https://github.com/huggingface/nanotron/tree/2411b022a75fb7f7561a1bb4166706da5e1b76de/tests)).

### Inferencia útil

Nanotron es una buena referencia intermedia entre un proyecto docente y Megatron: conserva recetas YAML legibles y paquetes relativamente directos, pero trata desde el principio el estado distribuido, la serialización y la topología como dominios propios.

## OpenAI: qué evidencia pública existe y qué no

### Hechos públicos

OpenAI publicó los pesos de gpt-oss y un repositorio con implementaciones de referencia para **inferencia** en PyTorch, Triton y Metal, además de herramientas y tests. El README dice expresamente que son implementaciones de inferencia y que la versión PyTorch es educativa, no una publicación del stack de entrenamiento ([repositorio oficial gpt-oss](https://github.com/openai/gpt-oss)). OpenAI describe públicamente aspectos de arquitectura, pretraining y post-training, pero se refiere a su “training stack” como interno, sin publicar su estructura ([anuncio oficial de gpt-oss](https://openai.com/index/introducing-gpt-oss/)).

### Límite de la evidencia

Por tanto, no es válido afirmar cómo organiza OpenAI internamente repositorios de datos, entrenamiento distribuido, checkpoints o tracking. `openai/gpt-oss` sí es evidencia útil de cómo empaqueta una implementación de referencia e interoperabilidad, pero no debe usarse como prueba de su arquitectura de entrenamiento.

## Comparación por preocupación

| Preocupación | TorchTitan | Megatron | MaxText | Nanotron |
|---|---|---|---|---|
| Modelo | carpeta por familia, single-device | core models + transformer blocks | models + layers + kernels | models + nn |
| Config | registro Python tipado + overrides CLI | configs tipados + scripts/recetas | YAML base + overlays de modelo/hardware | YAML validado por dataclasses |
| Datos | dataloaders por modalidad, stateful | datasets indexados y blends | adaptadores/backends en input pipeline | data + etapas configurables |
| Distribución | transformación separada del modelo | primitives core por eje de paralelismo | sharding JAX/configurado por topología | paquete `parallel` explícito |
| Checkpoints | manager DCP + adapters de formato | paquete distributed checkpointing | Orbax centralizado | pesos/optim/RNG/metadata separados |
| Evaluación | validator dentro de la receta | utilidades y ejemplos por workload | paquete eval + benchmarks | eval/LightEval separado |
| Tests | unit + integration + convergencia | unit + functional + performance | unit + integration + E2E TPU/GPU | unit/distributed + tests de ejemplos |
| Tracking | TensorBoard/W&B + logging estructurado | TensorBoard/W&B, artifacts | TensorBoard/Vertex TB | W&B + timers/profiler |

## Recomendaciones transferibles a `llm-from-scratch`

Estas son **inferencias de diseño**, no hechos sobre repos privados:

1. Mantener `model/` libre de launchers, W&B, Slurm y formatos de dataset.
2. Representar cada experimento con una config versionada y guardar una copia resuelta junto al checkpoint.
3. Hacer que el checkpoint completo incluya modelo, optimizador, scheduler, step, RNG y estado/posición de datos.
4. Separar un `train.py` coordinador de `trainer/` o componentes como optimizador, scheduler, métricas y validación.
5. Introducir distribución como capa posterior: primero single-device correcto; luego DDP/FSDP; TP/PP solo cuando el tamaño lo exija.
6. Tratar evaluación, generación y conversión/exportación como entrypoints separados del entrenamiento.
7. Disponer al menos de cuatro niveles de verificación: unitarios matemáticos, smoke test end-to-end, test de resume determinista y curva de convergencia de referencia.
8. Separar `experiments/` de APIs estables; el código experimental puede duplicar antes de generalizar.
9. Versionar pequeños fixtures de datos/tokenizer y valores golden, nunca el corpus o checkpoints grandes dentro de Git.
10. Registrar tanto calidad como sistemas: train/val loss, LR, grad norm, tokens procesados, tokens/s, utilización/memoria y tiempos de data loading/checkpoint.

El patrón profesional esencial es **separar responsabilidades y hacer reproducible el estado**, no maximizar el número de carpetas.
