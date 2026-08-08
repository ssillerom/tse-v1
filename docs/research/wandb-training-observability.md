# Observabilidad de pretraining en W&B: comparación y propuesta

Fecha de consulta: 2026-08-08

## Alcance

Este informe compara el seguimiento actual de `llm-from-scratch` con prácticas que están
documentadas en código o informes primarios de OLMo/OLMo-core, Qwen3, Apertus, Pythia y
DeepSeek-V3, además de las capacidades actuales de W&B.

No se inspeccionó un proyecto real de W&B: no se proporcionaron `entity`, `project` ni
credenciales. Por tanto, las afirmaciones sobre este repositorio proceden del código local y
las afirmaciones sobre W&B proceden de su documentación; no se infiere que un proyecto externo
use W&B sólo porque publique un informe de entrenamiento.

## Qué existe hoy en este repositorio

La base actual ya cubre correctamente el núcleo de una ejecución educativa reproducible:

- `src/training/wandb_logging.py` publica por paso `train/loss`, `optimizer/gradient_norm`,
  `optimizer/learning_rate`, tokens acumulados, tiempo del paso y tokens/segundo; publica
  validación agregada y por dominio, y una tabla periódica de prompts fijos.
- `src/scripts/train_v1.py` guarda en `wandb.init(config=...)` la arquitectura, entrenamiento,
  AdamW, receta/fuentes/fases, seed, dispositivo, número de parámetros y `torch.compile`.
- `src/training/trainer.py` mide tokens reales por paso, evita incluir la evaluación en el
  tiempo de entrenamiento, evalúa de forma determinista y conserva el RNG.
- `src/training/observers.py` lleva el consumo exacto por fuente, conserva el mejor checkpoint,
  checkpoints numerados y delega el log W&B después de persistir el estado.
- `src/training/checkpoint.py` restaura pesos, AdamW, step, RNG, contrato de datos, posición,
  tokens y la identidad del run W&B. La publicación del checkpoint es atómica.
- `docs/training/evaluation.md` ya separa validación durante pretraining, comparación de
  mezclas y evaluación post-hoc.

La conclusión es que no falta un logger básico: falta una capa de observabilidad de sistemas,
estabilidad, linaje y análisis longitudinal.

## Evidencia primaria por proyecto

### OLMo / OLMo-core: la referencia más accionable para telemetría

El repositorio de OLMo enlaza configuraciones, checkpoints y ejecuciones W&B de OLMo2 por
etapa. OLMo-core implementa una `WandBCallback` que registra desde rank 0, organiza el run con
`project`, `entity`, `group`, `tags` y `notes`, guarda la configuración local del trabajo y
permite cancelar desde W&B mediante tags.

Sus callbacks son especialmente útiles como catálogo de métricas:

- `SpeedMonitorCallback`: tokens/s instantáneos y promedio real, tokens globales, FLOPs/s,
  petaflops acumulados, batches/s, tiempo de carga, porcentaje de tiempo en carga, MFU y
  múltiplo de Chinchilla. Ignora el primer paso para no contaminar la medición con warm-up.
- `StabilityMonitorCallback`: detecta spikes de pérdida y norma de gradiente contra una ventana
  móvil, publica un `SpikeScore` acumulado y guarda su estado para reanudarlo.
- `GAPMonitorCallback`: obtiene estadísticas max/mean/var de gradientes, activaciones y
  parámetros; opcionalmente vuelca gradientes completos o muestras en `safetensors` para
  análisis offline.
- Callbacks adicionales de OLMo-core cubren profiler, memoria GPU, guardado de configuración,
  evaluador, checkpoints y notificaciones.

Referencias: [OLMo](https://github.com/allenai/olmo),
[WandBCallback](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/train/callbacks/wandb.py),
[SpeedMonitorCallback](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/train/callbacks/speed_monitor.py),
[StabilityMonitorCallback](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/train/callbacks/stability_monitor.py),
[GAPMonitorCallback](https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/train/callbacks/gap_monitor.py),
[changelog de OLMo-core](https://github.com/allenai/OLMo-core/blob/main/CHANGELOG.md).

### Qwen3: observabilidad del currículo y de la mezcla

El informe de Qwen3 describe un corpus anotado a escala de más de 30T tokens con dimensiones
como valor educativo, campo, dominio y seguridad. También describe optimización de mezcla a
nivel de instancia mediante modelos proxy, no sólo a nivel de fuente.

El pretraining tiene tres etapas explícitas: general, razonamiento y contexto largo. Cada etapa
cambia la distribución de datos y, en algunos casos, el contexto y la política de learning
rate. El informe evalúa 15 benchmarks en conocimiento general, razonamiento, matemáticas/STEM,
code y multilingüe, usando el mismo pipeline para las comparaciones.

La idea transferible no es copiar los números de Qwen, sino registrar cada cambio de régimen
como una fase observable: distribución efectiva, tokens de fase, longitud de secuencia,
learning rate, batch y evaluación relevante. El repositorio Qwen3 público no demuestra una
implementación W&B de su pretraining; su sección de training visible se centra en post-training
y aparece como TODO. No se debe atribuirle una métrica W&B que el código no publica.

Referencias: [informe técnico Qwen3](https://arxiv.org/html/2505.09388),
[repositorio Qwen3](https://github.com/QwenLM/Qwen3).

### Apertus / Swiss AI: trazabilidad operacional y reanudación

Apertus publica la infraestructura de entrenamiento, scripts de reconstrucción de datos,
evaluación y artefactos. Su código de pretraining está basado en Megatron-LM y los scripts de
Slurm muestran una operación de larga duración con:

- W&B y TensorBoard opcionales, throughput, normas de parámetros y memoria;
- checkpoint periódico, carga, guardado asíncrono y directorios separados de logs/checkpoints;
- señal de Slurm antes del límite de tiempo para guardar checkpoint y salir limpiamente;
- backup del código y captura del comando, commit, entorno TOML, lista de paquetes, `nvidia-smi`,
  nodos y variables de entorno.

Es un patrón muy valioso para este repositorio aunque el despliegue local sea de una sola GPU:
un run debe explicar no sólo la curva de pérdida, sino qué binario, código, hardware, comando y
estado de reanudación la produjeron.

Referencias: [documentación Apertus](https://www.apertus-ai.org/pages/documentation/),
[pretrain-code](https://github.com/swiss-ai/pretrain-code),
[scripts de pretraining](https://github.com/swiss-ai/pretrain-code/tree/main/pretraining),
[script Apertus 8B](https://raw.githubusercontent.com/swiss-ai/pretrain-code/main/pretraining/submit_apertus_8b.sh).

### Pythia / EleutherAI: cronología científica

Pythia publica datos, código y modelos; conserva 154 checkpoints por modelo, con el mismo orden
de datos entre ejecuciones. Esto permite estudiar dinámica de aprendizaje y hacer
intervenciones causales, no sólo comparar el modelo final.

Para este repo, la traducción práctica es conservar una cronología de checkpoints de análisis
con alias y metadatos de tokens, y distinguir esos checkpoints científicos de los mínimos
necesarios para reanudar. No hace falta subir cada estado de AdamW a W&B: puede bastar con
artefactos de pesos en hitos, un índice completo y checkpoints locales durables.

Referencia: [repositorio Pythia](https://github.com/EleutherAI/pythia).

### DeepSeek-V3: objetivo de estabilidad, no receta W&B

El informe de DeepSeek-V3 afirma que el pretraining no tuvo spikes de pérdida irreversibles ni
rollbacks. El informe sí explica infraestructura, coste, datos, entrenamiento y evaluación,
pero no ofrece un callback W&B comparable al de OLMo-core. La lección transferible es definir
un contrato explícito de estabilidad y demostrarlo con series de spikes, no presentar la
afirmación como una característica de su integración W&B.

Referencia: [informe técnico DeepSeek-V3](https://arxiv.org/html/2412.19437).

## Capacidades W&B relevantes

W&B ya puede aportar una parte importante sin instrumentación casera:

- registra automáticamente sistema, CPU/GPU, red, disco, memoria, stdout/stderr y, si se
  habilita el guardado de código, commit/diff y dependencias;
- permite `group`, `job_type`, tags y notes para organizar runs relacionados;
- `summary` sirve para conservar best/final y otros agregados sin duplicar series;
- Artifacts versionan datasets, manifiestos, checkpoints y outputs, y muestran linaje entre runs;
- Tables sirven para muestras, resultados de evaluación y eventos de checkpoint;
- Workspaces permiten guardar vistas compartidas con paneles de pérdida, rendimiento, sistema,
  fases y estabilidad;
- `run.alert()` y Automations cubren alertas de NaN, spikes, thresholds, fallos y cambios de
  artefactos. Las automatizaciones basadas en métricas dependen del tipo de despliegue W&B.

Referencias: [logging](https://docs.wandb.ai/models/track/log),
[system metrics](https://docs.wandb.ai/models/ref/python/experiments/system-metrics),
[Run API](https://docs.wandb.ai/models/ref/python/experiments/run),
[summary metrics](https://docs.wandb.ai/models/track/log/log-summary),
[Artifacts](https://docs.wandb.ai/models/artifacts),
[Tables](https://docs.wandb.ai/guides/track/log/log-tables),
[Workspaces](https://docs.wandb.ai/models/track/workspaces),
[reproducibilidad](https://docs.wandb.ai/models/track/reproduce_experiments),
[alerts](https://docs.wandb.ai/models/runs/alert),
[Automations](https://docs.wandb.ai/models/automations).

## Comparación y propuesta priorizada

| Área | Estado actual | Propuesta | Prioridad | Fuente/idea |
| --- | --- | --- | --- | --- |
| Curva de entrenamiento | Loss, grad norm, LR, tokens y throughput por paso | Mantener contrato; añadir media móvil, pérdida no finita y step explícito | P0 | Base actual + semántica de steps W&B |
| Validación | Aggregate y dominios, determinista | Añadir evaluación post-hoc como run `job_type=eval` enlazado al checkpoint y registrar benchmark, versión, shots, error estándar y n | P1 | Qwen3 / OLMo |
| Fases de datos | Fases y cuotas sólo en config; tokens acumulados por fuente | `data/phase`, progreso de fase, tokens del paso por fuente, fracción observada, longitud de secuencia y hash del contrato como series/config | P0 | Qwen3 / Apertus |
| Datos versionados | Hash local del manifest/recipe y checksums en el manifest | Artifact de entrada con manifest, recipe, tokenizer metadata y hashes; usarlo como input del run | P0 | W&B Artifacts / Apertus |
| Checkpoints | Locales, atómicos, best + retención, identidad W&B | Artifact sólo para best/final/hitos; aliases `latest`, `best`, `step-*`; Table de índice con step, tokens, val loss, bytes, duración y SHA256 | P0 | W&B Artifacts / Pythia |
| Reanudación | Restaura RNG, optimizer, posición, tokens y run ID | Registrar `run/resumed`, checkpoint origen, fallback, motivo de salida y hash de entorno; conservar metadatos de logging en el checkpoint | P1 | Apertus / OLMo |
| Rendimiento | Step time y tokens/s, eval excluida | Cargar vs compute vs optimizer vs eval vs checkpoint; tokens/s actual y promedio; FLOPs/s, MFU, petaflops y Chinchilla multiple | P0 | OLMo SpeedMonitor |
| Hardware | W&B puede recoger sistema automáticamente | Además, snapshot explícito de GPU, memoria peak, Torch/CUDA/cuDNN, driver, hostname/job y paths relevantes | P0 | Apertus / W&B system metrics |
| Estabilidad | Se aborta ante no finitos | Monitor de spikes de loss/grad norm con estado persistente, score acumulado y alertas con cooldown | P0 | OLMo StabilityMonitor / DeepSeek |
| Salud del modelo | Sólo norma global de gradiente | GAP ligero cada N pasos: max/mean/var por módulo; dumps de gradientes sólo en ventanas/anomalías como artifact | P1 | OLMo GAP |
| Optimizer | LR y grad norm | Pre/post clip, fraction clipped, norma de parámetros, norma de actualización y update/param ratio, sólo a frecuencia baja | P1 | OLMo / práctica de diagnósticos |
| Muestras | Tabla fija de continuaciones greedy | Un único log por step que incluya tabla cuando toque; añadir prompt id, tokens generados, stop reason y latencia | P1 | Pythia / Tables |
| Organización | Project/name/mode/id/resume | `group`, `job_type`, tags, notes, `save_code`, `dir` y nombres estables para seed/recipe/phase | P0 | OLMo / W&B Run |
| Alertas | Ninguna específica | NaN, spike, throughput anómalo, disco bajo, checkpoint fallido y run crash; cancelación opcional por tag | P1 | OLMo / W&B Alerts |
| Dashboard | Se puede construir manualmente con las series actuales | Vista guardada canónica con pérdida por tokens, dominios, fase, throughput/MFU, sistema, memoria, checkpoints, estabilidad y samples | P1 | W&B Workspaces |
| Profiling | No hay trace integrado | Ventanas PyTorch Profiler sólo en rehearsal/hitos y subir el trace como artifact | P2 | OLMo Profiler / Apertus |

## Esquema W&B sugerido

No conviene convertir cada tensor o cada documento en una métrica: W&B advierte que la
cardinalidad de claves, payload y frecuencia afectan al rendimiento. Propongo estas familias
estables:

```text
train/loss
train/loss_ema
train/tokens_in_step
train/tokens_seen
optimizer/learning_rate
optimizer/gradient_norm_pre_clip
optimizer/gradient_norm_post_clip
optimizer/clip_fraction
performance/step_seconds
performance/data_wait_seconds
performance/tokens_per_second
performance/tokens_per_second_avg
performance/flops_per_second
performance/mfu
performance/chinchilla_multiple
data/phase_index
data/phase_tokens_seen
data/source_tokens_in_step/<source>
data/source_tokens_seen/<source>
validation/loss
validation/perplexity
validation/<domain>/loss
stability/loss_spike
stability/grad_norm_spike
stability/spike_score_total
memory/max_allocated_bytes
checkpoint/save_seconds
checkpoint/step
```

`gap/*/<module>` sólo debe aparecer con una frecuencia configurable y, preferiblemente, con
una lista fija de módulos. Los dumps completos de gradientes, perfiles y checkpoints no deben
viajar como series escalares: deben ser artifacts con metadata y aliases.

## Orden de implementación propuesto

1. **Contrato y organización (P0):** ampliar metadatos de `wandb.init`, hash de datos,
   tokenizer, código/commit/dirty state, entorno, `group`, `job_type`, tags y notes; definir
   `step`/`commit` y evitar el log doble de la tabla de samples.
2. **Rendimiento y fases (P0):** separar tiempos de carga/cómputo/evaluación/checkpoint,
   calcular promedio estable, FLOPs/MFU y publicar la fase y mezcla observada.
3. **Artefactos y checkpoints (P0):** versionar input manifest/recipe y best/final/hitos;
   añadir índice de checkpoints y resumen de run.
4. **Estabilidad (P0/P1):** detector de spikes con estado en checkpoint, métricas de salud,
   alertas con cooldown y motivo de finalización.
5. **Diagnóstico profundo (P1):** GAP muestreado, normas de actualización, memoria explícita,
   tablas de muestras enriquecidas y runs de evaluación post-hoc.
6. **Perfilado y dashboard (P2):** traces acotados y una vista W&B guardada y versionada.

## Criterios antes de aceptar una implementación

- Un run offline debe producir exactamente el mismo esquema y poder validarse sin red.
- Reanudar desde un checkpoint debe conservar el mismo run, step, tokens, RNG y estado del
  detector de estabilidad.
- La curva de `performance/tokens_per_second` debe excluir warm-up y reportar por separado
  carga, cómputo, evaluación y checkpoint.
- El hash del artifact de datos debe coincidir con el contrato que usa el loader.
- Una evaluación externa debe poder reconstruirse desde un checkpoint artifact y dejar su propio
  run enlazado, sin contaminar la curva de pretraining.
- Los smoke tests deben comprobar no finitos, clipping, cambios de fase, fallback de checkpoint,
  modo `disabled`, tablas y límites de cardinalidad.

Este documento es una propuesta de diseño y comparación. No implementa ninguna de estas
mejoras.
