# Recetas públicas de entrenamiento de LLM

Fecha de revisión: 29 de julio de 2026.

## Qué significa aquí «receta abierta»

Que un modelo permita descargar sus pesos no significa que se pueda reproducir su
entrenamiento. Para distinguir ambos conceptos se usa esta escala:

- **A+ — reproducción científica completa:** datos, código ejecutable, configuración,
  checkpoints intermedios y métricas o logs.
- **A — reproducción práctica:** la receta se puede ejecutar de extremo a extremo, aunque
  falte algún artefacto secundario o exista fricción para recuperar datos antiguos.
- **B — receta detallada pero incompleta:** se conocen arquitectura, datos e hiperparámetros,
  pero falta al menos una pieza importante para reproducir exactamente el resultado.
- **C — informe técnico:** hay decisiones y cifras útiles, pero los datos o la infraestructura
  esencial no son públicos.
- **D — pesos o método abierto:** no existe una receta de entrenamiento reproducible.

La licencia de los pesos, la licencia del código y las licencias de cada fuente de datos son
cuestiones distintas. La nota indica las restricciones más relevantes, pero no es un análisis
legal.

## Comparación rápida

| Proyecto | Escala y tokens | Qué se publica | Limitación principal | Nivel |
|---|---:|---|---|:---:|
| **OLMo 2** | 1B: 4T + 50B; 7B: 4T + 3×50B; 13B: 5T + annealing | Datos, orden/mezclas, configs, código, W&B y checkpoints cada 1.000 pasos | Reproducir el cómputo completo sigue siendo muy caro | **A+** |
| **OLMo 3** | 7B: 5,93T; 32B: 5,50T; contexto final 65.536 | Dolma 3, scripts oficiales, checkpoints, logs y recetas de todas las etapas | Escala mínima de 7B y hasta 1.024 H100 | **A+** |
| **MobileLLM-R1** | 140M, 360M y 950M; 4,2T de pretraining + post-training | Fuentes, proporciones, código, modelos y tabla completa de entrenamiento | Licencia FAIR no comercial; receta muy costosa incluso para 360M | **A** |
| **SmolLM2** | 135M: 2T; 360M: 4T; 1,7B: 11T | Datos abiertos, código, configs y checkpoints Nanotron con estado del optimizador | Algunos detalles están repartidos entre paper, repositorio y configs | **A** |
| **Pythia** | 70M–12B; ~300B por modelo | Mismo orden de datos, código, 154 checkpoints por modelo y curvas | The Pile es una receta histórica con componentes hoy difíciles de recuperar | **A** |
| **Cerebras-GPT** | 111M–13B; 20 tokens por parámetro | Configs, pesos, código y receta Chinchilla sobre The Pile | Infraestructura original CS-2 y disponibilidad actual de The Pile | **A−** |
| **LLM360 Amber** | 7B; ~1,3T | Datos, código, 360 checkpoints, métricas y estados intermedios | Receta antigua, pesada y basada en RedPajama v1 | **A−** |
| **OpenELM** | 270M–3B; ~1,5T entrenados desde un pool de ~1,8T | CoreNet, configs, logs, checkpoints y fuentes públicas | Licencia Apple y disponibilidad/licencias de componentes históricos | **B+** |
| **TinyLlama** | 1,1B; 3T exposiciones | Código, config, datos públicos y varios checkpoints | Menos trazabilidad del orden y de los estados del optimizador | **B+** |
| **Poolside Laguna XS.2** | 33,4B totales, 3B activos; >30T | Informe técnico con mezcla, fases y optimización muy detalladas; pesos XS.2 | Corpus, generación sintética, tokenizer y pipeline exactos son internos | **C** |
| **Ornith 1.0** | 9B–397B; post-training sobre Gemma/Qwen | Pesos y descripción de self-scaffolding RL | No publica datos, config ni pipeline completos; no es pretraining desde cero | **D** |

## Recetas más útiles

### 1. OLMo 2: la referencia de transparencia

OLMo 2 es el patrón más sólido si el objetivo es poder explicar de dónde sale cada resultado.
Su entrenamiento tiene dos etapas:

1. **Stage 1:** pretraining principalmente web con `OLMo-mix-1124`. El 1B y el 7B ven
   aproximadamente 4T tokens; el 13B, 5T.
2. **Stage 2:** annealing con `Dolmino-mix-1124`, una mezcla de mayor calidad. El 1B usa 50B;
   el 7B entrena tres semillas de 50B y promedia los pesos; el 13B combina tres runs de 100B y
   uno de 300B.

La configuración oficial del 1B es especialmente relevante para comparar con un modelo
educativo pequeño:

- 16 capas, dimensión 2.048, 16 cabezas, contexto 4.096.
- RoPE con `theta=500000`, RMSNorm, QK-norm, SwiGLU y FlashAttention.
- AdamW con `lr=4e-4`, betas `(0.9, 0.95)`, `weight_decay=0.1`, clipping a 1.
- Batch global de 512 secuencias, equivalente a 2.097.152 tokens.
- Warmup lineal y decay coseno; BF16.

Ai2 publica las configuraciones exactas, los ficheros de datos, W&B y checkpoints al menos cada
1.000 pasos. Código y modelos usan Apache 2.0; cada corpus conserva su propia licencia.

Fuentes primarias:

- [Repositorio y tabla de reproducción de OLMo 2](https://github.com/allenai/OLMo)
- [Config oficial OLMo 2 1B, stage 1](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage1.yaml)
- [Config oficial OLMo 2 1B, stage 2](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo2-1B-stage2-seed42.yaml)
- [Modelo y arquitectura OLMo 2 1B](https://huggingface.co/allenai/OLMo-2-0425-1B)
- [Descripción de OLMo Mix y Dolmino](https://allenai.org/blog/olmo2)

### 2. OLMo 3: transparencia completa a escala moderna

OLMo 3 conserva la filosofía de OLMo 2, pero convierte el entrenamiento en un flujo de varias
etapas: pretraining general con Dolma 3, mid-training especializado y extensión de contexto con
Dolma 3 Longmino. El modelo 7B registra 5,93T tokens y el 32B 5,50T; ambos terminan con contexto
de 65.536. Longmino aporta aproximadamente 50B tokens al 7B y 100B al 32B.

Se publican Dolma 3, scripts oficiales de OLMo-core, checkpoints, informes de W&B y artefactos
de las distintas ramas Base, Instruct y Think. El pretraining llegó a utilizar 1.024 H100. Es
la mejor referencia de un flujo moderno completo, pero no es una escala razonable para copiar
literalmente en este repositorio.

Fuentes primarias:

- [Scripts oficiales de OLMo 3 en OLMo-core](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official/OLMo3)
- [Model card de OLMo 3 7B](https://huggingface.co/allenai/Olmo-3-1025-7B)
- [Informe técnico de OLMo 3](https://arxiv.org/abs/2512.13961)
- [Resumen oficial y Dolma 3 Longmino](https://allenai.org/blog/olmo3)
- [Documentación de Dolma](https://docs.allenai.org/training_data/dolma)

### 3. MobileLLM-R1: la referencia más cercana a 350M

MobileLLM-R1 es el caso más útil para este proyecto porque publica modelos de 140M, 360M y 950M.
La versión 360M parte de inicialización aleatoria y sigue:

1. Dos fases de pretraining de aproximadamente 2T tokens cada una, con secuencia 2.048,
   500.000 pasos por fase y `lr=4e-3`.
2. Dos fases de mid-training de 100B tokens, contexto 4.096 y `lr=3.6e-4`.
3. SFT general y SFT de razonamiento con contexto de hasta 32K.

El corpus único ronda 2T tokens y se remuestrea hasta 4,2T exposiciones de pretraining. La tabla
pública da las proporciones exactas por fase de FineWeb-Edu, StarCoder, OpenWebMath, Wikipedia,
Arxiv, DCLM/Dolmino y otras fuentes. Usa Adam/AdamW con betas `(0.9, 0.95)`, epsilon `1e-8` y
`weight_decay=0.1`.

Es una receta de verdad, no sólo un paper, pero tiene dos advertencias: entrenar 4,2T tokens
para 360M sigue siendo una inversión grande, y la licencia FAIR permite investigación no
comercial, no el mismo uso permisivo que Apache 2.0.

Fuentes primarias:

- [Repositorio MobileLLM y enlace al código de R1](https://github.com/facebookresearch/MobileLLM)
- [Paper MobileLLM-R1](https://arxiv.org/abs/2509.24945)
- [Model card 360M con mezcla e hiperparámetros](https://huggingface.co/facebook/MobileLLM-R1-360M)

### 4. SmolLM2: datos de calidad para modelos pequeños

SmolLM2 muestra que un modelo pequeño se beneficia mucho de seleccionar datos para su escala.
El 360M se entrena con 4T tokens y el 135M con 2T. Sus fuentes incluyen FineWeb-Edu, Stack-Edu,
OpenWebMath/InfiMM-WebMath, FineMath y Cosmopedia. El 1,7B amplía la receta a 11T y utiliza
entrenamiento por etapas, WSD y una extensión final de contexto.

Hugging Face publica el código de preparación, las configuraciones de Nanotron y checkpoints
con estado del optimizador. Es una plantilla muy buena para estudiar mezcla de web educativo,
código, matemáticas y datos sintéticos, sin depender de un corpus privado.

Fuentes primarias:

- [Repositorio SmolLM y recetas de texto](https://github.com/huggingface/smollm/tree/main/text)
- [Paper SmolLM2](https://arxiv.org/abs/2502.02737)
- [Colección oficial SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bc2fcbf4c01e2c)

### 5. Pythia: observar cómo aprende el modelo

Pythia está diseñado para ciencia del entrenamiento, no para maximizar un benchmark final.
Entrena ocho tamaños, de 70M a 12B, durante unos 300B tokens. Todos ven los datos en el mismo
orden y usan un batch de 2.097.152 tokens durante 143.000 pasos. La publicación incluye 154
checkpoints por modelo, con checkpoints densos al principio y después cada 1.000 pasos.

Es excepcional para comparar curvas, seeds y tamaños controlando el orden de los datos. La
limitación es práctica: utiliza The Pile y algunos componentes originales resultan hoy más
difíciles de obtener o redistribuir.

Fuentes primarias:

- [Repositorio y tabla completa de Pythia](https://github.com/EleutherAI/pythia)
- [Paper Pythia](https://arxiv.org/abs/2304.01373)

### 6. Cerebras-GPT: un baseline simple basado en Chinchilla

Cerebras-GPT entrena siete tamaños, de 111M a 13B, con una regla uniforme de 20 tokens por
parámetro. Usa una arquitectura GPT-3 clásica, contexto 2.048, tokenizer BPE de 50.257 tokens y
AdamW con betas `(0.9, 0.95)`. Publica las tasas de aprendizaje y batches de cada tamaño.

Para un smoke train científico es una receta más fácil de razonar que las mezclas multietapa.
La reproducción exacta tiene fricción porque el run original usó sistemas Cerebras CS-2 y The
Pile.

Fuentes primarias:

- [Paper Cerebras-GPT](https://arxiv.org/abs/2304.03208)
- [Model card con la tabla de entrenamiento](https://huggingface.co/cerebras/Cerebras-GPT-13B)
- [Código Cerebras Model Zoo](https://github.com/Cerebras/modelzoo)

### 7. LLM360 Amber: trazabilidad histórica completa

Amber es un 7B entrenado desde cero con aproximadamente 1,3T tokens de RedPajama. LLM360
publicó datos, código, métricas, configuraciones, estados intermedios y 360 checkpoints. Su
valor principal es demostrar qué artefactos hacen falta para auditar un entrenamiento de
principio a fin.

No es la mejor mezcla para comenzar hoy: RedPajama v1 arrastra fuentes históricas cuya
disponibilidad y licencia requieren una revisión actual.

Fuentes primarias:

- [Portal LLM360](https://www.llm360.ai/)
- [Paper LLM360](https://arxiv.org/abs/2312.06550)
- [Organización LLM360 y artefactos de Amber](https://huggingface.co/LLM360)

## Recetas útiles con más fricción

### OpenELM

Apple publicó modelos de 270M, 450M, 1,1B y 3B entrenados durante aproximadamente 1,5T tokens.
El pool público disponible ronda 1,8T tokens y combina RefinedWeb, The Pile deduplicado,
RedPajama y Dolma. También publicó CoreNet, configuraciones, logs y checkpoints. Es
técnicamente detallado, pero la licencia Apple y la disponibilidad actual de algunos
componentes reducen su reproducibilidad práctica.

- [Paper OpenELM](https://arxiv.org/abs/2404.14619)
- [Código CoreNet](https://github.com/apple/corenet)
- [Model card OpenELM 270M](https://huggingface.co/apple/OpenELM-270M)

### TinyLlama

TinyLlama entrena un modelo Llama de 1,1B con SlimPajama y StarCoder. Recorre alrededor de 1T
tokens únicos durante tres épocas, para 3T exposiciones totales, usando FlashAttention. El
repositorio, la configuración y numerosos checkpoints están publicados bajo Apache 2.0. Es un
buen ejemplo minimalista, aunque su trazabilidad es menor que OLMo o Pythia.

- [Repositorio TinyLlama](https://github.com/jzhang38/TinyLlama)
- [Paper TinyLlama](https://arxiv.org/abs/2401.02385)
- [Checkpoint final y puntos intermedios](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)

## Casos que no permiten reproducir el modelo

### Poolside Laguna XS.2

El informe de Laguna XS.2 contiene ideas avanzadas y cifras poco habituales en modelos
open-weight:

- MoE de 33,4B parámetros totales y 3B activos; 8 de 256 expertos más un experto compartido.
- Más de 30T tokens a partir de un pool de aproximadamente 27T tokens únicos.
- Mezcla declarada: 30,6% código, 25,2% web, 25,4% sintético/código-texto, 9% matemáticas,
  6,6% conocimiento, 1,4% instrucciones, 1,1% académico y 0,7% libros.
- Pretraining en 2.048 H200, batch global de 24M tokens, BF16, optimizador Muon/Moonlight y
  schedule WSD.
- Extensión de contexto mediante dos etapas de 100B tokens, primero a 32K y después a 128K.
- Mid-training de unas 60B instrucciones, SFT centrado en código y RL con recompensa
  verificable.

Poolside libera XS.2 con Apache 2.0, pero no publica el corpus exacto, el tokenizer entrenado,
la generación de datos sintéticos, Titan ni sus configuraciones ejecutables y checkpoints de
entrenamiento. Por tanto, es una referencia de decisiones, no una receta reproducible.

- [Informe técnico oficial de Laguna](https://poolside.ai/assets/laguna/laguna-m1-xs2-technical-report.pdf)
- [Colección de modelos Laguna](https://huggingface.co/collections/poolside/laguna-xs2)

### Ornith 1.0

Ornith no es un modelo base entrenado desde cero: parte de modelos Gemma y Qwen y aplica
post-training mediante self-scaffolding RL. En cada iteración el modelo propone o refina el
scaffold que usará para resolver una tarea y la recompensa actualiza tanto esa estrategia como
la respuesta. La descripción incluye GRPO, control de staleness y salvaguardas contra reward
hacking.

No se publican la mezcla de datos, cantidades, hiperparámetros completos ni un pipeline
ejecutable de entrenamiento. La licencia MIT de los pesos no convierte la metodología en una
receta reproducible, y también deben respetarse las condiciones del modelo base.

- [Descripción oficial de Ornith 1.0](https://deep-reinforce.com/ornith_1_0.html)
- [Repositorio oficial](https://github.com/deepreinforce-ai/Ornith-1)
- [Model card Ornith 9B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B)

## Qué copiar para este V1 de aproximadamente 350M

No conviene copiar una única receta completa. La combinación más razonable es:

1. **OLMo 2 para la disciplina experimental:** manifest versionado, configuración íntegra,
   seeds, orden de datos, W&B y checkpoints resumibles.
2. **MobileLLM-R1 360M para elegir la escala de arquitectura, estudiar el batch y diseñar
   fases de entrenamiento.** Su `lr=4e-3` no debe copiarse sin un LR sweep: depende de su
   arquitectura e inicialización concretas.
3. **SmolLM2 360M para la mezcla de datos:** web educativo como base, con porciones
   explícitas de código, matemáticas y conocimiento.
4. **Pythia para el protocolo de observación:** guardar más checkpoints al comienzo y usar
   prompts/evaluaciones fijas para ver cuándo aparecen capacidades.
5. **Cerebras-GPT como control sencillo:** una primera curva basada en tokens por parámetro
   permite comprobar el trainer antes de introducir múltiples etapas.

Poolside no es una receta adecuada para este V1: su mezcla es deliberadamente code-heavy, usa
MoE y requiere más de 30T tokens. Ornith tampoco responde a la fase actual, porque es una
técnica de post-training sobre modelos base ya entrenados.

Antes de un run grande, este repositorio debería demostrar tres escalones:

- **Overfit de un batch o corpus minúsculo:** prueba matemática de que loss y backpropagation
  funcionan.
- **Pilot repetible:** misma seed, mismos shards y mismo checkpoint deben reproducir la curva
  y reanudar sin cambiar el orden de muestras.
- **Ablation pequeña de datos y learning rate:** comparar pocas mezclas y tasas con el mismo
  presupuesto; elegir por validation loss/perplexity y evaluaciones fijas, no sólo por training
  loss.

El run final debería registrar como mínimo: versión y hash del corpus, tokenizer, distribución
de fuentes, tokens reales consumidos, tokens únicos, secuencia, batch global en tokens,
acumulación, optimizador, schedule, grad norm, throughput, validation loss/perplexity,
checkpoints, estado del optimizador y RNG. Ésa es la diferencia práctica entre publicar pesos y
publicar una receta de entrenamiento.
