# Comparación de la arquitectura V1 con LLM famosos

Fecha de consulta: 2026-08-08.

## Conclusión

La V1 es un Transformer causal decoder-only, denso y de tipo “Llama-style”:
pre-norm con RMSNorm, RoPE aplicado a Q/K, atención causal multi-head con MHA o
GQA, MLP SwiGLU con SiLU, conexiones residuales y proyección de salida sin bias.
La receta de entrenamiento usa GQA con 16 cabezas Q y 4 cabezas K/V (4:1); la
configuración genérica conserva MHA cuando no se especifica `n_kv_heads`. En la
configuración por defecto también ata los embeddings de entrada y salida. Estas
propiedades se observan en
[`ModelConfig`](../../src/model/config.py), [`TransformerBlock`](../../src/model/block.py),
[`MultiHeadAttention`](../../src/model/attention.py), [`SwiGLU`](../../src/model/swiglu.py)
y [`GPT`](../../src/model/gpt.py).

Por tanto, la V1 no es una arquitectura exótica: es una versión pequeña y
didáctica de la familia que popularizó Llama. La implementación permite comparar
MHA con GQA, y el run de entrenamiento usa GQA para reducir el coste de la caché
KV: cuatro cabezas Q comparten cada cabeza K/V. Raschka describe GQA precisamente
como una variante de MHA en la que varias cabezas Q comparten K/V
([Raschka, GQA](https://sebastianraschka.com/llm-architecture-gallery/gqa/)).

## Comparación

| Modelo | Parecido con V1 | Coincidencias | Diferencias importantes |
| --- | --- | --- | --- |
| **Llama 2 7B/13B** | **Muy alto** | El código oficial usa RMSNorm, RoPE, MLP SwiGLU, lineales sin bias, pre-norm, residuales y MHA cuando `n_kv_heads = n_heads` ([código oficial de Meta](https://raw.githubusercontent.com/meta-llama/llama/main/llama/model.py)). | V1 ata embeddings por defecto; el código oficial mantiene una salida separada. Cambian escala, redondeo del tamaño oculto del MLP, vocabulario y contexto. La ficha oficial confirma que Llama 2 7B/13B no usan GQA y que 70B sí ([model card de Meta](https://raw.githubusercontent.com/meta-llama/llama/main/MODEL_CARD.md)). |
| **OLMo 2 7B** | **Muy alto en componentes; medio en el bloque** | Decoder denso, MHA, RMSNorm, RoPE, SwiGLU/SiLU y lineales sin bias ([galería de Raschka](https://sebastianraschka.com/llm-architecture-gallery/), [configuración publicada](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct/blob/main/config.json)). | OLMo 2 coloca RMSNorm después de atención/MLP dentro del residual y añade QK-Norm; V1 hace pre-norm y no normaliza Q/K ([Raschka, comparación](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison?open=false), [Ai2](https://allenai.org/blog/olmo2)). |
| **Llama 3 / 3.1** | **Alto** | Transformer denso, RMSNorm pre-norm, RoPE, SwiGLU, residual y lineales sin bias; el informe oficial confirma GQA, RoPE y SwiGLU ([informe técnico de Llama 3](https://arxiv.org/abs/2407.21783), [galería de Raschka](https://sebastianraschka.com/llm-architecture-gallery/)). | Usa GQA con 8 cabezas KV, no MHA completa; también cambia el tamaño de vocabulario y la base de RoPE. |
| **Qwen3 denso (4B/8B/14B/32B)** | **Alto** | El informe oficial enumera GQA, SwiGLU, RoPE y RMSNorm con pre-normalización; además elimina los bias QKV ([informe técnico de Qwen3](https://arxiv.org/abs/2505.09388)). | Añade QK-Norm y usa GQA: por ejemplo, Qwen3-32B tiene 64 cabezas Q y 8 KV; las variantes grandes no atan embeddings. |
| **Mistral 7B** | **Alto** | Transformer denso causal con SiLU, RMSNorm y RoPE; su configuración publicada especifica además GQA y atención de ventana deslizante ([configuración publicada](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json)). | Usa GQA y sliding-window attention; el informe original especifica 32 cabezas Q, 8 KV y ventana de 4096 ([informe de Mistral 7B](https://arxiv.org/abs/2310.06825)). |
| **Mistral Small 3.1** | **Alto** | Dense, RMSNorm, RoPE, SwiGLU/SiLU y GQA; Raschka lo clasifica como GQA estándar sin ventana deslizante ([galería de Raschka](https://sebastianraschka.com/llm-architecture-gallery/), [configuración publicada](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Base-2503/blob/main/config.json)). | Sigue usando 32 Q / 8 KV y no es MHA completa. |
| **Phi-4** | **Alto** | Decoder denso, pre-norm RMSNorm, RoPE, MLP SiLU-gated, sin bias de atención y GQA ([galería de Raschka](https://sebastianraschka.com/llm-architecture-gallery/), [configuración publicada](https://huggingface.co/microsoft/phi-4/blob/main/config.json), [implementación del MLP](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/phi3/modeling_phi3.py)). | GQA 40 Q / 10 KV, contexto 16K y embeddings no atados. |
| **Gemma 3** | **Medio-alto** | Decoder denso, RMSNorm, RoPE y atención de la misma familia. | Añade QK-Norm, GQA y una mezcla de atención local/global 5:1; su stack no es el MHA + SwiGLU simple de V1 ([galería de Raschka](https://sebastianraschka.com/llm-architecture-gallery/)). |
| **DeepSeek V3/R1, Qwen3-MoE, Llama 4, GPT-OSS** | **Medio: misma familia, no mismo bloque** | Siguen siendo variantes de decoder Transformer con RMSNorm y subredes feed-forward/gated. | Sustituyen el MLP denso por MoE y/o la atención convencional por MLA, GQA o atención híbrida. La galería de Raschka documenta estas diferencias ([comparación de arquitecturas](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison?open=false), [galería](https://sebastianraschka.com/llm-architecture-gallery/)). |
| **GPT-2** | **Medio-bajo** | También es decoder-only, causal y usa MHA con residuales. | Usa LayerNorm, posiciones absolutas aprendidas y GELU; no tiene el paquete moderno RMSNorm + RoPE + SwiGLU ([galería de Raschka](https://sebastianraschka.com/llm-architecture-gallery/)). |

## Orden de cercanía

1. **Llama 2 7B/13B**: referencia histórica más cercana al bloque V1.
2. **OLMo 2 7B**: comparte atención multi-head, RoPE, RMSNorm y SwiGLU, pero cambia el orden de normalización y añade QK-Norm.
3. **Llama 3, Qwen3 denso, Mistral y Phi-4**: casi la misma receta conceptual, con GQA en lugar de MHA.
4. **Gemma 3**: misma familia, pero con atención local/global y QK-Norm.
5. **DeepSeek/Qwen-MoE/Llama 4/GPT-OSS**: descendientes más especializados; MoE y variantes de atención los alejan de V1.

## Lectura práctica para este proyecto

La V1 ya implementa el núcleo pedagógico correcto para entender Llama y otros
LLM modernos, incluido `n_kv_heads` para soportar GQA sin cambiar el resto del
bloque. La receta actual usa 16 Q / 4 KV y la cache incremental conserva sólo las
cuatro cabezas K/V por capa; al alcanzar el contexto máximo se reconstruye desde
la ventana reciente. Después se podría experimentar con QK-Norm y, mucho más
adelante, MoE.
