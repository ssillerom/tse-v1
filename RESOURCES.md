# Recursos para construir un LLM desde cero

## Conocimiento

- [Código oficial de _Build a Large Language Model (From Scratch)_ — Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch)
  Referencia del libro. Usar para contrastar una implementación propia después de intentarla.
- [Building a 350M Transformer From Scratch — John Enev](https://john463212.substack.com/p/building-a-350m-transformer-from)
  Caso práctico que motiva el proyecto: arquitectura, pruebas, preparación de datos y entrenamiento.
- [Attention Is All You Need — Vaswani et al.](https://arxiv.org/abs/1706.03762)
  Fuente primaria para scaled dot-product attention, multi-head attention y el transformer original.
- [RoFormer — Su et al.](https://arxiv.org/abs/2104.09864)
  Fuente primaria para rotary position embeddings.
- [OLMo-core — Ai2](https://github.com/allenai/OLMo-core)
  Referencia estructural para configs tipadas, datos stateful, train modules y recetas Python.
- [Documentación de `scaled_dot_product_attention` — PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
  Usar después de validar una atención causal ingenua.
- [FineWeb-Edu — Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  Fuente de datos para una fase de pretraining seria.

## Sabiduría (comunidades)

- [PyTorch Forums](https://discuss.pytorch.org/)
  Comunidad oficial para problemas específicos de autograd, precisión, memoria y rendimiento.

## Huecos

- Concretar hardware local, presupuesto de nube y disponibilidad semanal antes de la fase de escala.
