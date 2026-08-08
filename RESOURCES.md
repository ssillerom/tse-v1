# Lectura del proyecto y PyTorch Resources

## Knowledge

- [PyTorch: torch.nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
  Base de los modelos y capas; usarlo para entender registro de submódulos, parámetros,
  train/eval, to, state_dict y apply.
- [PyTorch: torch.nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
  Explica la transformación afín que aparece en Q/K/V, SwiGLU y lm_head.
- [PyTorch: torch.nn.Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
  Referencia para convertir IDs enteros de tokens en vectores.
- [PyTorch: ModuleList](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)
  Explica por qué GPT.blocks registra todos los bloques Transformer.
- [PyTorch: data loading](https://docs.pytorch.org/docs/stable/data.html)
  Referencia de Dataset, DataLoader, Sampler, batching, drop_last y workers.
- [PyTorch: cross_entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
  Contrato de logits, targets, reducción e ignore_index para la loss causal.
- [PyTorch: Tensor.backward](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html)
  Explica autograd y la acumulación de gradientes en los parámetros.
- [PyTorch: AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
  Referencia del optimizador que actualiza los parámetros después del backward.
- [PyTorch: scaled dot-product attention](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)
  Contrato de SDPA, causalidad, dropout y kernels optimizados.
- [VS Code: Code Navigation](https://code.visualstudio.com/docs/editing/editingevolved)
  Go to Definition, Go to References, Peek y navegación por símbolos.
- [VS Code: Python debugging](https://code.visualstudio.com/docs/python/debugging)
  Breakpoints, ejecución paso a paso, variables y configuración de debugpy.
- [VS Code: Python](https://code.visualstudio.com/docs/languages/python)
  Selección de intérprete, IntelliSense, pruebas y depuración con la extensión de Python.

## Wisdom (Communities)

- No se ha elegido todavía una comunidad. La prioridad actual es aprender leyendo y depurando
  este repositorio con pruebas locales pequeñas.

## Gaps

- Para una sesión posterior hará falta una fuente primaria específica sobre autograd internals
  y otra sobre profiling, si el objetivo pasa de entender el flujo a optimizarlo.
