# Misión: construir y entrenar un transformer desde cero

## Por qué
Convertir lo aprendido en _Build a Large Language Model (From Scratch)_ en comprensión operativa: ser capaz de diseñar, implementar, depurar, entrenar y explicar un decoder-only transformer moderno sin depender de una implementación prefabricada.

## El éxito se verá así
- Implementar en PyTorch cada componente importante y justificar sus formas, parámetros y función.
- Demostrar con tests que el modelo es causal, que los componentes son correctos y que el entrenamiento se puede reanudar de forma reproducible.
- Entrenar primero un modelo diminuto y después uno de decenas de millones de parámetros con curvas sanas y muestras progresivamente coherentes.
- Tomar una decisión informada sobre si merece la pena escalar el mismo sistema hasta aproximadamente 350M parámetros.

## Restricciones
- El usuario escribirá el código; el agente propondrá contratos, ejercicios, tests, revisiones y pistas graduadas.
- PyTorch será la abstracción principal; las implementaciones optimizadas solo se introducirán después de construir y validar su equivalente ingenuo.
- El presupuesto de GPU y el tiempo semanal aún están por concretar.

## Fuera de alcance
- Empezar directamente con un entrenamiento de 350M parámetros.
- Añadir GQA, Muon, Differential Attention, DDP, SFT o GRPO antes de tener un baseline pequeño correcto.
- Competir con modelos comerciales o interpretar una loss decreciente como prueba suficiente de corrección.
