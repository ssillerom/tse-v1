# Shards binarios de tokens con manifest versionado

La preparación materializa el corpus como shards binarios `uint16` y publica un
`manifest.json` versionado, en lugar de tokenizar durante el entrenamiento o depender del
formato de Hugging Face. Esta decisión reduce el coste repetido y desacopla entrenamiento de la
fuente, a cambio de fijar un contrato de almacenamiento y limitar el tokenizer actual a IDs que
quepan en `uint16`; el manifest conserva la procedencia y los parámetros necesarios para
detectar incompatibilidades.
