# Shards binarios de tokens con manifest versionado

La preparación materializa el corpus como shards binarios `uint16` y publica un
`manifest.json` versionado, en lugar de tokenizar durante el entrenamiento o depender del
formato de Hugging Face. Esta decisión reduce el coste repetido y desacopla entrenamiento de la
fuente, a cambio de fijar un contrato de almacenamiento y limitar el tokenizer actual a IDs que
quepan en `uint16`; el manifest conserva la procedencia y los parámetros necesarios para
detectar incompatibilidades.

El contrato de escritura actual es manifest v3. Cada shard declara el SHA-256 de sus bytes
publicados; el lector comprueba primero tamaño y después checksum antes de exponerlo al
dataset. Así se detectan corrupción silenciosa y sustituciones que conservan el mismo tamaño.
Los manifest v2 continúan siendo legibles para no invalidar preparaciones existentes, aunque
sin garantía de checksum. Calcular y verificar SHA-256 añade lecturas secuenciales de cada
shard durante la preparación y al abrir cada manifest; se acepta ese coste para que ningún
lector exponga datos sin comprobar su integridad.
