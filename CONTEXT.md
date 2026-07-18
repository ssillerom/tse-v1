# Preparación de datos para pretraining

Este contexto transforma documentos de una fuente externa en secuencias causales reproducibles
que un modelo de lenguaje puede consumir sin conocer el formato original del corpus.

## Language

**Documento fuente**:
Registro de texto individual recibido desde el dataset externo antes de tokenizarlo.
_Avoid_: ejemplo, fila

**Flujo de tokens**:
Secuencia ordenada de IDs producida al concatenar documentos tokenizados y separarlos con EOT.
_Avoid_: texto tokenizado, array global

**Shard**:
Fragmento binario acotado de un flujo de tokens perteneciente a un único split de salida.
_Avoid_: fichero de datos, chunk

**Manifest de datos**:
Contrato versionado que identifica la fuente, tokenizer, almacenamiento, partición, conteos y
shards de un dataset preparado.
_Avoid_: metadata, configuración

**Split de salida**:
Partición disjunta del dataset preparado, actualmente `train` o `validation`.
_Avoid_: carpeta, subset

**Secuencia de pretraining**:
Muestra de longitud fija formada por un input causal y su target desplazado un token.
_Avoid_: documento, batch

**Ventana causal**:
Región contigua de `seq_len + 1` tokens utilizada para construir una secuencia de pretraining.
_Avoid_: slice, segmento
