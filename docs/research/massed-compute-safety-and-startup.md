# Massed Compute: seguridad, persistencia y arranque de cargas de ML

Fecha de revisión: 8 de agosto de 2026.

## Alcance y conclusión

La documentación oficial que Massed enlaza actualmente se sirve principalmente en
vm-docs.massedcompute.com y en la API de vm.massedcompute.com; el producto y las
políticas están en massedcompute.com. Este informe usa únicamente esas fuentes
oficiales. No usa como autoridad las respuestas de la FAQ cuyo propio contenido
indica que son generadas por IA.

La conclusión operativa es conservadora: una VM GPU de Massed debe tratarse como
un entorno de trabajo efímero. La API dice que terminar una instancia “destruye
todos los datos”, y la interfaz también advierte que la terminación elimina los
datos asociados. Por tanto, datasets, checkpoints, cachés, tokens y resultados
no deben vivir únicamente en el disco de la VM. Para persistencia, Massed
documenta dos caminos: block volumes para conservar modelos entre recreaciones
(en la guía de Ollama) y una VM CPU que expone un directorio por NFS a las GPU
que se lanzan y terminan. ([API de VM](https://vm-docs.massedcompute.com/api/v1),
[elementos de una instancia en ejecución](https://vm-docs.massedcompute.com/docs/running-instances/instance-elements),
[guía oficial de Ollama](https://massedcompute.com/deploy-llm-ollama-gpu-cloud/),
[tutorial oficial de NFS](https://vm-docs.massedcompute.com/docs/advanced-tutorials/nfs-mount/))

Para datos sensibles, la documentación pública no permite afirmar que las VMs
GPU normales sean single-tenant, que el disco efímero esté cifrado, que la
memoria de GPU se limpie con una garantía contractual, ni que exista una
frontera de aislamiento concreta entre tenants. Massed sí publicita bare metal
como hardware dedicado, sin hipervisor ni vecinos, y sus políticas describen
medidas organizativas/técnicas y límites de responsabilidad. Si el modelo o
dataset requiere aislamiento físico o controles verificables, hay que pedir a
Massed evidencia específica o escoger bare metal, no inferirlo del hecho de que
la instancia se llame “VM”. ([bare metal oficial](https://massedcompute.com/products/bare-metal/),
[Privacy Policy](https://massedcompute.com/legal/privacy-policy/),
[DPA](https://massedcompute.com/legal/dpa/))

## 1. Seguridad, aislamiento y riesgos de datos

### Lo que sí declara Massed

- **Bare metal:** Massed lo describe como una máquina completa para un único
  tenant, sin hipervisor ni recursos compartidos; lo orienta expresamente a
  entrenamiento de IA y datos sensibles. Esa afirmación aplica a la oferta de
  bare metal, no automáticamente a las VMs GPU on-demand. ([Bare Metal GPU
  Servers](https://massedcompute.com/products/bare-metal/))
- **Seguridad de información personal:** la Privacy Policy afirma que existen
  medidas técnicas y organizativas razonables, pero también dice que ninguna
  transmisión por Internet ni tecnología de almacenamiento puede garantizar
  seguridad al 100 %, y que la transmisión de información es responsabilidad
  del usuario, que debe acceder desde un entorno seguro. ([Privacy Policy,
  sección 8](https://massedcompute.com/legal/privacy-policy/))
- **Datos personales sujetos al DPA:** cuando aplica el DPA, Massed se
  compromete a medidas adecuadas al riesgo, limita el acceso del personal a
  quienes necesitan conocer los datos y mantiene una lista de subprocessors.
  El DPA también indica que las operaciones primarias de procesamiento están en
  Estados Unidos y permite transferencias fuera del EEE, Reino Unido o Suiza
  con salvaguardas apropiadas. ([DPA, seguridad y subprocessors](https://massedcompute.com/legal/dpa/),
  [lista oficial de subprocessors](https://massedcompute.com/legal/subprocessors/))

### Riesgos y límites que deben asumirse

1. **El aislamiento de una VM no está especificado técnicamente en la
   documentación revisada.** No hay una descripción pública suficiente de
   hipervisor, aislamiento de memoria/VRAM, borrado de GPU, cifrado de discos,
   gestión de claves, snapshots o sanitización del host. La página de bare
   metal sí usa lenguaje explícito de “no shared tenants”; no debe extrapolarse
   a las VMs on-demand.
2. **La terminación es pérdida de datos, no una pausa.** Un checkpoint que solo
   está en /home/Ubuntu, /tmp, una caché local o el storage anunciado por el
   SKU se considera recuperable solo mientras la instancia exista. La API
   especifica que terminate elimina la instancia y destruye todos sus datos.
   ([API: terminate](https://vm-docs.massedcompute.com/api/v1))
3. **La caída de saldo puede acabar en terminación.** La documentación de
   instancias indica que, si falla una recarga, la VM puede quedar detenida; si
   el problema no se resuelve en 12 horas, Massed la termina. ([Estados de una
   instancia](https://vm-docs.massedcompute.com/docs/running-instances/instance-elements))
4. **NFS amplía la superficie de acceso.** El tutorial configura un export de
   /home/Ubuntu/data y una regla UFW para la IP del cliente. Al reemplazar una
   GPU, pide retirar o comentar la IP de la GPU terminada y registrar la nueva.
   La recomendación segura derivada es exportar solo a IPs concretas y no abrir
   el recurso a un rango amplio; el tutorial no documenta cifrado de NFS.
   ([NFS Mount](https://vm-docs.massedcompute.com/docs/advanced-tutorials/nfs-mount/))
5. **El DPA no equivale a una especificación de almacenamiento de checkpoints.**
   Sus compromisos se refieren al procesamiento de datos personales bajo ese
   acuerdo; no describen el ciclo de vida del disco local de una VM ni
   garantizan que un dataset de entrenamiento no personal tenga una retención o
   borrado concretos. Esta es una limitación de alcance, no una afirmación de
   que Massed inspeccione el contenido de los datasets.

## 2. Persistencia de discos y volúmenes

### Disco de la instancia

La API expone storage como parte de las especificaciones del producto, pero la
documentación de terminación es inequívoca: la instancia se elimina y se
destruyen sus datos. No hay que confundir la capacidad local anunciada por un
SKU con un volumen independiente y duradero. ([API de inventario y
terminación](https://vm-docs.massedcompute.com/api/v1))

### Block volumes

La guía oficial de Ollama dice que los modelos guardados en el disco de la VM se
pierden al terminar y recomienda montar un **Massed Compute block volume** en
/root/.ollama antes de descargar los modelos; según la guía, así las descargas
se conservan entre recreaciones de VMs, con coste adicional de almacenamiento.
Esto es evidencia oficial de persistencia para ese caso de uso, pero la
documentación pública revisada no explica de forma completa cómo crear,
adjuntar, desadjuntar, respaldar, cifrar o eliminar esos volúmenes, ni sus
límites de rendimiento o región. Para un dataset de entrenamiento, confirmar
esos detalles antes de transferir datos grandes. ([FAQ de la guía de Ollama](https://massedcompute.com/deploy-llm-ollama-gpu-cloud/))

### VM CPU + NFS

Massed documenta explícitamente el patrón de mantener una VM CPU como
almacenamiento persistente y montar su directorio en cada GPU nueva. El flujo
oficial es:

1. En la VM CPU, instalar nfs-kernel-server, exportar un directorio como
   /home/Ubuntu/data solo a la IP de la GPU y limitar UFW a esa IP.
2. En la GPU, instalar nfs-common, montar el recurso en
   /home/Ubuntu/storage y registrar el montaje en fstab con las opciones
   indicadas por Massed.
3. Al terminar una GPU, quitar su IP del export del servidor antes de autorizar
   la siguiente.

Este patrón conserva los datos al destruir la GPU, pero la VM CPU pasa a ser un
componente crítico: si también se termina o falla, se pierde el almacenamiento
que aloja el NFS. Para checkpoints importantes conviene mantener una segunda
copia externa; la documentación revisada no describe snapshots ni backups
automáticos del NFS. ([tutorial NFS oficial](https://vm-docs.massedcompute.com/docs/advanced-tutorials/nfs-mount/))

## 3. SSH y acceso inicial

La documentación recomienda usar claves SSH, preferentemente RSA o ED25519, y
recomienda una passphrase para la clave privada. Las claves se añaden a la
cuenta y se seleccionan al desplegar; cuando la instancia está en estado
Running, se conecta con la clave asociada. ([SSH Keys oficial](https://vm-docs.massedcompute.com/docs/connect-to-vm/ssh_keys))

También existe una ruta de conexión por usuario, IP y contraseña en la guía de
SSH, y los ejemplos antiguos de la API muestran un campo password en la
respuesta de una instancia. Para una carga de ML, la práctica preferible es
seleccionar una clave al lanzar, proteger su privada con passphrase, limitar
quién la conserva y no distribuir la contraseña de la VM. ([SSH oficial](https://vm-docs.massedcompute.com/docs/connect-to-vm/ssh),
[API de VM](https://vm-docs.massedcompute.com/api/v1))

La API usa tokens Bearer para sus endpoints. Un token de API no debe incluirse
en scripts compartidos, logs, comandos visibles ni imágenes de VM. ([Autenticación
de la API](https://vm-docs.massedcompute.com/api/v1))

## 4. Startup commands y variables de entorno

### Comando de arranque

El endpoint oficial de lanzamiento acepta command, descrito como “the command
you want to run on startup”, además de sshKeys. Por tanto, se puede usar para
bootstrap de una VM GPU: preparar el entorno, instalar dependencias, montar el
almacenamiento y dejar un servicio listo. La documentación no especifica
detalles importantes como shell exacta, timeout, manejo de errores, logs,
idempotencia, ni si el comando se vuelve a ejecutar en cada reboot o solo en el
arranque inicial. Esas propiedades deben probarse con una VM pequeña y no deben
suponerse. ([Launch API](https://vm-docs.massedcompute.com/api/v1))

Si se usa el flujo Docker de la consola, Massed permite modificar el docker run
con flags adicionales y advierte que el comando puede seguir ejecutándose
aunque la VM ya sea accesible; el contenedor puede tardar en quedar activo.
Conviene esperar a una comprobación de salud antes de comenzar una descarga o
un entrenamiento. ([Deploy with Docker](https://vm-docs.massedcompute.com/docs/docker/overview))

Para procesos que deban reiniciarse tras un reboot o recuperarse de un fallo,
la guía oficial de vLLM usa un servicio systemd, lo verifica con
systemctl is-active y consulta journalctl si falla. Ese patrón es más
observable que depender de un único comando de arranque para mantener un
servidor de inferencia. ([guía oficial de vLLM](https://massedcompute.com/deploy-vllm-openai-api-gpu-cloud/))

### env_vars en el lanzamiento

La API moderna también acepta env_vars. Massed documenta que los valores se
escriben en /home/Ubuntu/.env, se regeneran en cada boot a partir de los
valores del lanzamiento y sobrescriben ediciones manuales. Lo más importante
para seguridad es la propia advertencia oficial: en la API se cifran en reposo
con AES-256-GCM y se envían por TLS, pero dentro de la VM el archivo es texto
plano y cualquier persona con acceso al usuario por defecto puede leerlo. No
poner ahí un secreto que deba quedar oculto para ese usuario. ([Environment
Variables at VM Launch](https://vm-docs.massedcompute.com/docs/deploy/environment-variables))

La misma página muestra HF_TOKEN, OPENAI_API_KEY y WANDB_PROJECT como
ejemplos. Para entrenamiento, si se usa HF_TOKEN, debe ser de mínimo alcance,
rotarse cuando termine el experimento y no conservarse en checkpoints, logs o
imágenes. La rotación documentada de env_vars exige lanzar una nueva VM con
valores actualizados; no hay endpoint de actualización para una VM existente.
([Environment Variables at VM Launch](https://vm-docs.massedcompute.com/docs/deploy/environment-variables))

## 5. Comportamiento efímero y ciclo de vida

- **Initializing:** la VM está arrancando la imagen preconfigurada; solo
  después pasa a Running y queda disponible para conectarse. ([Running
  Instance Elements](https://vm-docs.massedcompute.com/docs/running-instances/instance-elements))
- **Restart:** la API tiene un endpoint explícito de reinicio. La documentación
  de env_vars dice que .env se regenera en cada boot, pero no ofrece una
  garantía equivalente para cada archivo local o para el estado de un proceso.
  ([API de VM](https://vm-docs.massedcompute.com/api/v1), [variables de entorno](https://vm-docs.massedcompute.com/docs/deploy/environment-variables))
- **Stopped:** una incidencia de facturación puede detener la VM; si no se
  resuelve en 12 horas, Massed la termina. ([Estados de una instancia](https://vm-docs.massedcompute.com/docs/running-instances/instance-elements))
- **Terminated:** la terminación elimina la VM y destruye todos los datos
  asociados. No tratarla como una suspensión reversible. ([API de terminate](https://vm-docs.massedcompute.com/api/v1))
- **Spot:** la guía de vLLM dice que las instancias spot pueden interrumpirse
  con poca antelación; recomienda spot para desarrollo/pruebas y on-demand para
  producción que requiera alta disponibilidad. ([guía oficial de vLLM](https://massedcompute.com/deploy-vllm-openai-api-gpu-cloud/))

## 6. Recomendación de lanzamiento para este proyecto

Para una carga de entrenamiento de este repositorio, el procedimiento seguro es:

1. Lanzar una GPU on-demand si el entrenamiento no tolera interrupciones; usar
   spot solo con reanudación comprobada y checkpoints frecuentes.
2. Adjuntar una clave SSH al lanzar y verificar primero nvidia-smi. Las guías
   oficiales de QLoRA y vLLM recomiendan una imagen Ubuntu con drivers NVIDIA
   preinstalados y comprobar la GPU antes de instalar o descargar modelos.
   ([guía oficial de QLoRA](https://massedcompute.com/train-llm-lora-qlora-gpu-cloud/),
   [guía oficial de vLLM](https://massedcompute.com/deploy-vllm-openai-api-gpu-cloud/))
3. Ejecutar un preflight antes de descargar datos o iniciar entrenamiento para
   comprobar identidad del host, disco, memoria, red y GPU cuando esté
   disponible. ([preflight oficial de servidor](https://massedcompute.com/run-server-preflight-checks-ubuntu-gpu-cloud/))
4. Montar primero el block volume verificado o el NFS; guardar ahí el manifest,
   shards, checkpoints y resultados. No iniciar el entrenamiento hasta que una
   prueba de escritura y lectura confirme que la ruta es realmente persistente.
5. Hacer el bootstrap idempotente y observable: instalar dependencias, registrar
   logs, comprobar nvidia-smi, validar el almacenamiento y salir con error si
   alguna comprobación falla. Para servidores, usar systemd y health checks.
6. Mantener fuera de env_vars cualquier secreto que no deba ser visible para
   el usuario de la VM; si se usa un token de Hugging Face, aplicar mínimo
   privilegio y rotación.
7. Antes de terminar, verificar que el último checkpoint y los logs existen en
   el almacenamiento persistente y, para resultados importantes, copiar una
   segunda vez fuera de la VM CPU/NFS o del block volume.

## Preguntas que siguen abiertas antes de cargar datos sensibles

La documentación pública revisada no responde de forma suficiente a estas
cuestiones: cifrado de discos/volúmenes y quién controla las claves; snapshots y
backups; retención y sanitización posterior al borrado; aislamiento técnico de
VMs y VRAM entre tenants; rendimiento/IOPS del block volume; límites y regiones
de adjunción; cifrado y controles de acceso de NFS; y semántica exacta del
command ante reboot, error o timeout. Deben confirmarse por escrito con
Massed, junto con el alcance contractual del DPA, antes de procesar secretos,
PII, PHI o propiedad intelectual de alto impacto.

## Fuentes oficiales consultadas

- [Massed Compute VM API](https://vm-docs.massedcompute.com/api/v1)
- [Documentación de SSH Keys](https://vm-docs.massedcompute.com/docs/connect-to-vm/ssh_keys)
- [Documentación de SSH](https://vm-docs.massedcompute.com/docs/connect-to-vm/ssh)
- [Variables de entorno al lanzar una VM](https://vm-docs.massedcompute.com/docs/deploy/environment-variables)
- [Deploy with Docker](https://vm-docs.massedcompute.com/docs/docker/overview)
- [Running Instance Elements](https://vm-docs.massedcompute.com/docs/running-instances/instance-elements)
- [Tutorial NFS Mount](https://vm-docs.massedcompute.com/docs/advanced-tutorials/nfs-mount/)
- [Deploy LLMs with Ollama](https://massedcompute.com/deploy-llm-ollama-gpu-cloud/)
- [Train LLM LoRA Models with QLoRA](https://massedcompute.com/train-llm-lora-qlora-gpu-cloud/)
- [Deploy vLLM with OpenAI API](https://massedcompute.com/deploy-vllm-openai-api-gpu-cloud/)
- [Run Server Preflight Checks](https://massedcompute.com/run-server-preflight-checks-ubuntu-gpu-cloud/)
- [Bare Metal GPU Servers](https://massedcompute.com/products/bare-metal/)
- [Privacy Policy](https://massedcompute.com/legal/privacy-policy/)
- [Data Processing Agreement](https://massedcompute.com/legal/dpa/)
- [Subprocessors](https://massedcompute.com/legal/subprocessors/)
