# findings — hallazgos del proyecto peng

Registro de sorpresas y descubrimientos técnicos durante el port de colibrì a MiMo-V2.5.
Cada entrada: el hallazgo técnico + una explicación para no entendidos.

---

## Arquitectura de MiMo-V2.5

### 1. El código de referencia oficial solo funciona con transformers 5.0–5.1

El `modeling_mimo_v2.py` que publica Xiaomi necesita a la vez una API que nació en
transformers 5.0 (`standardize_rope_params`) y otra que murió en la 5.2 (el argumento
`input_embeds` de `create_causal_mask`, renombrado después). Probamos las ruedas de
4.57 a 5.13: solo 5.0.0 y 5.1.0 satisfacen ambas. Usamos 5.1.0 en un venv dedicado
(`~/mimo-venv`) sin tocar ni una coma del código oficial.

> **En cristiano:** el "manual de instrucciones" del modelo solo se puede leer con una
> versión muy concreta de la librería de referencia — ni la que dice su propia etiqueta
> (4.57) ni la última. Encontramos la ventana exacta probando una por una.

### 2. Las capas de ventana deslizante tienen SU PROPIA geometría de atención

No solo cambia la ventana: las capas SWA usan más cabezas KV (8 vs 4 en el modelo real;
4 vs 2 en el tiny) y sus propias dimensiones de cabeza (`swa_head_dim`, `swa_v_head_dim`
— y ojo: si no están en el config, `swa_v_head_dim` hereda de `swa_head_dim`, NO de
`v_head_dim`). Lo descubrió el modelo tiny: su `o_proj` SWA es [128,192], no [128,128].

> **En cristiano:** el modelo alterna dos tipos de "vista": 9 capas miran toda la
> conversación y 39 solo las últimas 128 palabras. Resulta que además usan gafas de
> distinta graduación — si el motor asume la misma para todas, los números no cuadran.

### 3. Sink bias: un "desagüe de atención" solo en capas SWA

Cada cabeza de las capas de ventana tiene un logit extra aprendido que participa en el
softmax pero cuya probabilidad se descarta — un sumidero que absorbe atención sobrante.
Las capas full no lo tienen (`add_full_attention_sink_bias: false`).

> **En cristiano:** cuando el modelo solo ve las últimas 128 palabras, necesita un sitio
> donde "aparcar" la atención que no quiere gastar en ninguna de ellas. Es un truco de
> estabilidad; sin implementarlo, los resultados divergen.

### 4. Los valores (V) se multiplican por 0.707 antes de todo

`attention_value_scale: 0.707` (≈ 1/√2) se aplica al vector V justo tras la proyección,
antes de entrar en la cache KV. Lo aplicamos en runtime (coste despreciable) para que el
mismo código sirva al oráculo f32 y al contenedor cuantizado.

### 5. QKV fusionado y o_proj fuera de la cuantización FP8

El checkpoint guarda q, k y v en un solo tensor por capa (`qkv_proj`, layout
"fused_qkv") — y todos los `o_proj` están en la lista `ignored_layers`: viajan en
bf16 mientras el resto va en FP8 con escalas de bloque 128×128. El converter tendrá
que tratarlos distinto.

### 6. Config con trampas pequeñas

- `rope_theta` y `partial_rotary_factor` aparecen DUPLICADOS (top-level y dentro de
  `rope_parameters`); `swa_rope_theta` solo top-level.
- `routed_scaling_factor` es `null` JSON (no ausente) → hay que tratar null como 1.0.
- El epsilon de normalización se llama `layernorm_epsilon` (GLM usa `rms_norm_eps`).
- RoPE es parcial (primeros 64 de 192 dims) y NO-interleaved (estilo NeoX/rotate_half),
  con theta distinto por tipo de capa: 10.000.000 (full) vs 10.000 (SWA).

> **En cristiano:** el fichero de configuración tiene varios campos con nombres o
> formatos ligeramente distintos a los de otros modelos. Cada uno es una mina si el
> motor asume el formato "habitual".

### 6b. La atención C clavó la validación a la primera

Con los hechos verificados de antemano (orden exacto: proyección fusionada → V×0.707 →
RoPE parcial → cache → ventana p−128+1 → sink en denominador → o_proj), la
implementación C pasó teacher-forcing 32/32 y greedy 20/20 en el primer intento, sin
una sola iteración de debugging. Verificado dos veces por un revisor independiente.

> **En cristiano:** cuando inviertes el tiempo en leer el plano original con lupa antes
> de construir, la pieza encaja a la primera. Todo el trabajo de arqueología del config
> y el modeling (hallazgos 1–6) se cobró aquí.

Nota de fragilidad anotada: el rope_dim por tipo de capa se deriva por reescala entera
desde el valor full; si un config futuro tuviera `swa_head_dim ≠ head_dim` con factor
no exacto, podría diferir en ±1 del cálculo de la referencia. Los configs actuales
(tiny y real) son inmunes; el oráculo lo detectaría al instante en cualquier caso.

### 6c. Cuantización en el tiny: int8 exacto, int4 voltea un argmax, packing sin pérdidas

Matriz completa en el oráculo tiny: int8 reproduce el f32 **token-exacto** (32/32,
20/20 — ni un flip); int4 voltea UNA posición de 32 (el resto de la divergencia greedy
es cascada autoregresiva de ese único flip); int4 empaquetado ≡ int4 sin empaquetar
byte a byte (packing lossless); kernels enteros IDOT ≡ dequant exacto a int8; todo
determinista. Verificado independientemente por el controlador.

> **En cristiano:** comprimir los pesos a la mitad (int8) no cambió ni un solo
> resultado; comprimir a un cuarto (int4) cambió uno de 32 — y en un modelo de pesos
> aleatorios donde cualquier ruido voltea decisiones al borde. En el modelo real,
> entrenado, los márgenes son mucho mayores.

### 6d. El cap de cache se auto-sube silenciosamente (gotcha de benchmark)

Pedir `cap=2` por CLI no da cache de 2: el auto-raise del upstream (feature del
2026-07-10) lo sube hasta llenar el presupuesto de RAM. Para forzar el cap pedido:
`CAP_RAISE=0`. Con él, la evicción LRU real mantiene tokens exactos (20/20) con
hit-rate 88%→81% — streaming bajo presión validado.

## Tokenizer

### 7. Formato moderno de merges: string única, no pares

GLM guarda las reglas BPE como pares `["Ġ","Ġ"]`; MiMo (tokenizers ≥0.20) las guarda
como una sola string `"Ġ Ġ"`. El parser C original habría dereferenciado NULL. Añadido
soporte para ambos formatos en `tok.h` (+ fallo explícito si una entrada viene sin
separador — hallazgo del code review).

### 8. Dígitos: de uno en uno, no de tres en tres

La regex de pretokenización de GLM agrupa números en trozos de hasta 3 dígitos
(`\p{N}{1,3}`, estilo cl100k); la de MiMo/Qwen los toma de uno en uno (`\p{N}`).
Auto-detectado del tokenizer.json. Validado contra la librería oficial con 6 casos
unicode + 4 adversariales (emoji ZWJ, run de 13 dígitos…): ids idénticos.

> **En cristiano:** dos modelos pueden trocear "2026" de forma distinta: uno como
> "202"+"6", otro como "2"+"0"+"2"+"6". Si el motor trocea distinto que el modelo
> espera, entiende otra cosa. Nota pendiente: la detección es una búsqueda de texto
> simple — endurecer antes de soportar una tercera familia.

## Converter (FP8 → contenedor int4/int8)

### 8b. El converter valida contra sí mismo: container ≡ cuantización runtime

El gate del converter no es "se parece": el contenedor pre-cuantizado debe producir
EXACTAMENTE los mismos tokens que el motor cuantizando al vuelo los mismos pesos —
misma aritmética (`np.rint` ≡ `lrintf`), mismas escalas por fila. Verificado en tiny y
en el fixture de 396M, también bajo evicción de cache. Además: el path GLM quedó
probado byte-idéntico antes/después del refactor con un harness A/B sobre shards
sintéticos.

### 8c. Trampas del checkpoint real anotadas para la conversión de 316 GB

- Los shards de MiMo llegan hasta **34.4 GB** (GLM ~5 GB): el guardarraíl
  `--min-free-gb` por defecto (20) es INSUFICIENTE — subirlo al lanzar.
- `save_file` no es atómico: un corte durante la escritura final deja un shard de
  salida truncado que el resume saltará; si pasa, borrar el último `out-*` y relanzar.
- Con ~4 MB/s medidos en esta línea, la descarga sola ≈ 22 h (resumible en cualquier
  punto; los shards de visión/audio/MTP nunca se descargan gracias al filtro sobre
  el weight_map).
- `--io-bits 16` estaba silenciosamente roto en el upstream (overflow de astype int8);
  ahora bits≥16 → f32 explícito. Los defaults de `--arch mimo` (dense int8, experts
  int4, io f32) reproducen el punto de operación validado del motor.

> **En cristiano:** convertir 316 GB llevará un día entero de descarga. Todo el
> esfuerzo de esta fase fue asegurar que, cuando ese día termine, el resultado sea
> correcto a la primera — cada pieza del pipeline demuestra producir bits idénticos
> a la referencia antes de tocar el modelo real.

### 8d. Chat template de MiMo: ChatML con trampas propias

- Los turnos se unen SIN salto de línea tras `<|im_end|>` (el ChatML de Qwen lleva `\n`
  — venir de ahí garantiza el error) y sin token BOS.
- Sistema por defecto: "You are MiMo, a helpful AI assistant engineered by Xiaomi."
- Thinking viene ACTIVADO por defecto en el template oficial (`<think>` tras
  `<|im_start|>assistant\n`); con razonamiento largo puede comerse todo el presupuesto
  de tokens antes de la respuesta visible.
- Los ids de stop se arman POR NOMBRE del tokenizer (`<|endoftext|>`, `<|im_end|>`),
  no por id del config — el config del snapshot solo declara uno de los dos.
- `generation_config` oficial: temperature 1.0, top_p 0.95 (y un `do_sample: false`
  contradictorio que ignoramos).
- Validación sin modelo: `TEMPLATE_DUMP=1` vuelca el prompt renderizado antes de
  tokenizar y un test lo compara contra `apply_chat_template` de HF — 5/5 casos.

> **En cristiano:** cada modelo tiene su "protocolo de conversación" — dónde van las
> etiquetas de quién habla. Un espacio de más y el modelo balbucea. Lo comprobamos
> contra la herramienta oficial sin necesitar el modelo descargado.

### 8e. La estimación de descarga era 15× pesimista

Con HF anónimo una prueba corta midió ~4 MB/s → estimamos 22 h. La descarga real
sostiene **~59 MB/s** con 2 streams: shard de 34.4 GB en 9.7 min → ~2-4 h el total.
Moraleja: no estimar ancho de banda con una conexión fría y rate-limit de handshake.

### 8f. El checkpoint real corta pares peso/escala en las fronteras de shard

Primer contacto con el checkpoint FP8 de verdad: crash. El shard 0 trae 4096 pesos
pero 4095 escalas — el escritor del checkpoint partió un par `weight`/`weight_scale_inv`
justo en el límite del archivo (la escala quedó en el shard siguiente). Ningún test
podía verlo: nuestras fuentes de prueba eran bf16 (sin escalas separadas). Fix:
resolver la escala vía el índice del repo y traer solo sus bytes por HTTP Range
(las escalas son KB). El shard de 34 GB descargado se conserva y se reutiliza.

> **En cristiano:** el modelo viene en 16 "cajas" y el embalador cortó una pieza en
> dos cajas distintas. Nuestro desembalador asumía piezas completas por caja. Es
> EXACTAMENTE el tipo de fallo para el que montamos conversión resumible: se arregla
> el desembalador y se continúa donde iba, sin re-descargar nada.

### 11b. Cuidado con monitores que se buscan a sí mismos

El monitor del convertidor tenía dos bugs de novato: su `$(...)` se expandió una capa
de shell antes de tiempo (quedó vigilando strings vacíos), y su `pgrep -f convert_fp8`
se encontraba A SÍ MISMO (el patrón estaba en su propia línea de comando) — reportaba
"VIVO" con el convertidor muerto. Reglas: scripts de monitor a archivo (no inline con
anidamiento de comillas), y patrones de pgrep que no aparezcan en el comando del
propio monitor (`pgrep -f 'python.*convert_fp8'`).

## Entorno / herramientas

### 9. Benchmark de disco con ceros = mentira

Primer iobench sobre fichero de ceros: 8.5 GB/s "imposibles" (más que el bus físico).
Causa: cache del host Windows sobre el VHDX. Con datos aleatorios y caches frías:
2.75 GB/s reales (O_DIRECT, 8 hilos). Regla: benchmarks de disco siempre con datos
aleatorios y tras reinicio de WSL.

### 10. El tamaño "lógico" engaña con OneDrive

37.6 GB "en disco" eran en su mayoría placeholders solo-en-línea (0 bytes reales).
Borrarlos no liberó casi nada. Medir con `Length` de PowerShell cuenta el tamaño
lógico, no el físico.

### 11. WSL: quirks que costaron tiempo

- `/tmp` es tmpfs y se borra cuando la VM de WSL se apaga entre comandos — nada de
  estado entre invocaciones ahí; usar `~` o `/mnt/c`.
- `echo EXIT=$?` dentro del mismo `wsl -e bash -c "..."` desde PowerShell/Git Bash
  reporta mal el código de salida; comprobar desde la shell exterior.
- El VHDX de WSL crece pero no encoge (sparse deshabilitado por defecto por riesgo de
  corrupción); compactar requiere admin (`Optimize-VHD`) o export/import.

### 12. Validación token-exacta como método

Todo cambio se acepta solo si el motor C reproduce bit a bit los tokens de la
implementación de referencia (teacher-forcing 32/32, greedy 20/20). El modelo tiny
aleatorio con arquitectura real es la herramienta: barato de generar, determinista
(mismo seed → mismos bytes), y expone errores que un modelo entrenado escondería
(pesos aleatorios → márgenes mínimos → cualquier desviación voltea el argmax).

> **En cristiano:** antes de descargar 316 GB, construimos una maqueta a escala del
> modelo con las mismas piezas y comprobamos que nuestra copia del motor produce
> EXACTAMENTE los mismos resultados que el original, número a número. Si la maqueta
> cuadra perfecta, el grande cuadrará.
