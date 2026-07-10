<p align="center"><b>peng (鹏)</b> — motor pequeño, modelo inmenso, viento de disco.</p>

# peng

**peng** adapta la tecnología de [colibrì](https://github.com/JustVugg/colibri) (motor de
inferencia MoE en C puro con streaming de experts desde disco) para ejecutar
**MiMo-V2.5 de Xiaomi (311B parámetros, 15B activos)** en hardware doméstico.

El nombre viene del pájaro mitológico chino del *Zhuangzi*: una criatura colosal que se
mantiene en vuelo cabalgando el torbellino — como este motor mantiene "en el aire" un
modelo de 311B parámetros sobre una corriente continua de lecturas de NVMe, en una
máquina con 32 GB de RAM.

Proyecto derivado de colibrì (Apache 2.0, © JustVugg). El README original del motor
upstream está en [`docs/README-colibri-upstream.md`](docs/README-colibri-upstream.md).

## La idea heredada de colibrì

Un MoE activa pocos parámetros por token. En MiMo-V2.5: de 311B totales solo ~15B
trabajan en cada token, y la mayor parte son 8 experts (de 256 posibles) por cada una de
las 47 capas MoE. Entonces:

- la **parte densa** (atención, embeddings, capa 0 densa) vive **residente en RAM a int4**;
- los **12.032 experts enrutados** (47 capas × 256, ~12.6 MB cada uno a int4) viven **en
  disco** (~165 GB) y se leen bajo demanda, con cache LRU por capa + cache de aprendizaje
  que fija los experts más usados en la RAM sobrante.

Coste por token frío: 47 × 8 × 12.6 MB ≈ **4.7 GB de lecturas** → en el NVMe de la máquina
de desarrollo (2.75 GB/s medidos en lecturas aleatorias de 19 MB) el techo físico es
**~0.6 tok/s**, mejorando con cache caliente. No es rápido: es un modelo de 311B
respondiendo en una máquina de sobremesa.

## Por qué MiMo-V2.5

Elegido tras comparar candidatos (GLM-5.2 744B, Hunyuan Hy3 295B, Qwen3.5-122B):

| criterio | MiMo-V2.5 | GLM-5.2 | Hy3 | Qwen3.5-122B |
|---|---|---|---|---|
| disco int4 | **~165 GB** ✓ | ~370 GB ✗ (no cabe) | ~157 GB | ~65 GB |
| activos/token (→ velocidad en disco) | **15B** | 40B | 21B | 10B |
| router | **idéntico a colibrì** (sigmoid noaux_tc) | idéntico | distinto | distinto |
| atención | GQA simple ✓ | MLA+DSA (ya hecho) | GQA, experts heterogéneos | Gated DeltaNet (muy difícil en C) |
| formato checkpoint | **FP8 128×128, igual que GLM** ✓ | — | distinto | distinto |

MiMo-V2.5 maximiza el reuso de colibrì: router, converter FP8→int4, streaming, kernels
AVX2 int4/int8 — todo se hereda casi intacto. Solo cambia la atención (y es *más simple*
que la MLA+DSA de GLM).

## Arquitectura de MiMo-V2.5 (verificada contra config.json y modeling oficial)

- 48 capas: capa 0 densa (MLP 16384), capas 1–47 MoE (256 experts top-8, inter. 2048, sin shared expert)
- Atención híbrida: **9 capas full** (índices 0,5,11,17,23,29,35,41,47; 64Q/4KV heads,
  theta 10M) y **39 capas sliding-window 128** (64Q/8KV heads, theta 10k, con
  *attention sink bias* por cabeza)
- head_dim 192 (K) / 128 (V); RoPE parcial no-interleaved sobre los primeros 64 dims
- QKV fusionado en un solo tensor; `o_proj` aparte y en bf16 en el checkpoint (resto FP8 e4m3)
- V escalado ×0.707 antes de atención (se pliega en los pesos al convertir)
- Vocabulario 152.576, byte-BPE estilo GPT-2/Qwen

Consecuencia bonita de la ventana deslizante: 39 de 48 capas necesitan KV-cache de solo
128 tokens — el contexto largo sale casi gratis en RAM comparado con un modelo full-attention.

## Método (el de colibrì): validación token-exacta antes que nada

Nada se da por bueno sin reproducir **bit a bit** los tokens de la implementación de
referencia (`transformers` + `modeling_mimo_v2.py` oficial):

1. **Oráculo tiny**: modelo aleatorio minúsculo con la arquitectura real (patrón híbrido,
   RoPE parcial, sink bias, router) generado con el código oficial → tokens de referencia.
2. **Motor C** (`c/mimo.c`) debe clavar teacher-forcing 32/32 y greedy 20/20 contra el
   oráculo, con secuencias que crucen la frontera de la ventana deslizante.
3. Solo entonces: converter del checkpoint real (316 GB FP8 → ~165 GB int4) y primer chat.

## Estado — bitácora

### 2026-07-10 — nace el proyecto

- **Colibrì validado en la máquina de desarrollo** (Xeon W-2140B 8c/16t, 32 GB RAM,
  WSL2): compilación limpia, tests C 3/3, tests Python 25/25, y el motor GLM reproduce
  el oráculo `transformers` token-exacto — teacher-forcing 32/32, greedy 20/20 — tanto
  en el modelo tiny como en un fixture de 313M parámetros con las formas reales.
- **Disco medido con el patrón de acceso real** (lecturas aleatorias de 19 MB, 8 hilos,
  `iobench` del upstream sobre fichero de 22 GiB de datos aleatorios): 1.78 GB/s
  buffered, **2.75 GB/s O_DIRECT** (satura a 8 hilos). Primer intento con fichero de
  ceros dio 8.5 GB/s falsos por la cache del host — moraleja: bench siempre con datos
  aleatorios y caches frías.
- **GLM-5.2 descartado en esta máquina** (~370 GB int4 > 222 GB libres); evaluados
  Hy3 y Qwen3.5-122B; **elegido MiMo-V2.5** (tabla arriba).
- **Hechos de arquitectura verificados** contra `config.json` crudo y
  `modeling_mimo_v2.py` oficial (85 kB). Cinco sorpresas que el resumen de la model card
  no contaba: QKV fusionado, sink bias en capas SWA, V×0.707, KV heads distintos por tipo
  de capa (4 full / 8 SWA), y `o_proj` fuera de la cuantización FP8.
- **Diseño aprobado y spec escrito**:
  [`docs/superpowers/specs/2026-07-10-peng-mimo-design.md`](docs/superpowers/specs/2026-07-10-peng-mimo-design.md).
  Enfoque: motor hermano `c/mimo.c` (precedente: `olmoe.c` del upstream), headers y
  kernels compartidos intactos, MTP y multimodal fuera del v1.
- Siguiente: plan de implementación detallado, luego fase 1 (oráculo + tokenizer).

## Hoja de ruta

- [x] Validar el motor colibrì upstream en la máquina de desarrollo
- [x] Medir el disco con el patrón de acceso del motor
- [x] Elegir modelo objetivo y verificar su arquitectura contra el código oficial
- [x] Spec de diseño aprobado
- [x] Fase 1 — oráculo tiny (`tools/make_mimo_oracle.py`) + validación del tokenizer
- [x] Fase 2 — `c/mimo.c`: atención GQA híbrida → **TF 32/32, greedy 20/20**
- [x] Fase 3 — streaming de experts + cuantización (int8 token-exacto; packing lossless)
- [x] Fase 4 — suite completa verde (C 3/3, Python 32/32) + fixture 396M (**TF 20/20**)
- [x] Fase 5 — converter `--arch mimo` (container ≡ runtime-quant, token a token)
- [ ] Fase 6 — descarga real (316 GB), conversión y primer chat con 311B en 32 GB de RAM

### Resultados de validación (2026-07-10)

| Gate | Resultado |
|---|---|
| Tokenizer C vs HF AutoTokenizer | 6+4 casos unicode, ids idénticos |
| Tiny oracle (6 capas, patrón híbrido real) TF / greedy, f32 | **32/32 · 20/20** |
| Tiny, experts int8 | 32/32 · 20/20 (sin un solo flip) |
| Tiny, experts int4 packed vs sin packing | byte-idéntico (packing lossless) |
| Kernels enteros IDOT vs dequant exacto (int8) | tokens idénticos |
| LRU con evicción forzada (`CAP_RAISE=0`, cap=2) | 20/20 exacto, hit 88%→81% |
| Fixture 396M (formas reales) TF / greedy, f32 | **20/20 · 8/8** |
| Container convertido vs cuantización runtime (tiny y fixture) | tokens idénticos, también bajo evicción |
| ASan/UBSan (TF, greedy, spec-decode, evicción) | limpio |

Pendiente para la Fase 6, además del disco: sustituir el chat template heredado de GLM
por el oficial de MiMo (marcado como TODO bloqueante en `mimo.c`) y ajustar los defaults
de sampling al `generation_config` de MiMo.

## Licencia

Apache 2.0, como el colibrì original. Los pesos de MiMo-V2.5 los publica Xiaomi bajo su
propia licencia en [Hugging Face](https://huggingface.co/XiaomiMiMo/MiMo-V2.5).
