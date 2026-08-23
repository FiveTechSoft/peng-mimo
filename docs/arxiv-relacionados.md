# Papers arXiv relacionados — candidatos para peng-mimo (2026-08-23)

Mapeo al estado medido del repo (findings §49/§50): régimen frío **disco-bound**
(~55-70% del wall), matmul ya arreglado con MM_THREADS, MTP/DRAFT rechazado hoy
por costo de verify sobre disco, PILOT recall ~71.6% heredado de colibri.

## Ranking por ROI esperado

### 1. FATE — Cross-Layer Gate ([2502.12224](https://arxiv.org/abs/2502.12224))
- **Sin entrenamiento**: usa los inputs de la gate de la capa adyacente para
  predecir los expertos de la siguiente (nuestro PILOT hace lo mismo con ~71%
  recall); su extra es el **cache shallow-favoring**: las capas tempranas reusan
  siempre los mismos expertos → se pinean preferentemente → hit 99%.
- Reportan hasta 4.1× decoding vs load-on-demand, sin pérdida de calidad,
  escalando con presupuesto de memoria.
- **Por qué primero**: I/O puro, lossless, se valida OFFLINE con nuestro
  `.coli_usage` antes de tocar el motor. Ataca directamente el polo dominante.
- Verificación propia: `scripts/layer_reuse.py` sobre `.coli_usage`
  (§51) — ¿las capas tempranas de MiMo concentran reuso entre turnos?

### 2. MoE-SpeQ ([2511.14102](https://arxiv.org/pdf/2511.14102))
- Decodificación especulativa donde un draft fuertemente cuantizado predice
  (>90%) los top-k del target → prefetch proactivo durante la ventana de I/O +
  **gobernador adaptativo** que ajusta el grado de especulación según
  PCIe/I/O/acceptance.
- **Para nosotros**: resucita `DRAFT=2` que hoy pierde (§50.2): especular SOLO
  cuando la ventana de I/O lo permite y usar el router del draft para traer
  antes los expertos del verify. Esfuerzo medio-alto; segundo.

### 3. ReMoE ([2605.27081](https://arxiv.org/pdf/2605.27081v1))
- Fine-tuning del ROUTER para maximizar reuso de expertos residentes en
  memoria limitada. Versión entrenada de nuestro `CACHE_ROUTE` (rechazado por
  calidad): sin el penalty porque el router aprende, no se fuerza.
- Router de MiMo es pequeño (~50M params total) pero requiere pipeline de
  calibración + gates de calidad completos. Tercero; complementa UNION179.

### Contexto / ideas secundarias
- [APEX](https://arxiv.org/abs/2608.11688) (ago 2026) — prefetch adaptativo edge;
  benchmark comparativo para TRAJ/PILOT.
- [PreScope](https://arxiv.org/pdf/2509.23638v1) — scheduling de prefetch
  resource-constrained.
- [OD-MoE](https://arxiv.org/pdf/2512.03927v1) — predictor SEP con modelo sombra;
  alternativa sin cache (filosofía TRAJ/pathpack llevada al extremo).
- [MoBiLE](https://arxiv.org/html/2510.12357v1) — big-little experts en GPU
  consumidor; diseño alternativo de tier VRAM↔disco.
- [Estudio empírico MoE en consumer/edge](https://arxiv.org/html/2606.21428v1) —
  metodología de benchmarks de referencia.

## Validación offline con datos propios (2026-08-23)

Antes de tocar el motor, dos hipótesis testeadas contra `.coli_usage` real
(124k selecciones históricas):

1. **FATE shallow-favoring: REFUTADO para MiMo.** Las capas tempranas son las
   *menos* concentradas (top-40/256 cubre 81% en L1-8 vs 87% en L20-47;
   `scripts/layer_reuse.py`). El pin sesgado a capas tempranas no paga aquí.
2. **Política de pin ya óptima**: global-por-frecuencia = cuota-per-capa al
   mismo presupuesto (27.0% vs 26.8%; `scripts/pin_policy_test.py`). La
   cobertura escala lineal con RAM: 700 slots→41%, 1500→61%, 1900→68%
   (confirma la proyección de la upgrade a 64 GB).

3. **MTP re-medido con TEMP=0**: aceptación real **36–42%** (los 8–22% de §50.2
   eran sampling TEMP=1). El bloqueo del draft es puramente el costo I/O del
   verify → exactamente lo que ataca MoE-SpeQ.

**Veredicto final — orden de ataque:**
1. **MoE-SpeQ**: gobernador adaptativo + usar el router del draft para
   prefetchear los expertos del verify durante la ventana de I/O. Con
   aceptación 40% real, eliminar el I/O extra convierte DRAFT=2 en +20-30%.
2. ReMoE (router training): después, si se monta pipeline de calibración.
3. FATE/OD-MoE: ideas de predicción ya cubiertas por PILOT/TRAJ en este motor.
