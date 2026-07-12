# Informe de mejoras — FiveTechSoft/peng-mimo

Revisión técnica puntual (no auditoría formal ni certificación de seguridad).

## Ficha

| Campo | Valor |
|---|---|
| Repositorio | https://github.com/FiveTechSoft/peng-mimo |
| Rama auditada | `peng` |
| Commit auditado | [`1482e26`](https://github.com/FiveTechSoft/peng-mimo/commit/1482e26fb9f19ef676575368228c54a918941a18) (2026-07-11) |
| Recheck local | `187b452` + worktree (2026-07-12) |
| Alcance | motor C, carga/conversión, tests, CLI, API, web, docs, seguridad práctica, rendimiento |
| Cambios hechos por el auditor en el repo | ninguno |

## Resumen ejecutivo

`peng-mimo` ejecuta MiMo-V2.5 311B en hardware de consumo vía streaming de expertos desde NVMe. El trabajo forense (arquitectura, QKV, tokenizer, cuant, I/O) es fuerte. El motor `mimo` compila y los tests C básicos pasan.

**Estado de producto: experimental.** Bloqueos antes de recomendarlo ampliamente:

1. Endurecer carga de modelos no confiables y corregir overflow de plantilla de chat.
2. Gate token-exact reproducible en CI para `mimo`.
3. Superficie pública Peng/MiMo (no GLM/Colibri por defecto).
4. Conversión 152 GB atómica y versionada.
5. Unificar comandos, variables, defaults y documentación.

## Severidad

| Nivel | Significado |
|---|---|
| **P0** | Corrupción de memoria, integridad de artefacto o flujo principal incorrecto |
| **P1** | Fiabilidad, reproducibilidad, RAM, despliegue o UX principal |
| **P2** | Calidad, mantenimiento, docs, compatibilidad |
| **P3** | Optimizar solo con evidencia |

## Tabla de estado (F-xx)

Actualizada al recheck local. `open` = sin fix; `partial` = algo movido; `fixed` = cerrado en HEAD worktree.

| ID | Sev | Estado | Notas recheck |
|---|---|---|---|
| **F-01** | P0 | partial → in progress | Validación básica de header/offsets/dtypes en `st.h` (esta PR) |
| **F-02** | P0 | **fixed** (esta PR) | `mimo_turn_render` seguro; rechazo controlado; test overflow |
| **F-03** | P0 | open | README 32/32 vs findings 31/32–18/20; sin CI MiMo exact |
| **F-04** | P0 | open | API/web/setup/coli defaults GLM |
| **F-05** | P0 | open | `save_file` directo al destino en converter principal |
| **F-06** | P1 | open | `/resolve/main` mutable en conversión |
| **F-07** | P1 | open | `make check` → glm + deps no fijadas |
| **F-08** | P1 | open | Stop web no cancela child |
| **F-09** | P1 | open | `/health` no mira `poll()` |
| **F-10** | P1 | open | Bind externo sin fail-closed fuerte |
| **F-11** | P1 | open | KV SWA a `max_t` (ring diferido) |
| **F-12** | P1 | open | PILOT=0/TOPP/NUC knobs confusos |
| **F-13** | P1 | open | Quickstart cwd; `/more` vs `/mas` |
| **F-14** | P1 | open | Planner MLA/GLM |
| **F-15** | P1 | **fixed** (post-auditoría) | `Makefile`: `mimo` enlaza `$(CUDA_OBJ)`; CUDA medido en RTX 3060 |
| **F-16** | P1 | open | Sin tag/manifest/format_version |
| **F-17** | P1 | open | Sin constraints Python |
| **F-18** | P1 | open | Docs/estado contradictorios |
| **F-19** | P2 | open | Suite calidad int4 + benches trazables |
| **F-20** | P2 | open | SSE/a11y web |
| **F-21** | P2/P3 | open | pread_exact, LSan, etc. |

## Mapa a roadmap de producto

```text
R0  F-01, F-02, F-05, F-06
R1  F-03, F-07, F-17
R2  F-04, F-12, F-13, F-14, F-15(docs)
R3  F-08, F-09, F-10, F-20
R4  F-11
R5  F-16, F-18
R6  F-19, F-21 + opts medidas
```

Ver también: roadmap de producto (si existe en `docs/`) y `roadmap.md` (GPU/speed, anexo experimental).

## Hallazgos detallados

### F-01 — P0 — Loader incompleto ante archivos no confiables

**Evidencia (audit @ 1482e26):** `st_init` confía en header/JSON/shapes/offsets; `st_read_f32` copia `t->nbytes` sin capacidad del destino; `json.h` parcial; token IDs no validados vs vocab.

**Mejora:** límites de header, `hlen` vs `file_size`, aritmética overflow-safe, `nbytes` coherente con dtype (salvo U8 packed), offsets dentro del archivo, duplicados, capacidad en APIs de lectura, corpus malformado + ASan.

**Criterio:** artefactos corruptos → error limpio; modelo válido sin regresión de tokens.

### F-02 — P0 — Overflow en builder de plantilla

**Evidencia:** `mimo_turn_render` sumaba retornos de `snprintf` (longitud *necesaria*) y el caller usaba `bl` como tamaño real. ASan con ~70k chars: heap-buffer-overflow, exit 134.

**Fix (esta PR):** append seguro; retorno `-1` si no cabe; SERVE/TEMPLATE_DUMP rechazan sin leer fuera de buffer; test de overflow.

### F-03 — P0 — Exactitud sin gate reproducible

README 32/32 TF y 20/20 greedy vs findings 31/32 y 18/20 post-MTP. `make check` no es gate `mimo` limpio.

### F-04 — P0 — Superficie pública GLM/Colibri

API `render_chat` `[gMASK]<sop>`, engine default `glm`, web branding Colibri, setup/Makefile GLM.

### F-05 — P0 — Converter reanudable puede aceptar truncado

Escritura directa a `outp` + skip si existe. Usar `.tmp` + validar + `os.replace` + manifest.

### F-06 — P1 — Revisiones mutables en conversión

`/resolve/main` durante horas. Fijar SHA al inicio (`--revision`).

### F-07 — P1 — `make check` no limpio/reproducible MiMo

Deps Python no fijadas; check porta `glm`; sin CI workflows.

### F-08 — P1 — Cancel web no cancela generación

Abort del `fetch` no mata el child; scheduler ocupado.

### F-09 — P1 — Health sin liveness real

`/health` no comprueba `process.poll()`.

### F-10 — P1 — Exposición API

Bind no loopback sin fail-closed fuerte; key en localStorage.

### F-11 — P1 — KV SWA a max_t

~1.5 GB recuperables a CTX 4096 con ring (posición lógica para RoPE).

### F-12 — P1 — Knobs contradictorios

`PILOT=0` como “presente”, `NUC` vs `NUCLEUS`, `TOPP` expert vs token top_p.

### F-13 — P1 — Quickstart no copiable

`cd c` + `python3 c/chat_peng.py`; aliases `/more` vs `/mas`.

### F-14 — P1 — Planner GLM para MiMo

Campos MLA; subestima KV híbrido.

### F-15 — P1 — CUDA mimo (audit) → **fixed post-audit**

En HEAD actual, `c/Makefile` hace que `mimo` dependa y enlace `$(CUDA_OBJ)`. No retirar el claim CUDA; documentar `make mimo CUDA=1`.

### F-16 … F-21

Release/manifest/licencias (F-16), deps fijadas (F-17), docs de estado (F-18), calidad/benches (F-19), web SSE/a11y (F-20), I/O/mantenimiento (F-21). Detalle completo en el informe original de revisión; esta copia prioriza estado y fixes en curso.

## Fortalezas a conservar

Motor C pequeño; método oracle-first; docs honestas sobre NVMe/RAM; contenedor HF publicado; validaciones de config MiMo; cola HTTP acotada; `pread_full` >2 GB; tests de primitivas; package-lock + TS estricto.

## Verificación del auditor (1482e26)

- `make -C c mimo` OK; tests C 3/3.
- `make check` falla en clon limpio por `transformers` no declarado.
- ASan: overflow plantilla 70k chars.
- Web: npm test/build OK en revisión paralela.
- GitHub: 0 workflows, 0 tags (en fecha de auditoría).

## Límites de la revisión

Sin contenedor 152 GB en el host del auditor; sin oracle full-feature fijado; sin CUDA en host de auditoría (sí en dev local posterior); sin pentest/fuzz exhaustivo; cifras SWA ring calculadas, no RSS medido en 311B.
