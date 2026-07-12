#!/usr/bin/env python3
"""Chat interactivo con peng (MiMo-V2.5 311B). El modelo carga UNA vez;
la conversacion persiste en KV entre turnos. Ctrl-D o /salir para terminar.

Bench de tok/s (meta 1.0 por defecto):
  python3 chat_peng.py --bench
  python3 chat_peng.py --bench --runs 3 --warmup 1 --ngen 24 --target 1.0
  # en chat interactivo:  /bench   o  /bench 3

Perfiles (env o flags):
  --fast     maxima velocidad (CUDA, TOPP=0.55, TEMP=0.7, PILOT, REPIN)
  --quality  mas anclado (TEMP=0.4, TOPP=0.85) — mas lento
  --profile NAME  heat map aislado: SNAP/.coli_usage.NAME (COLI_PROFILE)
  default    equilibrio calidad+velocidad
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time

READY = b"\x01\x01READY\x01\x01"
END = b"\x01\x01END\x01\x01"

DEFAULT_BENCH_PROMPT = "Write one short sentence about Rome."
DEFAULT_TARGET = 1.0


def find_mimo() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [
        os.environ.get("MIMO"),
        os.path.join(here, "mimo"),
        "/mnt/c/colibri/c/mimo",
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    sys.exit("no encuentro el binario 'mimo' (prueba MIMO=/ruta/al/binario)")


def build_env(args) -> dict:
    """Defaults: velocidad (CUDA/DIRECT/PILOT/REPIN/I4S) + calidad razonable int4."""
    snap = os.path.expanduser(os.environ.get("SNAP", "~/mimo25_i4"))
    ngen = str(args.ngen if args.ngen is not None else os.environ.get("NGEN", "80"))

    # Speed profile defaults — user/env always wins if already set
    if args.fast:
        # Fewer experts/token; anticipatory I/O ON; avoid PILOT_DEPTH=2 (CPU tax)
        temp_d, nuc_d, topp_d = "0.7", "0.9", "0.55"
        topk_d, draft_d, pdepth = "0", "0", "1"
    elif args.quality:
        temp_d, nuc_d, topp_d = "0.4", "0.85", "0.85"
        topk_d, draft_d, pdepth = "0", "0", "1"
    else:
        temp_d, nuc_d, topp_d = "0.6", "0.9", "0.60"
        topk_d, draft_d, pdepth = "0", "0", "1"

    env = dict(os.environ)
    env.update({
        "SNAP": snap,
        "SERVE": "1",
        "THINK": os.environ.get("THINK", "0"),
        "NGEN": ngen,
        "TEMP": os.environ.get("TEMP", temp_d),
        "NUCLEUS": os.environ.get("NUCLEUS", nuc_d),
        "TOPP": os.environ.get("TOPP", topp_d),
        "TOPK": os.environ.get("TOPK", topk_d),
        "DRAFT": os.environ.get("DRAFT", draft_d),
        "DIRECT": os.environ.get("DIRECT", "1"),
        "COLI_CUDA": os.environ.get("COLI_CUDA", "1"),
        "CUDA_DENSE": os.environ.get("CUDA_DENSE", "1"),
        "PILOT": os.environ.get("PILOT", "1"),
        "PILOT_DEPTH": os.environ.get("PILOT_DEPTH", pdepth),
        "PREFETCH": os.environ.get("PREFETCH", "1"),
        "REPIN": os.environ.get("REPIN", "32"),
        "I4S": os.environ.get("I4S", "1"),
        "OVERLAP": os.environ.get("OVERLAP", "1"),
        "MEMWATCH": os.environ.get("MEMWATCH", "1"),
        "TRAJ": os.environ.get("TRAJ", "1"),
        "TRAJ_K": os.environ.get("TRAJ_K", "8"),
        "TRAJ_DEPTH": os.environ.get("TRAJ_DEPTH", "2"),
    })
    # Domain heat map (colibri #71): chat vs code vs … do not share pin history
    prof = getattr(args, "profile", None) or os.environ.get("COLI_PROFILE") or os.environ.get("PENG_PROFILE")
    if prof:
        env["COLI_PROFILE"] = prof
    return env


def start_engine(mimo: str, env: dict, *, err_log: str | None = None) -> subprocess.Popen:
    cwd = os.path.dirname(mimo)
    err = open(err_log, "wb") if err_log else subprocess.DEVNULL
    p = subprocess.Popen(
        [mimo, "64", "4", "8"],
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err,
        bufsize=0,
    )
    p._err_file = err if err_log else None  # type: ignore[attr-defined]
    print("cargando modelo (~20–50 s)...", flush=True)
    t0 = time.perf_counter()
    buf = b""
    while READY not in buf:
        c = p.stdout.read(1)
        if not c:
            sys.exit("el motor murio durante la carga (revisa SNAP, binario CUDA, RAM)")
        buf += c
    # Engine: READY\n + STAT\n  — after marker, skip blanks until STAT
    if read_stat(p) is None:
        print("(aviso: sin STAT inicial)", flush=True)
    print(f"listo en {time.perf_counter() - t0:.1f}s. peng 鹏 · MiMo-V2.5 311B int4", flush=True)
    return p


def parse_stat(line: str) -> dict | None:
    """STAT n_tokens tok/s hit% RSS_GB [prompt_toks truncated?]"""
    parts = line.split()
    if len(parts) < 4 or parts[0] != "STAT":
        return None
    out = {
        "tokens": int(float(parts[1])),
        "tok_s": float(parts[2]),
        "hit": float(parts[3]),
        "rss": float(parts[4]) if len(parts) > 4 else 0.0,
    }
    if len(parts) > 5:
        out["prompt_toks"] = int(float(parts[5]))
    if len(parts) > 6:
        out["truncated"] = int(float(parts[6]))
    return out


def read_stat(p: subprocess.Popen) -> dict | None:
    """After END or READY the engine prints \\n then STAT. Skip blank lines."""
    for _ in range(8):
        line = p.stdout.readline()
        if not line:
            return None
        s = line.decode(errors="replace").strip()
        if not s:
            continue
        st = parse_stat(s)
        if st:
            return st
        # unexpected non-STAT line — keep scanning briefly
    return None


def drain_until_end(p: subprocess.Popen, *, stream: bool = False) -> bytes:
    """Read stdout until END marker. If stream, print non-marker bytes."""
    pend = b""
    out = b""
    while True:
        c = p.stdout.read(1)
        if not c:
            sys.exit("\nel motor termino inesperadamente")
        pend += c
        if pend.endswith(END):
            break
        hold = 0
        for k in range(min(len(END) - 1, len(pend)), 0, -1):
            if pend.endswith(END[:k]):
                hold = k
                break
        if len(pend) > hold:
            chunk = pend[: len(pend) - hold]
            out += chunk
            if stream:
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
            pend = pend[len(pend) - hold :]
    return out


def one_turn(p: subprocess.Popen, text: str, *, quiet: bool = False) -> dict | None:
    """Envia un turno de usuario y consume hasta END+STAT. Devuelve stats o None."""
    if not quiet:
        print("peng> ", end="", flush=True)
    p.stdin.write(text.encode() + b"\n")
    p.stdin.flush()
    drain_until_end(p, stream=not quiet)
    st = read_stat(p)
    if st and not quiet:
        print(
            f"\n  [{st['tokens']} tokens · {st['tok_s']:.2f} tok/s · "
            f"hit {st['hit']:.0f}% · RSS {st['rss']:.2f} GB]\n",
            flush=True,
        )
    elif not quiet:
        print(flush=True)
    return st


def reset_kv(p: subprocess.Popen) -> None:
    p.stdin.write(b"\x02RESET\n")
    p.stdin.flush()
    drain_until_end(p, stream=False)
    read_stat(p)  # STAT 0 after reset


def run_bench(
    p: subprocess.Popen,
    *,
    prompt: str,
    runs: int,
    warmup: int,
    target: float,
) -> int:
    """Protocolo: warmup (descarta) + N runs con /reset entre medias. Gate mediana >= target."""
    print("\n=== peng bench (SERVE, KV reset entre runs) ===", flush=True)
    print(
        f"prompt={prompt!r}\n"
        f"warmup={warmup}  runs={runs}  target={target:.2f} tok/s\n"
        f"TEMP={os.environ.get('TEMP', '?')} TOPP={os.environ.get('TOPP', '?')} "
        f"NGEN={os.environ.get('NGEN', '?')} "
        f"COLI_CUDA={os.environ.get('COLI_CUDA', '0')} "
        f"CUDA_DENSE={os.environ.get('CUDA_DENSE', '0')} "
        f"DIRECT={os.environ.get('DIRECT', '?')}",
        flush=True,
    )

    measured: list[dict] = []
    total = warmup + runs
    for i in range(total):
        kind = "warmup" if i < warmup else f"run {i - warmup + 1}/{runs}"
        print(f"\n--- {kind} ---", flush=True)
        # Contexto limpio para medir decode comparable (no multiturno)
        if i > 0:
            reset_kv(p)
        t0 = time.perf_counter()
        st = one_turn(p, prompt, quiet=False)
        wall = time.perf_counter() - t0
        if not st:
            print(f"  (sin STAT; wall {wall:.1f}s)", flush=True)
            continue
        st["wall"] = wall
        label = "WARM" if i < warmup else "MEAS"
        print(
            f"  [{label}] engine={st['tok_s']:.3f} tok/s  wall={wall:.2f}s  "
            f"tokens={st['tokens']}  hit={st['hit']:.1f}%  RSS={st['rss']:.2f} GB",
            flush=True,
        )
        if i >= warmup:
            measured.append(st)

    if not measured:
        print("\nbench: sin mediciones utiles", flush=True)
        return 1

    rates = [m["tok_s"] for m in measured]
    hits = [m["hit"] for m in measured]
    med = statistics.median(rates)
    mean = statistics.mean(rates)
    hit_med = statistics.median(hits)
    print("\n=== resumen ===", flush=True)
    print(f"  tok/s por run:  {', '.join(f'{r:.3f}' for r in rates)}", flush=True)
    print(f"  mediana:        {med:.3f} tok/s", flush=True)
    print(f"  media:          {mean:.3f} tok/s", flush=True)
    print(f"  hit mediana:    {hit_med:.1f}%", flush=True)
    print(f"  target:         {target:.2f} tok/s", flush=True)
    if med >= target:
        print(f"  GATE: PASS  (mediana {med:.3f} >= {target:.2f})", flush=True)
        return 0
    print(
        f"  GATE: FAIL  (mediana {med:.3f} < {target:.2f})  "
        f"falta {target - med:.3f} tok/s",
        flush=True,
    )
    print(
        "  pistas: 2o run en adelante (cache), COLI_CUDA=1 CUDA_DENSE=1, "
        "modelo en ext4 no /mnt/c, mas hit-rate / RAM",
        flush=True,
    )
    return 1


def interactive(p: subprocess.Popen, args) -> int:
    print(
        f"env: TEMP={os.environ.get('TEMP')} TOPP={os.environ.get('TOPP')} "
        f"COLI_CUDA={os.environ.get('COLI_CUDA')} PILOT={os.environ.get('PILOT')} "
        f"REPIN={os.environ.get('REPIN')} I4S={os.environ.get('I4S')}\n"
        "comandos: /salir  /reset  /mas  /bench [runs]  /bench-prompt <texto>\n"
        "perfiles: --fast | --quality | --bench\n",
        flush=True,
    )
    bench_prompt = args.prompt
    while True:
        try:
            turn = input("tú> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not turn:
            continue
        if turn in ("/salir", "/exit", "/quit"):
            break
        if turn == "/reset":
            reset_kv(p)
            print("(KV borrada)\n", flush=True)
            continue
        if turn == "/mas" or turn == "/more":
            p.stdin.write(b"\x02MORE\n")
            p.stdin.flush()
            print("peng> ", end="", flush=True)
            drain_until_end(p, stream=True)
            st = read_stat(p)
            if st:
                print(
                    f"\n  [{st['tokens']} tokens · {st['tok_s']:.2f} tok/s · "
                    f"hit {st['hit']:.0f}% · RSS {st['rss']:.2f} GB]\n",
                    flush=True,
                )
            continue
        if turn.startswith("/bench-prompt"):
            rest = turn[len("/bench-prompt") :].strip()
            if rest:
                bench_prompt = rest
                print(f"(prompt de bench = {bench_prompt!r})\n", flush=True)
            else:
                print(f"(prompt actual = {bench_prompt!r})\n", flush=True)
            continue
        if turn == "/bench" or turn.startswith("/bench "):
            parts = turn.split()
            runs = args.runs
            if len(parts) >= 2:
                try:
                    runs = max(1, int(parts[1]))
                except ValueError:
                    print("uso: /bench [runs]\n", flush=True)
                    continue
            run_bench(
                p,
                prompt=bench_prompt,
                runs=runs,
                warmup=args.warmup,
                target=args.target,
            )
            print()
            continue

        one_turn(p, turn)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="peng chat + bench de tok/s (meta 1.0 por defecto)"
    )
    ap.add_argument(
        "--bench",
        action="store_true",
        help="modo bench no interactivo (warmup + runs, exit 0 si mediana >= target)",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=3,
        help="runs medidos tras warmup (default 3)",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="runs de calentamiento descartados (default 1)",
    )
    ap.add_argument(
        "--prompt",
        default=os.environ.get("BENCH_PROMPT", DEFAULT_BENCH_PROMPT),
        help=f"prompt de bench (default: {DEFAULT_BENCH_PROMPT!r})",
    )
    ap.add_argument(
        "--target",
        type=float,
        default=float(os.environ.get("BENCH_TARGET", DEFAULT_TARGET)),
        help="tok/s mediana objetivo (default 1.0)",
    )
    ap.add_argument(
        "--ngen",
        type=int,
        default=None,
        help="max tokens por respuesta (default env NGEN o 80; en --bench suele 24)",
    )
    speed = ap.add_mutually_exclusive_group()
    speed.add_argument(
        "--fast",
        action="store_true",
        help="perfil velocidad: TOPP=0.55 TEMP=0.7 + CUDA/PILOT/REPIN/I4S",
    )
    speed.add_argument(
        "--quality",
        action="store_true",
        help="perfil calidad: TEMP=0.4 TOPP=0.85 (mas lento, menos alucinacion)",
    )
    ap.add_argument(
        "--profile",
        default=None,
        metavar="NAME",
        help="mapa de experts aislado: SNAP/.coli_usage.NAME (env COLI_PROFILE)",
    )
    args = ap.parse_args()

    if args.bench and args.ngen is None and "NGEN" not in os.environ:
        # bench: menos tokens, comparables con findings §23
        args.ngen = 24

    mimo = find_mimo()
    env = build_env(args)
    # reflejar en os.environ para el resumen del bench
    for k in ("TEMP", "TOPP", "NGEN", "COLI_CUDA", "CUDA_DENSE", "DIRECT", "SNAP"):
        if k in env:
            os.environ[k] = env[k]

    err_log = "/tmp/peng_engine.err" if args.bench else None
    p = start_engine(mimo, env, err_log=err_log)
    try:
        if args.bench:
            rc = run_bench(
                p,
                prompt=args.prompt,
                runs=max(1, args.runs),
                warmup=max(0, args.warmup),
                target=args.target,
            )
            if err_log and os.path.isfile(err_log):
                print(f"\n(engine stderr → {err_log})", flush=True)
            return rc
        return interactive(p, args)
    finally:
        try:
            p.stdin.close()
        except Exception:
            pass
        p.terminate()
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
        errf = getattr(p, "_err_file", None)
        if errf:
            try:
                errf.close()
            except Exception:
                pass
        print("chat cerrado.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
