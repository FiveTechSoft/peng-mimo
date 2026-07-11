#!/usr/bin/env python3
"""Chat interactivo con peng (MiMo-V2.5 311B). El modelo carga UNA vez;
la conversacion persiste en KV entre turnos. Ctrl-D o /salir para terminar."""
import os, subprocess, sys

SNAP = os.path.expanduser(os.environ.get("SNAP", "~/mimo25_i4"))
HERE = os.path.dirname(os.path.abspath(__file__))
ENV = dict(os.environ, SNAP=SNAP, SERVE="1", THINK=os.environ.get("THINK", "0"),
           NGEN=os.environ.get("NGEN", "80"))
READY = b"\x01\x01READY\x01\x01"
END = b"\x01\x01END\x01\x01"

p = subprocess.Popen([os.path.join(HERE, "mimo"), "64", "4", "8"], cwd=HERE, env=ENV,
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, bufsize=0)
print("cargando modelo (~20 s)...", flush=True)
buf = b""
while READY not in buf:
    c = p.stdout.read(1)
    if not c:
        sys.exit("el motor murio durante la carga")
    buf += c
p.stdout.readline()  # resto de linea READY
p.stdout.readline()  # STAT inicial
print("listo. peng 鹏 · MiMo-V2.5 311B · ~0.3 tok/s en frio — paciencia.")
print("comandos: /salir, /reset (borra conversacion), /mas (continua respuesta)\n")

while True:
    try:
        turn = input("tú> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not turn:
        continue
    if turn == "/salir":
        break
    if turn == "/reset":
        p.stdin.write(b"\x02RESET\n"); p.stdin.flush()
    elif turn == "/mas":
        p.stdin.write(b"\x02MORE\n"); p.stdin.flush()
    else:
        p.stdin.write(turn.encode() + b"\n"); p.stdin.flush()
    print("peng> ", end="", flush=True)
    pend = b""
    while True:
        c = p.stdout.read(1)
        if not c:
            sys.exit("\nel motor termino inesperadamente")
        pend += c
        if pend.endswith(END):
            break
        # retiene una cola que aun podria ser prefijo del marcador END
        hold = 0
        for k in range(min(len(END) - 1, len(pend)), 0, -1):
            if pend.endswith(END[:k]):
                hold = k; break
        if len(pend) > hold:
            sys.stdout.buffer.write(pend[:len(pend) - hold]); sys.stdout.flush()
            pend = pend[len(pend) - hold:]
    stat = p.stdout.readline().decode().split()
    if len(stat) >= 4 and stat[0] == "STAT":
        print(f"\n  [{stat[1]} tokens · {float(stat[2]):.2f} tok/s · hit {stat[3]}% · RSS {stat[4]} GB]\n")

p.stdin.close(); p.terminate()
print("chat cerrado.")
