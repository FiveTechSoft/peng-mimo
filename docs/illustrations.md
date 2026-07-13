# peng — illustrated tour

Fourteen conceptual infographics covering the whole project, generated from the
project docs (findings.md, roadmap.md, README). Numbers shown match the measured
sections they cite unless noted.

## The idea

![Title — streaming a 311B model on consumer hardware](../assets/illustrations/peng1.png)

![The Peng metaphor: 4.7 GB of cold reads per token riding a 2.75 GB/s NVMe stream](../assets/illustrations/peng2.png)

## Architecture

![Residency map: dense int4 in RAM, 12,032 routed experts on NVMe, LRU membrane between](../assets/illustrations/peng3.png)

![Model selection matrix: why MiMo-V2.5 over GLM-5.2, Hunyuan Hy3, Qwen3.5](../assets/illustrations/peng4.png)

![MiMo-V2.5 logical geometry: 48 layers, dense layer 0, top-8 of 256 experts, 15B active of 311B](../assets/illustrations/peng5.png)

![Hybrid attention: 9 full-attention layers + 39 sliding-window layers keep long context cheap in RAM](../assets/illustrations/peng6.png)

![Token generation flow: VRAM pin → RAM learned pin → per-layer LRU → NVMe pread](../assets/illustrations/peng7.png)

## Correctness before speed

![Bit-exact validation pipeline: tiny oracle → C engine matching → lossless packing → full container](../assets/illustrations/peng8.png)

## Performance

![The physics of the cold start](../assets/illustrations/peng9.png)

> Note: two details in this panel don't match findings.md — the measured RAM-starved
> cap reduction was 64→22 slots (not 64→3), and the rig CPU spec shown is illustrative.

![PILOT prefetching and the 0.60 tok/s record (§37 conditions)](../assets/illustrations/peng10.png)

![The quality vs speed frontier: §40 expert-trim measurements](../assets/illustrations/peng11.png)

## Operations

![NVMe endurance reality: reads don't consume TBW; heat is the constraint; never stream over /mnt/c](../assets/illustrations/peng12.png)

## Philosophy

![Corriente peng: the residual stream as a river (TAO=1, φ/Fibonacci knobs, habit channels)](../assets/illustrations/peng13.png)

![Navigation over loading: the desktop frontier paradigm. Gate 1.0 tok/s still open.](../assets/illustrations/peng14.png)
