# Experiment B — Meltdown & Spectre

COSC 6385 November homework. Reproduce the Meltdown and Spectre transient-execution
attacks, document the setup and problems, and discuss mitigations.

**Full write-up: [REPORT.md](REPORT.md).**

## TL;DR result

Tested on an **AMD Ryzen 7 9800X3D (Zen 5)** with an up-to-date Linux kernel:

- **Meltdown (CVE-2017-5754): cannot be reproduced** — the AMD core never
  forwarded faulting kernel loads, so `spectre-meltdown-checker` reports the CPU
  as *not affected* and the IAIK self-test garbles. This matches the provided
  example report, where patched Intel machines also failed Meltdown.
- **Spectre v1 (CVE-2017-5753): leak confirmed** — the speculatively-loaded
  secret cache line is ~3x faster to reload (≈82 vs ≈261 cycles). Clean full
  plaintext recovery is unreliable here because of the 96 MB 3D V-Cache and the
  aggressive Zen 5 prefetcher.

## Layout

```
meltdown-spectre/
├── REPORT.md              full report (attacks, setup, results, fixes)
├── run_experiment.sh      one-shot reproduction script
├── src/
│   ├── Makefile
│   ├── cache_calibration.c    measure cache hit vs miss latency
│   ├── spectre_leak_proof.c   prove the speculative leak (mean-time)
│   ├── spectre_robust.c       mean-time recovery variant
│   └── spectre_original.c     Kocher/flxwu PoC (reference)
└── evidence/              captured terminal logs (the "screenshots" / results)
```

## Reproduce

```bash
# optional: fetch the upstream demos + checker referenced by the assignment
git clone https://github.com/speed47/spectre-meltdown-checker
git clone https://github.com/IAIK/meltdown && make -C meltdown

./run_experiment.sh
```

## Credits / upstream

- IAIK/meltdown — https://github.com/IAIK/meltdown
- flxwu/spectre-attack-demo — https://github.com/flxwu/spectre-attack-demo
  (based on Paul Kocher's original Spectre PoC)
- speed47/spectre-meltdown-checker — https://github.com/speed47/spectre-meltdown-checker
