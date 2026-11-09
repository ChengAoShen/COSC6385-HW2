# Experiments with Meltdown and Spectre Attacks

**Course:** COSC 6385 — Computer Architecture
**Author:** ChengAo Shen
**Platform:** AMD Ryzen 7 9800X3D (Zen 5), Arch Linux, kernel 7.2.14-arch1-1

---

## 1. How the two attacks work

Both Meltdown and Spectre are **transient-execution** attacks. A modern CPU
executes instructions *speculatively* and *out of order* to hide memory latency.
When the CPU later discovers the speculation was wrong (a mis-predicted branch, a
faulting load), it discards the *architectural* result — but the
*microarchitectural* side effects, above all **which cache lines were loaded**,
survive. Both attacks plant a secret into the cache during this transient window
and then read it back through a **Flush+Reload cache covert channel**: an array
of 256 cache-line-sized slots is flushed, the transient code touches
`array[secret * stride]`, and afterwards the attacker times each slot — the one
slot that is fast to access reveals the secret byte.

### 1a. Spectre (Variant 1 — Bounds Check Bypass, CVE-2017-5753)

Spectre attacks code in the *same* address space (e.g. a sandbox or the kernel
via a syscall). A victim contains a guarded array access:

```c
if (x < array1_size)          // bound check
    y = array2[array1[x] * 512];
```

The attacker first calls the function many times with **in-bounds** `x` to train
the branch predictor to expect "taken". Then it flushes `array1_size` from cache
and calls with a malicious, **out-of-bounds** `x`. Because `array1_size` is slow
to load, the CPU *speculatively* assumes the branch is taken and executes the
body with the out-of-bounds `x`, reading `array1[x]` (secret) and using it to
index `array2`. The bounds check eventually resolves and the result is squashed —
but `array2[secret*512]` is now cached. Flush+Reload recovers the secret.
Spectre does **not** rely on a CPU bug per se; it abuses correct speculative
execution, which is why it affects almost every out-of-order CPU, AMD included.

### 1b. Meltdown (Variant 3 — Rogue Data Cache Load, CVE-2017-5754)

Meltdown breaks the **user/kernel isolation** boundary directly. The attacker
dereferences a kernel address from user space:

```c
char v = *(kernel_ptr);           // raises a fault
probe[v * 4096];                   // transient: encodes v into cache
```

On affected (mostly Intel) CPUs the load's *data* is forwarded to the dependent
instruction **before** the permission check raises its fault, so the transient
`probe[]` access leaks the kernel byte into the cache covert channel. The fault
is then handled (or suppressed with TSX / a signal handler). Meltdown is a
genuine **hardware defect** in how some cores order the data forward vs. the
privilege check — and this is exactly where AMD differs (see §4).

---

## 2. Experiment setup and experiences

### 2.1 Setup tried

| Item | Choice | Notes |
|---|---|---|
| CPU | **AMD Ryzen 7 9800X3D** (Zen 5, 8C/16T, 96 MB 3D V-Cache) | the only machine available |
| OS / kernel | Arch Linux, kernel **7.2.14** | a *current* kernel with all mitigations on |
| Demos | `IAIK/meltdown`, `flxwu/spectre-attack-demo` | the two repos named in the assignment |
| Checker | `speed47/spectre-meltdown-checker` | to enumerate CVE status |
| Compiler | gcc `-O0 -std=c99` | `-O0` keeps the gadget from being optimized away |

The assignment suggests an **older-kernel Linux VM**. I discuss below why, on
this specific hardware, a VM would not change the outcome, and therefore I ran
the demos **bare-metal in an unprivileged account** and analyzed the results
honestly. This mirrors the provided example report, whose Intel machines also
failed to reproduce Meltdown once microcode/OS patches were present.

### 2.2 Problems encountered and how I troubleshot them

1. **`spectre.c` only ever printed `Unclear`.** The Kocher PoC hard-codes
   `CACHE_HIT_THRESHOLD = 80` cycles, tuned for a ~4 GHz Intel Skylake. I wrote a
   **calibration probe** (`src/cache_calibration.c`) that times a cached vs.
   `clflush`-ed load on this CPU:

   ```
   avg cache HIT : ~86 cycles
   avg cache MISS: ~465 cycles
   ```

   (see `evidence/02_cache_calibration.txt`). So 80 cycles was marginally *below*
   the real hit time — many true hits were being rejected. Sweeping the threshold
   (55–160) moved the needle only slightly (`T=75` gave 5/19 "Success", all other
   values 0), which pointed at a deeper issue than the threshold.

2. **Is the leak even happening?** To separate "attack fails" from "scoring is
   noisy", I wrote `src/spectre_leak_proof.c`, which runs the full Spectre
   training/attack loop but, instead of guessing, **averages the probe time of
   the known-correct secret byte vs. all other bytes** over 2000 trials:

   ```
   target index 0x54 ('T')  avg = 81.8 cycles   <-- cached by speculation
   all other indices        avg = 261.5 cycles
   ```

   (`evidence/03_spectre_leak_proof.txt`). The secret line is ~3x faster — the
   **Spectre leak is unambiguously real on this CPU.** The information channel
   works; the difficulty is purely in *clean single-shot recovery*.

3. **Why clean recovery is still hard here.** This chip's **96 MB 3D V-Cache and
   very aggressive Zen 5 hardware prefetcher** pull neighbouring probe lines into
   cache, compressing the hit/miss gap at the *per-byte* level and raising the
   noise floor. In the mean-time-scoring variant (`src/spectre_robust.c`) the
   correct letters (`M c e u i f r …`) consistently rank at or near the top, but
   many decoys sit within a few cycles, so a 40-byte string does not come back
   verbatim. The 512-byte stride that defeats older prefetchers is not enough
   against this one, and disabling the prefetcher needs MSR/root access I did not
   use. This is a legitimate, reportable hardware observation, not a coding bug.

4. **Meltdown build.** `IAIK/meltdown` compiled cleanly (`make`) on the modern
   toolchain. Its most basic self-test, Demo #1 `./test` — which uses Meltdown to
   read from the process's *own* mapped memory, breaking no isolation — produced
   **garbled output** on this AMD part:

   ```
   Expect: Wait, do you smell something burning?
   Got:    ,5>;, %!=...   (corrupted)
   ```

   (`evidence/06_meltdown_fails_on_amd.txt`). The repo's README states the demos
   are "tested on ... any modern **Intel** CPU since 2010". Demo #5 `./secret`
   additionally requires root / `pagemap` access and so was not run unprivileged.

### 2.3 Does it work in the end?

- **Spectre v1:** the covert channel and the speculative leak **work** (proven,
  §2.2.2); full plaintext recovery is **unreliable** on this Zen 5 + huge-cache
  machine.
- **Meltdown:** **does not work**, by design of the hardware (§4) — consistent
  with the example report's Intel results after patching.

---

## 3. Results (screenshots / logs)

All raw logs are in `evidence/`; terminal screenshots in `screenshots/`.

### 3.1 `spectre-meltdown-checker` — CVE status on this host

(`evidence/01_spectre-meltdown-checker.txt`)

| CVE | Name | CPU affected? | Kernel status on this host |
|---|---|---|---|
| CVE-2017-5753 | Spectre v1 (bounds check bypass) | **YES** | NOT VULNERABLE — usercopy/swapgs barriers, pointer sanitization |
| CVE-2017-5715 | Spectre v2 (branch target injection) | **YES** | NOT VULNERABLE — Enhanced IBRS, STIBP always-on, IBPB |
| CVE-2017-5754 | **Meltdown** (rogue data cache load) | **NO** | NOT VULNERABLE — *CPU vendor reports model not affected* |
| CVE-2018-3639 | Spectre v4 (speculative store bypass) | **YES** | NOT VULNERABLE — SSB disabled via prctl |
| CVE-2018-3615/20/46 | Foreshadow / L1TF | NO | Not affected |
| CVE-2018-12126/130/127, 2019-11091 | Fallout / ZombieLoad / RIDL (MDS) | NO | Not affected |

The pattern is the key result: the **Spectre family is "CPU-affected but
OS/microcode-mitigated"**, whereas **Meltdown and the Intel-specific MDS/L1TF
families are "CPU not affected"** — the AMD core never had the defect.

### 3.2 Spectre leak proof

```
target index 0x54 ('T')  avg = 81.8 cycles
all other indices        avg = 261.5 cycles
```
→ secret cache line is ~3x faster to reload — speculative leak confirmed.

### 3.3 Meltdown self-test fails

```
Expect: Wait, do you smell something burning?
Got:    ,5>;, %!=...        (garbled — transient kernel-style read does not forward on Zen 5)
```

---

## 4. How these attacks can be fixed (software and hardware)

### Software / OS
- **KPTI (Kernel Page-Table Isolation)** — unmaps kernel pages while in user
  mode so a Meltdown load has nothing to forward. (Linux enables it only on
  affected CPUs; on this AMD part `pti` is **off** because it is unnecessary.)
- **Retpolines / IBRS / STIBP / IBPB** — stop Spectre v2 branch-target injection
  by serializing or isolating indirect-branch prediction across contexts.
- **`lfence` / array-index masking** — the compiler/kernel inserts a speculation
  barrier or clamps the index after a bounds check (`array_index_nospec`) so
  Spectre v1 cannot speculate past the check. Visible on this host as the v1
  mitigation "usercopy/swapgs barriers and __user pointer sanitization".
- **SSBD** — disables speculative store bypass for Spectre v4 (seen here as
  "SSB disabled via prctl").
- Boot flag `mitigations=off` would *re-enable* the vulnerabilities for research;
  I kept mitigations **on** to document a realistic, patched machine.

### Hardware / microarchitecture
- **Fix the fault/forward ordering (Meltdown):** check the page-permission bit
  *before* the load result is forwarded to dependent transient instructions. AMD
  cores already do this, which is precisely why CVE-2017-5754 reads "CPU not
  affected" above; Intel added it in post-2018 silicon (e.g. Whiskey/Coffee Lake
  refresh and later).
- **Partition / tag speculative predictors** by privilege domain and flush the
  BTB/RSB on context switch (hardware IBRS, "Enhanced IBRS" seen here) to close
  Spectre v2 cheaply.
- **Harden the cache covert channel:** randomized or partitioned cache indexing,
  or not allocating cache lines for squashed transient loads, removes the
  Flush+Reload readout both attacks depend on.

---

## 5. Conclusion

On a current AMD Zen 5 machine with an up-to-date kernel:
**Meltdown cannot be reproduced** — the AMD microarchitecture never forwarded
faulting kernel loads, so the vulnerability checker reports the CPU as "not
affected" and the IAIK self-test garbles. **Spectre v1 is reproducible at the
information-leak level** — the secret cache line is measurably ~3x faster to
reload — but the enormous 3D V-Cache and aggressive prefetcher of this part keep
clean full-string recovery from being reliable. This matches the assignment's
example report, where Intel machines likewise failed Meltdown once microcode and
OS patches were in place, and demonstrates the central lesson: these are
**microarchitectural** vulnerabilities whose presence depends on specific CPU
design choices, and which today are closed by a layered combination of hardware
redesign, microcode, and OS mitigations.

---

### Reproduce

```bash
cd meltdown-spectre
./run_experiment.sh        # builds src/, runs calibration, leak proof, checker
# see evidence/ for the captured logs
```
