#!/usr/bin/env bash
# Reproduce the Meltdown/Spectre experiment on the local machine.
# Captures all logs into evidence/. Run from the meltdown-spectre/ directory.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p evidence

echo "[*] Building Spectre PoC variants..."
make -C src

echo "[*] 00  Host CPU & kernel vulnerability state"
{
  echo "\$ uname -a";  uname -a; echo
  echo "# CPU:"; grep -m1 'model name' /proc/cpuinfo
  echo; echo "# /sys/devices/system/cpu/vulnerabilities:"
  for f in /sys/devices/system/cpu/vulnerabilities/*; do
    printf '%-18s %s\n' "$(basename "$f"):" "$(cat "$f")"
  done
} | tee evidence/00_host_info.txt

echo "[*] 02  Cache hit/miss calibration"
./src/cache_calibration | tee evidence/02_cache_calibration.txt

echo "[*] 03  Spectre leak proof (secret byte vs. others, mean cycles)"
./src/spectre_leak_proof | tee evidence/03_spectre_leak_proof.txt

echo "[*] 04  Spectre original Kocher PoC (first 40 bytes)"
timeout 60 ./src/spectre_original 2>&1 | head -45 | tee evidence/04_spectre_poc_original.txt

echo "[*] 05  Spectre robust mean-time recovery"
timeout 150 ./src/spectre_robust 40 1000 | tee evidence/05_spectre_robust.txt

# Optional: the CVE checker and the IAIK meltdown demo (clone them first)
#   git clone https://github.com/speed47/spectre-meltdown-checker
#   git clone https://github.com/IAIK/meltdown && make -C meltdown
echo "[*] 01  spectre-meltdown-checker (if present)"
if [ -x spectre-meltdown-checker/spectre-meltdown-checker.sh ]; then
  bash spectre-meltdown-checker/spectre-meltdown-checker.sh --no-color \
    > evidence/01_spectre-meltdown-checker.txt 2>/dev/null
  echo "    -> evidence/01_spectre-meltdown-checker.txt"
fi

echo "[*] 06  IAIK Meltdown self-test (if present)"
if [ -x meltdown/test ]; then
  { echo "\$ ./test"; ./meltdown/test; } | tee evidence/06_meltdown_fails_on_amd.txt
fi

echo "[*] Done. See evidence/ for all logs."
