# Build the Spectre PoC variants used in this experiment.
# -O0 keeps the speculation gadget from being optimized away.
CC      ?= gcc
CFLAGS  ?= -O0 -std=c99 -Wall

BINS = cache_calibration spectre_leak_proof spectre_robust spectre_original

all: $(BINS)

cache_calibration: cache_calibration.c
	$(CC) $(CFLAGS) -o $@ $<

spectre_leak_proof: spectre_leak_proof.c
	$(CC) $(CFLAGS) -o $@ $<

spectre_robust: spectre_robust.c
	$(CC) $(CFLAGS) -o $@ $<

spectre_original: spectre_original.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
