# Build the S_diff scanner.
#   make          OpenMP build if the compiler supports -fopenmp (Linux gcc), else a serial build
#                 using omp_stub/ (e.g. Apple clang); the scan driver runs points in parallel anyway.
CC     ?= cc
CFLAGS ?= -O3 -march=native
ifeq ($(shell uname -m),arm64)
  CFLAGS = -O3 -mcpu=native
endif

OMP := $(shell echo 'int main(void){return 0;}' | $(CC) -fopenmp -x c - -o /dev/null 2>/dev/null && echo yes)

sdiff_scan: sdiff_scan.c
ifeq ($(OMP),yes)
	$(CC) $(CFLAGS) -fopenmp -o $@ $< -lm
else
	$(CC) $(CFLAGS) -Iomp_stub -Wno-unknown-pragmas -o $@ $< -lm
endif

clean:
	rm -f sdiff_scan
