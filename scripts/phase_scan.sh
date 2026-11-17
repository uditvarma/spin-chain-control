#!/bin/bash
# (N, a) scan of the steady-state S_diff for the phase diagram (C kernel csim/sdiff_scan).
#
#   scripts/phase_scan.sh [OUTDIR]          (default: results/sdiff_phase_L2520)
#
# Paper parameters (override with environment variables):
#   NMIN=3 NMAX=20   helix periods
#   NA=101           a = 0, 0.01, ..., 1
#   L0=2520          L = largest multiple of lcm(2, N) <= L0
#   NS=32            random initial states per point (same states for every a)
#   T=20000          steps; steady state = mean over the last 25 %
#   JOBS=<ncpu>      points run in parallel (one single-threaded process each)
# One point = one (N, a); output <OUTDIR>/N<NN>/a<a>_{samples.csv,ts.f32,meta.json}.
# Resumable: points whose meta.json exists are skipped.
set -e
cd "$(dirname "$0")/.."
OUT=${1:-results/sdiff_phase_L2520}
NMIN=${NMIN:-3}; NMAX=${NMAX:-20}; NA=${NA:-101}; L0=${L0:-2520}; NS=${NS:-32}; T=${T:-20000}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN)}
make -s -C csim sdiff_scan
export OUT NS T OMP_NUM_THREADS=1

gcd() { local a=$1 b=$2; while (( b )); do local t=$b; b=$((a % b)); a=$t; done; echo $a; }
for N in $(seq $NMIN $NMAX); do
  q=$(( 2 * N / $(gcd 2 $N) )); L=$(( L0 / q * q ))
  for i in $(seq 0 $((NA - 1))); do
    awk -v N=$N -v L=$L -v i=$i -v na=$NA 'BEGIN { printf "%d %d %.3f\n", N, L, (na > 1 ? i / (na - 1) : 0) }'
  done
done | sort -k2,2nr -k3,3nr |    # big L and active (slow) points first
xargs -P "$JOBS" -L 1 bash -c 'N=$0 L=$1 a=$2; d=$OUT/$(printf N%02d $N); tag=$(printf a%.3f $a)
  [[ -f $d/${tag}_meta.json ]] && exit 0; mkdir -p $d
  csim/sdiff_scan --N $N --L $L --T $T --ns $NS --na 1 --amin $a --amax $a --out $d --tag $tag 2>>$d/run.log
  echo "N=$N L=$L a=$a done"'
