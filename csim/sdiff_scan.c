/*
 * sdiff_scan.c -- (N, a) scan of S_diff for the deterministically controlled
 * classical Heisenberg chain (static isotropic model, Jx = Jy = Jz = 1).
 *
 * One step = Hamiltonian flow for time tau (2nd-order Strang sublattice
 * splitting with exact precession, |S| = 1 and energy conserved to round-off;
 * same scheme as sandbox/src/integrator.jl) followed by the control map
 *     S_j <- [(1-a) S0_j + a S_j] / |(1-a) S0_j + a S_j|
 * toward the period-N helix S0_j = (0, cos(2 pi (j+1)/N), sin(2 pi (j+1)/N))
 * (0-based j; identical to helix() in sandbox/src/model.jl with 1-based j).
 *
 * Observable: S_diff(t) = (1/L) sum_j |S_j - S0_j|, measured after every step.
 *
 * Absorption shortcut (exact up to tol): if the helix is linearly stable,
 * a * exp(tau * max_k lambda_k) < 1, and max_j |S_j - S0_j| < absorb_tol, the
 * sample has been captured by the helix; S_diff = 0 from then on.
 *
 * Layout: struct-of-arrays per sublattice with one ghost cell, so each sweep
 * is a branch-free unit-stride loop that the compiler vectorises.
 *   even sites j = 2m   -> A[m],   A has ghost A[M] = A[0]
 *   odd  sites j = 2m+1 -> B[m] = Bbuf[m+1], ghost Bbuf[0] = B[M-1]
 * A[m] neighbours: B[m-1], B[m]  = Bbuf[m],   Bbuf[m+1]
 * B[m] neighbours: A[m],   A[m+1] = Abuf[m],  Abuf[m+1]
 *
 * Parallelism: OpenMP over (a, sample) tasks, dynamic schedule, largest a first.
 * Samples use the same initial state for every a (common random numbers).
 *
 * Output (in --out dir):
 *   N<NN>_samples.csv  a,sample,L,Sdiff_ss,Sdiff_ss_std,Sdiff_final,t_abs,steps
 *   N<NN>_ts.f32       float32 S_diff(t) on the time grid, shape [na][ns][ng]
 *   N<NN>_meta.json    parameters, a grid, time grid, a_c^lin
 *
 * Build: gcc -O3 -march=native -fopenmp -o sdiff_scan sdiff_scan.c -lm
 */
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

/* ------------------------------------------------------------------ RNG */
static inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
typedef struct { uint64_t s[4]; } rng_t;
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t rng_next(rng_t *r) { /* xoshiro256** */
    uint64_t *s = r->s, res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}
static inline double rng_u01(rng_t *r) { return (rng_next(r) >> 11) * 0x1.0p-53; }
static void rng_seed(rng_t *r, uint64_t a, uint64_t b, uint64_t c) {
    uint64_t x = a * 0x9E3779B97F4A7C15ULL ^ (b + 0x632BE59BD9B4E019ULL) * 0xD1B54A32D192ED03ULL ^
                 (c + 1) * 0xC2B2AE3D27D4EB4FULL;
    for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&x);
}

/* --------------------------------------------------------- linear theory */
/* lambda(k,N)^2 = -4 cos q (cos k - cos q)(cos k - 1), q = 2 pi / N; max over lattice k. */
static double lambda_max(int N, int L) {
    double cq = cos(2 * M_PI / N), best = 0;
    for (int m = 0; m <= L / 2; m++) {
        double ck = cos(2 * M_PI * m / L);
        double l2 = -4 * cq * (ck - cq) * (ck - 1);
        if (l2 > best) best = l2;
    }
    return sqrt(best);
}

/* ----------------------------------------------------------- integrator */
#define NPOLY 7 /* terms in x^2; truncation x2^7/14! < 1e-20 for x2 <= 0.04 (checked at startup) */
static double CC[NPOLY], CS[NPOLY], C2[NPOLY];
static void init_poly(void) {
    double f = 1; /* f = n! as we go */
    double fact[2 * NPOLY + 3];
    fact[0] = 1;
    for (int n = 1; n < 2 * NPOLY + 3; n++) fact[n] = fact[n - 1] * n;
    for (int k = 0; k < NPOLY; k++) {
        double sg = (k & 1) ? -1.0 : 1.0;
        CC[k] = sg / fact[2 * k];
        CS[k] = sg / fact[2 * k + 1];
        C2[k] = sg / fact[2 * k + 2];
    }
    (void)f;
}

/* Precess every target spin for time d about the field of its two neighbours. */
static void sweep(double *restrict tx, double *restrict ty, double *restrict tz,
                  const double *restrict nx, const double *restrict ny, const double *restrict nz,
                  int M, double d) {
    const double dt2 = d * d;
    double cc[NPOLY], cs[NPOLY], c2[NPOLY];
    for (int k = 0; k < NPOLY; k++) { cc[k] = CC[k]; cs[k] = CS[k]; c2[k] = C2[k]; }
#pragma omp simd
    for (int m = 0; m < M; m++) {
        double hx = nx[m] + nx[m + 1], hy = ny[m] + ny[m + 1], hz = nz[m] + nz[m + 1];
        double sx = tx[m], sy = ty[m], sz = tz[m];
        double x2 = (hx * hx + hy * hy + hz * hz) * dt2;
        double c = cc[NPOLY - 1], s1 = cs[NPOLY - 1], s2 = c2[NPOLY - 1];
        for (int k = NPOLY - 2; k >= 0; k--) {
            c = c * x2 + cc[k]; s1 = s1 * x2 + cs[k]; s2 = s2 * x2 + c2[k];
        }
        double a1 = d * s1;
        double hd = (hx * sx + hy * sy + hz * sz) * (dt2 * s2);
        double cx = hy * sz - hz * sy, cy = hz * sx - hx * sz, cz = hx * sy - hy * sx;
        tx[m] = sx * c - cx * a1 + hx * hd;
        ty[m] = sy * c - cy * a1 + hy * hd;
        tz[m] = sz * c - cz * a1 + hz * hd;
    }
}

typedef struct { int par; double d; } op_t; /* par 0 = A (even), 1 = B (odd) */
typedef struct { int nops; op_t ops[4096]; } split_t;

static split_t make_split(double tau, double dt) {
    int n = (int)ceil(tau / dt - 1e-9); if (n < 1) n = 1;
    if (2 * n + 1 > 4096) { fprintf(stderr, "too many substeps\n"); exit(1); }
    double h = tau / n;
    split_t s = {0};
    for (int i = 0; i < n; i++) { /* A(h/2) B(h) A(h/2), merging neighbours */
        op_t seq[3] = {{0, h / 2}, {1, h}, {0, h / 2}};
        for (int q = 0; q < 3; q++) {
            if (s.nops && s.ops[s.nops - 1].par == seq[q].par) s.ops[s.nops - 1].d += seq[q].d;
            else s.ops[s.nops++] = seq[q];
        }
    }
    return s;
}

typedef struct {
    int M;
    double *Ax, *Ay, *Az;       /* length M+1 (ghost at M)   */
    double *Bbx, *Bby, *Bbz;    /* length M+1 (ghost at 0)   */
} chain_t;

static void chain_alloc(chain_t *c, int M) {
    c->M = M;
    double **p[6] = {&c->Ax, &c->Ay, &c->Az, &c->Bbx, &c->Bby, &c->Bbz};
    for (int i = 0; i < 6; i++) {
        if (posix_memalign((void **)p[i], 64, sizeof(double) * (M + 8))) { perror("alloc"); exit(1); }
        memset(*p[i], 0, sizeof(double) * (M + 8));
    }
}
static void chain_free(chain_t *c) {
    free(c->Ax); free(c->Ay); free(c->Az); free(c->Bbx); free(c->Bby); free(c->Bbz);
}

static void hamiltonian_step(chain_t *c, const split_t *s) {
    int M = c->M;
    for (int i = 0; i < s->nops; i++) {
        if (s->ops[i].par == 0) { /* update A from B: ghost Bb[0] = B[M-1] = Bb[M] */
            c->Bbx[0] = c->Bbx[M]; c->Bby[0] = c->Bby[M]; c->Bbz[0] = c->Bbz[M];
            sweep(c->Ax, c->Ay, c->Az, c->Bbx, c->Bby, c->Bbz, M, s->ops[i].d);
        } else {                   /* update B from A: ghost A[M] = A[0] */
            c->Ax[M] = c->Ax[0]; c->Ay[M] = c->Ay[0]; c->Az[M] = c->Az[0];
            sweep(c->Bbx + 1, c->Bby + 1, c->Bbz + 1, c->Ax, c->Ay, c->Az, M, s->ops[i].d);
        }
    }
}

/* Control on one sublattice fused with measurement: returns sum |d|, updates max |d|. */
static double control_measure(double *restrict x, double *restrict y, double *restrict z,
                              const double *restrict x0, const double *restrict y0,
                              const double *restrict z0, int M, double a, double *maxd) {
    const double b = 1 - a;
    double sum = 0, mx = *maxd;
#pragma omp simd reduction(+ : sum) reduction(max : mx)
    for (int m = 0; m < M; m++) {
        double v1 = b * x0[m] + a * x[m], v2 = b * y0[m] + a * y[m], v3 = b * z0[m] + a * z[m];
        double inv = 1 / sqrt(v1 * v1 + v2 * v2 + v3 * v3);
        v1 *= inv; v2 *= inv; v3 *= inv;
        x[m] = v1; y[m] = v2; z[m] = v3;
        double d1 = v1 - x0[m], d2 = v2 - y0[m], d3 = v3 - z0[m];
        double dn = sqrt(d1 * d1 + d2 * d2 + d3 * d3);
        sum += dn;
        mx = dn > mx ? dn : mx;
    }
    *maxd = mx;
    return sum;
}

/* ----------------------------------------------------------------- setup */
typedef struct {
    int N, L, T, ns, na;
    double amin, amax, tau, dt, ss_frac, absorb_tol;
    uint64_t seed;
    const char *out;
    const char *tag;  /* file prefix (default N<NN>) */
    const char *init_file; /* trace mode: read 3 x L initial state, print S_diff(t) */
    double trace_a;
    int sample;           /* >= 0: single-sample mode with random initial state #sample */
    const char *st_out;   /* single-sample mode: write |d_j|(t) as float32 rows */
    int stride;           /* time stride for st_out rows */
} params_t;

static void helix_fill(chain_t *h, int N) {
    for (int m = 0; m < h->M; m++) {
        for (int p = 0; p < 2; p++) {
            long j = 2L * m + p;                      /* 0-based site */
            double th = 2 * M_PI * (double)((j + 1) % N) / N;
            double y = cos(th), z = sin(th);
            if (p == 0) { h->Ax[m] = 0; h->Ay[m] = y; h->Az[m] = z; }
            else { h->Bbx[m + 1] = 0; h->Bby[m + 1] = y; h->Bbz[m + 1] = z; }
        }
    }
}

static void random_fill(chain_t *c, rng_t *r) {
    for (int m = 0; m < c->M; m++)
        for (int p = 0; p < 2; p++) {
            double z = 2 * rng_u01(r) - 1, ph = 2 * M_PI * rng_u01(r), rr = sqrt(1 - z * z);
            double x = rr * cos(ph), y = rr * sin(ph);
            if (p == 0) { c->Ax[m] = x; c->Ay[m] = y; c->Az[m] = z; }
            else { c->Bbx[m + 1] = x; c->Bby[m + 1] = y; c->Bbz[m + 1] = z; }
        }
}

static int make_grid(int T, int per_decade, int *g) {
    int n = 0, last = 0;
    int npts = (int)ceil(per_decade * log10(T > 10 ? T : 10)) + 1;
    for (int t = 1; t <= T && t <= 10; t++) g[n++] = last = t;
    for (int i = 0; i < npts; i++) {
        int t = (int)lround(pow(10.0, log10((double)T) * i / (npts - 1)));
        if (t > last && t <= T) g[n++] = last = t;
    }
    if (last != T) g[n++] = T;
    return n;
}

/* One sample: returns steady-state mean/std, final value, absorption time; fills ts. */
typedef struct { double ss, ss_std, fin; int t_abs, steps; } res_t;

static res_t run_sample(const params_t *P, const chain_t *H, chain_t *c, const split_t *sp,
                        double a, int can_absorb, const int *grid, int ng, float *ts) {
    int M = c->M, L = P->L, T = P->T;
    int W = (int)lround(P->ss_frac * T); if (W < 1) W = 1;
    int t_ss = T - W + 1, gi = 0;
    double ssum = 0, ssq = 0, sd = 0;
    res_t R = {0};
    for (int t = 1; t <= T; t++) {
        hamiltonian_step(c, sp);
        double maxd = 0;
        double s = control_measure(c->Ax, c->Ay, c->Az, H->Ax, H->Ay, H->Az, M, a, &maxd) +
                   control_measure(c->Bbx + 1, c->Bby + 1, c->Bbz + 1, H->Bbx + 1, H->Bby + 1,
                                   H->Bbz + 1, M, a, &maxd);
        sd = s / L;
        if (t >= t_ss) { ssum += sd; ssq += sd * sd; }
        if (gi < ng && t == grid[gi]) ts[gi++] = (float)sd;
        R.steps = t;
        if (can_absorb && maxd < P->absorb_tol) { R.t_abs = t; break; }
    }
    if (R.t_abs) { /* S_diff = 0 (to < tol) from here on */
        sd = 0;
        while (gi < ng) ts[gi++] = 0.0f;
    }
    double mean = ssum / W, var = ssq / W - mean * mean;
    R.ss = mean; R.ss_std = var > 0 ? sqrt(var) : 0; R.fin = sd;
    return R;
}

static void usage(void) {
    fprintf(stderr,
            "sdiff_scan --N n --L l [--T 20000] [--ns 24] [--amin 0] [--amax 1] [--na 101]\n"
            "           [--tau 1] [--dt 0.1] [--ss_frac 0.25] [--absorb 1e-10] [--seed 1] --out dir [--tag prefix]\n"
            "  trace mode: --init file.txt --a a  (file: L lines 'Sx Sy Sz'), prints t S_diff\n"
            "  single sample: --sample s --a a [--st_out file --stride k]  prints Sdiff_ss sigma_t t_abs Sdiff_final\n");
    exit(1);
}

int main(int argc, char **argv) {
    params_t P = {.N = 0, .L = 0, .T = 20000, .ns = 24, .na = 101, .amin = 0, .amax = 1,
                  .tau = 1, .dt = 0.1, .ss_frac = 0.25, .absorb_tol = 1e-10, .seed = 1,
                  .out = NULL, .tag = NULL, .init_file = NULL, .trace_a = -1,
                  .sample = -1, .st_out = NULL, .stride = 1};
    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        if (i + 1 >= argc) usage();
        const char *v = argv[++i];
        if (!strcmp(k, "--N")) P.N = atoi(v);
        else if (!strcmp(k, "--L")) P.L = atoi(v);
        else if (!strcmp(k, "--T")) P.T = atoi(v);
        else if (!strcmp(k, "--ns")) P.ns = atoi(v);
        else if (!strcmp(k, "--na")) P.na = atoi(v);
        else if (!strcmp(k, "--amin")) P.amin = atof(v);
        else if (!strcmp(k, "--amax")) P.amax = atof(v);
        else if (!strcmp(k, "--tau")) P.tau = atof(v);
        else if (!strcmp(k, "--dt")) P.dt = atof(v);
        else if (!strcmp(k, "--ss_frac")) P.ss_frac = atof(v);
        else if (!strcmp(k, "--absorb")) P.absorb_tol = atof(v);
        else if (!strcmp(k, "--seed")) P.seed = strtoull(v, NULL, 10);
        else if (!strcmp(k, "--out")) P.out = v;
        else if (!strcmp(k, "--tag")) P.tag = v;
        else if (!strcmp(k, "--init")) P.init_file = v;
        else if (!strcmp(k, "--a")) P.trace_a = atof(v);
        else if (!strcmp(k, "--sample")) P.sample = atoi(v);
        else if (!strcmp(k, "--st_out")) P.st_out = v;
        else if (!strcmp(k, "--stride")) P.stride = atoi(v);
        else usage();
    }
    if (P.N < 1 || P.L < 4 || P.L % 2 || P.L % P.N) {
        fprintf(stderr, "need N >= 1, L even, >= 4 and a multiple of N\n");
        return 1;
    }
    init_poly();
    split_t sp = make_split(P.tau, P.dt);
    for (int i = 0; i < sp.nops; i++)
        { /* |h| <= 2, so x2 <= 4 d^2; require the first omitted term x2^NPOLY/(2 NPOLY)! < 1e-17 */
            double x2 = 4 * sp.ops[i].d * sp.ops[i].d, term = 1;
            for (int k = 1; k <= 2 * NPOLY; k++) term /= k;
            if (pow(x2, NPOLY) * term > 1e-17) { fprintf(stderr, "dt too large for polynomial rotation\n"); return 1; }
        }
    int M = P.L / 2;
    chain_t H; chain_alloc(&H, M); helix_fill(&H, P.N);
    double lmax = lambda_max(P.N, P.L), ac_lin = exp(-P.tau * lmax);

    /* ---------------- trace mode (validation against the Julia code) */
    if (P.init_file) {
        FILE *f = fopen(P.init_file, "r"); if (!f) { perror("init"); return 1; }
        chain_t c; chain_alloc(&c, M);
        for (int j = 0; j < P.L; j++) {
            double x, y, z;
            if (fscanf(f, "%lf %lf %lf", &x, &y, &z) != 3) { fprintf(stderr, "bad init\n"); return 1; }
            int m = j / 2;
            if (j % 2 == 0) { c.Ax[m] = x; c.Ay[m] = y; c.Az[m] = z; }
            else { c.Bbx[m + 1] = x; c.Bby[m + 1] = y; c.Bbz[m + 1] = z; }
        }
        fclose(f);
        for (int t = 1; t <= P.T; t++) {
            hamiltonian_step(&c, &sp);
            double maxd = 0;
            double s = control_measure(c.Ax, c.Ay, c.Az, H.Ax, H.Ay, H.Az, M, P.trace_a, &maxd) +
                       control_measure(c.Bbx + 1, c.Bby + 1, c.Bbz + 1, H.Bbx + 1, H.Bby + 1,
                                       H.Bbz + 1, M, P.trace_a, &maxd);
            printf("%d %.17g\n", t, s / P.L);
        }
        return 0;
    }

    /* ---------------- single-sample mode: same initial state as sample #s of a scan.
       Prints "Sdiff_ss sigma_t t_abs Sdiff_final" (no early stop; t_abs = first step with
       max_j |d_j| < absorb_tol, 0 if never); optionally writes |d_j| (float32, L per row)
       every `stride` steps to st_out. */
    if (P.sample >= 0) {
        chain_t c; chain_alloc(&c, M);
        rng_t r; rng_seed(&r, P.seed, (uint64_t)P.N, (uint64_t)P.sample);
        random_fill(&c, &r);
        FILE *fo = P.st_out ? fopen(P.st_out, "wb") : NULL;
        float *row = malloc(sizeof(float) * P.L);
        int W = (int)lround(P.ss_frac * P.T); if (W < 1) W = 1;
        double ssum = 0, ssq = 0, sd = 0; int t_abs = 0;
        for (int t = 1; t <= P.T; t++) {
            hamiltonian_step(&c, &sp);
            double maxd = 0;
            double s = control_measure(c.Ax, c.Ay, c.Az, H.Ax, H.Ay, H.Az, M, P.trace_a, &maxd) +
                       control_measure(c.Bbx + 1, c.Bby + 1, c.Bbz + 1, H.Bbx + 1, H.Bby + 1,
                                       H.Bbz + 1, M, P.trace_a, &maxd);
            sd = s / P.L;
            if (t > P.T - W) { ssum += sd; ssq += sd * sd; }
            if (!t_abs && maxd < P.absorb_tol) t_abs = t;
            if (fo && t % P.stride == 0) {
                for (int m = 0; m < M; m++) {
                    double d1 = c.Ax[m] - H.Ax[m], d2 = c.Ay[m] - H.Ay[m], d3 = c.Az[m] - H.Az[m];
                    row[2 * m] = (float)sqrt(d1 * d1 + d2 * d2 + d3 * d3);
                    d1 = c.Bbx[m + 1] - H.Bbx[m + 1]; d2 = c.Bby[m + 1] - H.Bby[m + 1]; d3 = c.Bbz[m + 1] - H.Bbz[m + 1];
                    row[2 * m + 1] = (float)sqrt(d1 * d1 + d2 * d2 + d3 * d3);
                }
                fwrite(row, sizeof(float), P.L, fo);
            }
        }
        if (fo) fclose(fo);
        double mean = ssum / W, var = ssq / W - mean * mean;
        printf("%.10g %.6g %d %.10g\n", mean, var > 0 ? sqrt(var) : 0, t_abs, sd);
        return 0;
    }
    if (!P.out) usage();
    mkdir(P.out, 0755);
    char tagbuf[64];
    if (!P.tag) { snprintf(tagbuf, sizeof tagbuf, "N%02d", P.N); P.tag = tagbuf; }

    int grid[4096], ng = make_grid(P.T, 20, grid);
    double *av = malloc(sizeof(double) * P.na);
    for (int i = 0; i < P.na; i++) av[i] = P.na == 1 ? P.amin : P.amin + (P.amax - P.amin) * i / (P.na - 1);
    size_t ntask = (size_t)P.na * P.ns;
    float *ts = calloc(ntask * ng, sizeof(float));
    res_t *res = calloc(ntask, sizeof(res_t));

    double t0 = omp_get_wtime();
    int done = 0;
#pragma omp parallel
    {
        chain_t c; chain_alloc(&c, M);
#pragma omp for schedule(dynamic, 1)
        for (size_t k = 0; k < ntask; k++) {
            int ia = P.na - 1 - (int)(k / P.ns), s = (int)(k % P.ns); /* large a first */
            double a = av[ia];
            rng_t r; rng_seed(&r, P.seed, (uint64_t)P.N, (uint64_t)s);
            random_fill(&c, &r);
            int can_absorb = a * exp(P.tau * lmax) < 1 && P.absorb_tol > 0;
            size_t idx = (size_t)ia * P.ns + s;
            res[idx] = run_sample(&P, &H, &c, &sp, a, can_absorb, grid, ng, ts + idx * ng);
#pragma omp atomic
            done++;
        }
        chain_free(&c);
    }
    double wall = omp_get_wtime() - t0;
    long long steps = 0;
    for (size_t k = 0; k < ntask; k++) steps += res[k].steps;

    char fn[1024];
    snprintf(fn, sizeof fn, "%s/%s_samples.csv", P.out, P.tag);
    FILE *f = fopen(fn, "w");
    fprintf(f, "a,sample,L,Sdiff_ss,Sdiff_ss_std,Sdiff_final,t_abs,steps\n");
    for (int ia = 0; ia < P.na; ia++)
        for (int s = 0; s < P.ns; s++) {
            res_t *R = &res[(size_t)ia * P.ns + s];
            fprintf(f, "%.6f,%d,%d,%.10g,%.6g,%.10g,%d,%d\n", av[ia], s, P.L, R->ss, R->ss_std, R->fin,
                    R->t_abs, R->steps);
        }
    fclose(f);
    snprintf(fn, sizeof fn, "%s/%s_ts.f32", P.out, P.tag);
    f = fopen(fn, "wb"); fwrite(ts, sizeof(float), ntask * ng, f); fclose(f);
    snprintf(fn, sizeof fn, "%s/%s_meta.json", P.out, P.tag);
    f = fopen(fn, "w");
    fprintf(f, "{\"N\": %d, \"L\": %d, \"T\": %d, \"nsamples\": %d, \"na\": %d, \"tau\": %g, \"dt\": %g,\n"
               " \"integrator\": \"Strang order 2, exact precession\", \"model\": \"static Jx=Jy=Jz=1\",\n"
               " \"ss_frac\": %g, \"absorb_tol\": %g, \"seed\": %llu, \"lambda_max\": %.12g, \"ac_lin\": %.12g,\n"
               " \"walltime_s\": %.2f, \"threads\": %d, \"total_steps\": %lld,\n"
               " \"ns_per_site_step\": %.3f,\n \"ts_shape\": [%d, %d, %d], \"ts_layout\": \"[a][sample][t] float32\",\n \"a\": [",
            P.N, P.L, P.T, P.ns, P.na, P.tau, P.dt, P.ss_frac, P.absorb_tol, (unsigned long long)P.seed,
            lmax, ac_lin, wall, omp_get_max_threads(), steps,
            1e9 * wall * omp_get_max_threads() / ((double)steps * P.L), P.na, P.ns, ng);
    for (int i = 0; i < P.na; i++) fprintf(f, "%s%.6f", i ? ", " : "", av[i]);
    fprintf(f, "],\n \"tgrid\": [");
    for (int i = 0; i < ng; i++) fprintf(f, "%s%d", i ? ", " : "", grid[i]);
    fprintf(f, "]}\n");
    fclose(f);
    fprintf(stderr, "N=%d L=%d: %d tasks, %.1f s, %.2f ns/site/step/core, a_c^lin=%.4f\n", P.N, P.L,
            (int)ntask, wall, 1e9 * wall * omp_get_max_threads() / ((double)steps * P.L), ac_lin);
    return 0;
}
