# Controlling the classical Heisenberg chain

Code for the deterministically controlled classical Heisenberg chain ($J_x=J_y=J_z=1$, ring of $L$ unit spins).
One time step is Hamiltonian evolution for time $\tau = 1$ followed by the control map onto a period-$N$ helix,

$$
\mathbf S^0_j = \bigl(0,\ \cos\tfrac{2\pi j}{N},\ \sin\tfrac{2\pi j}{N}\bigr), \qquad
\mathbf S_j \;\to\; \frac{(1-a)\,\mathbf S^0_j + a\,\mathbf S_j}{\lvert (1-a)\,\mathbf S^0_j + a\,\mathbf S_j\rvert}.
$$

The order parameter is the activity $S_{\rm diff} = \frac1L\sum_j \lvert\mathbf S_j - \mathbf S^0_j\rvert$
(0 when the chain is captured by the helix). The intermediate cone phase is detected by the long-range tilt correlator

$$
G_x = \frac{1}{L(b+1)}\sum_j\sum_{r=b}^{2b} S^x_j\,S^x_{j+r},
$$

with $b = 64$ much larger than the correlation length $\xi$ of the chaotic phase.
$G_x$ is exactly 0 on the helix and of order $e^{-b/\xi}\approx 0$ in the chaotic phase. Inside cone domains it equals $s_+^2$, the squared tilt of the cone. Linear stability of the helix gives the threshold
$a_c^{\rm lin}(N) = e^{-\tau\max_k\lambda(k,N)}$ with $\lambda(k,N)^2 = -4\cos q\,(\cos k-\cos q)(\cos k-1)$, $q = 2\pi/N$.

## Layout

```
Project.toml, src/     Julia package HeisenbergControl: model, exact sublattice-splitting integrator,
                       control map, observables, linear stability (a_c^lin), space-time trajectories
test/runtests.jl       validation suite              julia --project -e 'using Pkg; Pkg.test()'
csim/sdiff_scan.c      fast C kernel for the S_diff scans (same integrator, control and initial-state
                       convention as the Julia package); `make -C csim`
scripts/
  phase_scan.sh        (N, a) scan of the steady-state S_diff            -> results/sdiff_phase_L2520/
  phase_diagram.jl     heatmap of S_diff in (N, a) with a_c^lin(N)       -> figures/phase_Sdiff_N3-20.pdf
  order_params_scan.jl (N, a) scan of S_diff, G_x and λ (Julia)          -> results/order_params_L1200/
  order_params_diagram.jl  heatmaps of G_x, λ, S_diff                    -> figures/order_param_{Gx,lambda,Sdiff}_N3-20.pdf
paper/                 the phase diagrams as they appear in the paper (output of the two diagram scripts):
                       order_param_{Sdiff,Gx,lambda}_N3-20_clean.pdf (L ≈ 1200), order_params_N3-20_clean.pdf
                       (3 panels), phase_Sdiff_N3-20.pdf (L ≈ 2520)
  spacetime_gifs.jl    space-time GIFs of |S_j(t) - S0_j| across a       -> figures/spacetime_N*_L*.gif
  Project.toml         plotting environment (Plots, LaTeXStrings)
```

## Requirements

* Julia ≥ 1.11
* a C compiler (gcc or clang). OpenMP is used if available; otherwise a serial build is made and
  the scan runs one process per core.

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'           # the package (simulations)
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'   # plotting environment
```

## Phase diagram

```bash
scripts/phase_scan.sh                                   # N = 3..20, a = 0:0.01:1, L ≈ 2520, 32 samples, T = 20000
julia --project=scripts scripts/phase_diagram.jl        # -> figures/phase_Sdiff_N3-20.pdf
```

* $L$ is the largest multiple of $\mathrm{lcm}(2,N)$ that is $\le 2520$ (the helix must close on the ring).
* Each point averages $S_{\rm diff}$ over the last 25 % of the $T$ steps. The colour is the mean over 32 random initial
  states, which are the same for every $a$.
* A sample stops early once it is captured by a linearly stable helix ($\max_j|\mathbf S_j-\mathbf S^0_j| < 10^{-10}$).
* Cost: about $2\times10^{-8}$ s per site-step per core, so the full scan is roughly 15 core-hours
  (about 1–2 h on 10 cores). The scan is resumable, and the parameters can be overridden with environment
  variables (see the header of `phase_scan.sh`), e.g. a quick test:
  `NMIN=4 NMAX=8 NA=21 L0=240 T=2000 NS=8 scripts/phase_scan.sh results/test`.

## Cone phase: G_x and Lyapunov exponent

```bash
julia -t auto --project scripts/order_params_scan.jl          # N = 3..20, a = 0:0.01:1, L ≈ 1200, 16 samples, T = 6000, b = 64
julia --project=scripts scripts/order_params_diagram.jl     # -> figures/order_param_{Gx,lambda,Sdiff}_N3-20.pdf
```

* Each sample is averaged over the last 25 % of the steps. $\lambda$ is the leading Lyapunov exponent from the exact
  tangent dynamics. The colour is the mean over 16 random initial states.
* The figures show two curves: $a_c^{\rm lin}(N)$ (orange) and the cone saddle-node $a_{\rm sn}(N) = 1/(1+2G_{\max})$
  (dashed), where $G_{\max} = \max_{0\le s\le 1}\sqrt{1-s^2}\,\sin(\kappa s/2)$ and $\kappa = 2\tau(1-\cos 2\pi/N)$.
* Optional arguments: `OUTDIR L0 T nsamples Nmin Nmax b`. The scan takes about 4 core-hours (about 25 min on 10 threads)
  and can be resumed, since each finished N is skipped.

## Space-time GIFs

```bash
julia -t auto --project=scripts scripts/spacetime_gifs.jl                  # the three GIFs below
julia -t auto --project=scripts scripts/spacetime_gifs.jl figures 6 1200 0.6 0.8 21   # custom: N L amin amax nframes [T]
```

| GIF | N | L | a (21 frames) |
|---|---|---|---|
| `spacetime_N3_L1500.gif` | 3 | 1500 | 0 … 1 |
| `spacetime_N4_L1600.gif` | 4 | 1600 | 0.5 … 1 |
| `spacetime_N5_L1500.gif` | 5 | 1500 | 0.5 … 1 |

Each frame is one trajectory of $T = L$ steps from the same random initial state, shown on a fixed colour
scale $|\mathbf S_j(t)-\mathbf S^0_j| \in [0, 2]$.
