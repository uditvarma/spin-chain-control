#!/usr/bin/env julia
#
# (N, a) scan of the order parameters S_diff, G_x (long-range tilt correlator) and λ (Lyapunov exponent).
#
#   julia -t auto --project=. scripts/order_params_scan.jl [OUTDIR] [L0] [T] [nsamples] [Nmin] [Nmax] [b]
#       defaults: results/order_params_L1200  1200  6000  16  3  20  64     (the paper parameters)
#
# a = 0, 0.01, ..., 1. L = largest multiple of lcm(2, N) <= L0. Same initial states, integrator
# (τ = 1, dt = 0.1, 2nd-order splitting), control and tangent dynamics as HeisenbergControl.run_point.
# Per sample, averaged over the last 25 % of the T steps:
#   Sdiff  = (1/L) Σ_j |S_j - S0_j|                         activity (0 on the helix)
#   Gxr    = G_x = 1/(L(b+1)) Σ_j Σ_{r=b}^{2b} S^x_j S^x_{j+r}
#            long-range tilt correlator: 0 on the helix, s_+^2 inside cone domains, ~ exp(-b/ξ) in chaos
#   lam    leading Lyapunov exponent (exact tangent dynamics)
#   also   Cx = (1/L) Σ_j S^x_j S^x_{j+1} (nearest-neighbour tilt correlation), absMx = |(1/L) Σ_j S^x_j|,
#          mxb = block tilt, mean over floor(L/b) disjoint blocks of |(1/b) Σ_{j∈block} S^x_j|
# Absorbed samples (max_j |d_j| < 1e-10 with a linearly stable helix) stop early: the remaining steps
# contribute Sdiff = Cx = absMx = mxb = Gxr = 0 and lam = λ of the helix.
# One CSV per N in OUTDIR; resumable (finished N are skipped). Cost at the defaults: about 4 core-hours in
# total (about 25 min on 10 threads).
using HeisenbergControl, Printf, Random
const HC = HeisenbergControl

function tilt_obs(X)
    M = size(X, 1); cx = 0.0; mx = 0.0
    @inbounds for m in 1:M
        xo = X[m, 1]; xe = X[m, 4]; xn = X[m == M ? 1 : m + 1, 1]
        cx += xo * xe + xe * xn
        mx += xo + xe
    end
    return cx / (2M), mx / (2M)
end

# block tilt m_b and long-range correlator G_x; sx, cs are length-L / (L+1) work buffers
function tilt_longrange!(sx, cs, X, b)
    M = size(X, 1); L = 2M
    2b < L || throw(ArgumentError("need 2b < L"))
    @inbounds for m in 1:M
        sx[2m - 1] = X[m, 1]; sx[2m] = X[m, 4]
    end
    nb = L ÷ b; mb = 0.0
    @inbounds for k in 0:nb-1
        s = 0.0
        for j in k*b+1:(k+1)*b
            s += sx[j]
        end
        mb += abs(s) / b
    end
    cs[1] = 0.0                                  # cyclic prefix sums: cs[i+1] = Σ_{j≤i} sx[j]
    @inbounds for j in 1:L
        cs[j + 1] = cs[j] + sx[j]
    end
    tot = cs[L + 1]
    P(i) = div(i, L) * tot + cs[rem(i, L) + 1]      # Σ_{j=1}^{i} sx[mod1(j, L)], i ≥ 0
    g = 0.0
    @inbounds for j in 1:L
        g += sx[j] * (P(j + 2b) - P(j + b - 1))
    end
    return mb / nb, g / (L * (b + 1))
end

function run_one(p::RunParams, s::Int, X0, spl, λfp, canabsorb, b)
    rng = HC._sample_rng(p, s)
    Sf = random_state!(Matrix{Float64}(undef, 3, p.L), rng)
    v = randn(rng, 3)
    X = to_soa(Sf); D = to_soa(HC._init_tangent(Sf, v))
    fB = zeros(length(p.epsB))
    W = max(1, round(Int, p.ss_frac * p.T)); t_ss = p.T - W + 1
    sd = 0.0; cx = 0.0; amx = 0.0; mxb = 0.0; gxr = 0.0; lam = 0.0; t_abs = 0
    sx = zeros(p.L); cs = zeros(p.L + 1)
    J = Couplings()
    for t in 1:p.T
        hamiltonian_step!(X, D, spl, J)
        control!(X, D, X0, p.a)
        lt = HC._renorm_tangent!(D)
        m = measure(fB, X, X0, p.epsB, p.rho_eps)
        if t >= t_ss
            c, mxv = tilt_obs(X)
            mb, g = tilt_longrange!(sx, cs, X, b)
            sd += m.Sdiff; cx += c; amx += abs(mxv); mxb += mb; gxr += g; lam += lt
        end
        if canabsorb && m.maxd < p.absorb_tol
            t_abs = t
            lam += λfp * (p.T - max(t, t_ss - 1))   # remaining steps sit on the helix
            break
        end
    end
    return (sd / W, cx / W, amx / W, mxb / W, gxr / W, lam / W, t_abs)
end

function main(outdir, L0, T, ns, Nmin, Nmax, b)
    mkpath(outdir)
    avals = round.(collect(0.0:0.01:1.0); digits = 2)
    for N in Nmin:Nmax
        f = joinpath(outdir, @sprintf("N%02d.csv", N))
        isfile(f) && (println("skip N=$N (exists)"); continue)
        q = lcm(2, N); L = (L0 ÷ q) * q
        X0 = to_soa(helix(L, N)); spl = Splitting(1.0, 0.1; order = 2)
        tasks = [(a, s) for a in reverse(avals) for s in 1:ns]
        res = Vector{NTuple{7,Float64}}(undef, length(tasks))
        t0 = time()
        Threads.@threads :dynamic for k in eachindex(tasks)
            a, s = tasks[k]
            p = RunParams(; N, L, a, T, nsamples = ns, lyapunov = true)
            λfp, can = HC._absorption_rule(p)
            r = run_one(p, s, X0, spl, λfp, can, b)
            res[k] = (r[1:6]..., Float64(r[7]))
        end
        open(f * ".tmp", "w") do io
            println(io, "N,L,a,T,sample,b,Sdiff,Cx,absMx,mxb,Gxr,lam,t_abs")
            for (k, (a, s)) in enumerate(tasks)
                r = res[k]
                @printf(io, "%d,%d,%.2f,%d,%d,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g,%d\n", N, L, a, T, s, b,
                        r[1], r[2], r[3], r[4], r[5], r[6], Int(r[7]))
            end
        end
        mv(f * ".tmp", f; force = true)
        @printf("N=%2d L=%d done in %.0f s\n", N, L, time() - t0); flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    outdir = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "results", "order_params_L1200")
    L0 = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1200
    T  = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 6000
    ns = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 16
    Nmin = length(ARGS) > 4 ? parse(Int, ARGS[5]) : 3
    Nmax = length(ARGS) > 5 ? parse(Int, ARGS[6]) : 20
    b  = length(ARGS) > 6 ? parse(Int, ARGS[7]) : 64
    main(outdir, L0, T, ns, Nmin, Nmax, b)
end
