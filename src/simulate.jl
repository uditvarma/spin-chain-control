# ---------------------------------------------------------------------------
# Single-sample evolution and multi-threaded runs at one (N, L, a) point.
# ---------------------------------------------------------------------------

"""
    RunParams(; N, L, a, T, kwargs...)

All parameters for one (N, L, a) point.

| field          | default     | meaning |
|:---------------|:------------|:--------|
| `N`            | –           | helix period |
| `L`            | –           | chain length (even, multiple of N) |
| `a`            | –           | control parameter (a → 0 strong control) |
| `T`            | –           | number of time steps (Hamiltonian + control) |
| `model`        | `:static`   | `:static` (Jx=Jy=Jz=1) or `:random_signs` (Jx,Jy = ±1 per step) |
| `τ`            | 1.0         | Hamiltonian evolution time per step |
| `dt`           | 0.1         | integrator substep |
| `order`        | 2           | integrator order (2 or 4); (2, 0.1) is converged within statistics (see README) |
| `nsamples`     | 100         | random initial states |
| `lyapunov`     | true        | evolve tangent vector for λ (≈2× cost) |
| `epsB`         | [1e-3,1e-2,5e-2] | brighton thresholds ε (first one also used for n_dom) |
| `rho_eps`      | 0.1         | threshold on |d_j| for active density ρ |
| `ss_frac`      | 0.25        | steady-state averages over the last ss_frac·T steps |
| `per_decade`   | 40          | log-spaced time-series points per decade |
| `absorb_tol`   | 1e-10       | stop a sample once max_j |d_j| < tol (helix linearly stable; see README) |
| `Tabs_max`     | 0           | if > T: keep evolving unabsorbed samples (no recording) up to Tabs_max to measure absorption times |
| `seed`         | 1           | master seed; sample s uses the same initial state for every a |
"""
Base.@kwdef struct RunParams
    N::Int
    L::Int
    a::Float64
    T::Int
    model::Symbol = :static
    τ::Float64 = 1.0
    dt::Float64 = 0.1
    order::Int = 2
    nsamples::Int = 100
    lyapunov::Bool = true
    epsB::Vector{Float64} = [1e-3, 1e-2, 5e-2]
    rho_eps::Float64 = 0.1
    ss_frac::Float64 = 0.25
    per_decade::Int = 40
    absorb_tol::Float64 = 1e-10
    Tabs_max::Int = 0
    seed::Int = 1
end

"""
    time_grid(T; per_decade = 40)

Sorted unique integer times: every step up to 10, then log-spaced, always ending at T.
"""
function time_grid(T::Int; per_decade::Int = 40)
    pts = Set{Int}(1:min(T, 10))
    n = max(2, ceil(Int, per_decade * log10(max(T, 10))) + 1)
    for x in range(0, log10(T); length = n)
        push!(pts, clamp(round(Int, 10.0^x), 1, T))
    end
    push!(pts, T)
    return sort!(collect(pts))
end

# Couplings for one Hamiltonian step (random signs of Jx, Jy drawn per step, uniform in space).
@inline _couplings(model::Symbol, rng, J0::Couplings) = model === :random_signs ?
    Couplings(rand(rng, Bool) ? 1.0 : -1.0, rand(rng, Bool) ? 1.0 : -1.0, 1.0) : J0

# Seed depends on (seed, N, L, s) but NOT on a: common random numbers across a
# (same initial states and, for :random_signs, same sign sequences) reduce the
# variance of differences between neighbouring a values.
_sample_rng(p::RunParams, s::Int) = Xoshiro(hash((p.seed, p.N, p.L, p.model, s)))

# Initial tangent vector: random direction at the middle site (as in the paper),
# projected onto the tangent plane of the sphere, unit norm.
function _init_tangent(Sf::Matrix{Float64}, v::Vector{Float64})
    D = zeros(3, size(Sf, 2))
    m = size(Sf, 2) ÷ 2
    s = Sf[:, m]
    w = v .- dot(v, s) .* s
    D[:, m] .= w ./ norm(w)
    return D
end

# Normalise the tangent vector; return log of its growth over the last step.
function _renorm_tangent!(D::Matrix{Float64})
    n2 = 0.0
    @inbounds @simd for i in eachindex(D)
        n2 += D[i] * D[i]
    end
    inv = 1 / sqrt(n2)
    @inbounds @simd for i in eachindex(D)
        D[i] *= inv
    end
    return 0.5 * log(n2)
end
_renorm_tangent!(::Nothing) = NaN

"""
    run_sample(p, s; S0, spl, tg, λfp, canabsorb) -> NamedTuple

Evolve sample `s`. Returns time series on the grid `tg` and steady-state averages.
`ts_lam[k]` is the Lyapunov exponent averaged over the block (tg[k-1], tg[k]].
"""
function run_sample(p::RunParams, s::Int; X0 = to_soa(helix(p.L, p.N)),
                    spl = Splitting(p.τ, p.dt; order = p.order),
                    tg = time_grid(p.T; per_decade = p.per_decade),
                    λfp_abs = _absorption_rule(p), λfp = λfp_abs[1], canabsorb = λfp_abs[2])
    rng = _sample_rng(p, s)
    Sf = random_state!(Matrix{Float64}(undef, 3, p.L), rng)
    v = randn(rng, 3)            # always drawn, so the RNG stream is identical with/without λ
    D = p.lyapunov ? to_soa(_init_tangent(Sf, v)) : nothing
    return _evolve!(to_soa(Sf), D, p, rng, X0, spl, tg, λfp, canabsorb)
end

function _evolve!(S::Matrix{Float64}, D, p::RunParams, rng, S0, spl, tg, λfp, canabsorb)
    # S, D, S0 are in SoA layout (M×6)
    T = p.T
    ne = length(p.epsB)
    ng = length(tg)
    fB = zeros(ne)
    ts_Sdiff = zeros(ng); ts_ndef = zeros(ng); ts_rho = zeros(ng); ts_ndom = zeros(Int, ng)
    ts_fB = zeros(ng, ne); ts_lam = fill(NaN, ng)
    W = max(1, round(Int, p.ss_frac * T))
    t_ss = T - W + 1
    ss_Sdiff = 0.0; ss_ndef = 0.0; ss_rho = 0.0; ss_ndom = 0.0; ss_lam = 0.0
    ss_fB = zeros(ne)
    gi = 1; blocklam = 0.0; blockstart = 0
    t = 0; t_abs = 0
    J0 = Couplings()
    while t < T
        t += 1
        J = _couplings(p.model, rng, J0)
        hamiltonian_step!(S, D, spl, J)
        control!(S, D, S0, p.a)
        lt = _renorm_tangent!(D)
        m = measure(fB, S, S0, p.epsB, p.rho_eps)
        blocklam += lt
        if t >= t_ss
            ss_Sdiff += m.Sdiff; ss_ndef += m.ndef; ss_rho += m.rho; ss_ndom += m.ndom; ss_lam += lt
            ss_fB .+= fB
        end
        if t == tg[gi]
            ts_Sdiff[gi] = m.Sdiff; ts_ndef[gi] = m.ndef; ts_rho[gi] = m.rho; ts_ndom[gi] = m.ndom
            @views ts_fB[gi, :] .= fB
            ts_lam[gi] = blocklam / (t - blockstart)
            blockstart = t; blocklam = 0.0
            gi += 1
        end
        if canabsorb && m.maxd < p.absorb_tol
            t_abs = t
            break
        end
    end
    if t_abs == 0 && canabsorb && p.Tabs_max > T
        # keep evolving (trajectory only) to measure the absorption time
        while t < p.Tabs_max
            t += 1
            hamiltonian_step!(S, spl, _couplings(p.model, rng, J0))
            control!(S, S0, p.a)
            if measure(fB, S, S0, p.epsB, p.rho_eps).maxd < p.absorb_tol
                t_abs = t
                break
            end
        end
    end
    if t_abs > 0 && t_abs < T
        t = t_abs
        # Absorbed onto the helix: from here on S_diff, ρ, f_B, n_dom = 0 (to < absorb_tol)
        # and the tangent dynamics are those of the linearised map at the fixed point.
        lyap = p.lyapunov
        for k in gi:ng
            if lyap
                ts_lam[k] = k == gi ? (blocklam + λfp * (tg[k] - t)) / (tg[k] - blockstart) : λfp
            end
        end
        nfill = T - max(t, t_ss - 1)                 # steps of the ss window after absorption
        lyap && (ss_lam += λfp * nfill)
    end
    nl = p.lyapunov ? 1.0 : NaN
    return (; ts_Sdiff, ts_ndef, ts_rho, ts_ndom, ts_fB, ts_lam,
              ss_Sdiff = ss_Sdiff / W, ss_ndef = ss_ndef / W, ss_rho = ss_rho / W, ss_ndom = ss_ndom / W,
              ss_fB = ss_fB ./ W, ss_lam = nl * ss_lam / W, t_abs)
end

# Fixed-point exponent and whether absorbed samples may be short-cut:
#   static: analytic λ_fp; random signs: only if the helix is a fixed point for every
#   sign choice (N = 2, 4), with λ_fp measured numerically by `lyapunov_helix`.
function _absorption_rule(p::RunParams)
    if p.model === :static
        λfp = lambda_fixed_point(p.N, p.a; τ = p.τ, L = p.L)
    elseif helix_is_fixed_point(p.N, p.model)
        λfp = lyapunov_helix(p.N, p.a; L = min(p.L, lcm(2, p.N) * cld(256, lcm(2, p.N))), model = p.model,
                             τ = p.τ, dt = p.dt, order = p.order, seed = p.seed)[1]
    else
        return NaN, false
    end
    return λfp, λfp < 0 && p.absorb_tol > 0
end

"""
    run_point(p::RunParams; verbose = false) -> Dict{String,Any}

Run all samples (multi-threaded over samples) and collect per-sample data:
  * `tgrid`                          time grid (Int)
  * `ts_Sdiff, ts_ndef, ts_rho, ts_lam`  ng × nsamples (Float32)
  * `ts_fB`                          ng × nε × nsamples (Float32)
  * `ts_ndom`                        ng × nsamples (Int32)
  * `ss_Sdiff, ss_ndef, ss_rho, ss_ndom, ss_lam`  nsamples (Float64), late-time averages
  * `ss_fB`                          nε × nsamples
  * `t_abs`                          absorption time per sample (0 = not absorbed by max(T, Tabs_max))
plus all parameters and wall time.
"""
function run_point(p::RunParams; verbose::Bool = false)
    check_size(p.L, p.N)
    0 <= p.a <= 1 || throw(ArgumentError("a must be in [0,1]"))
    X0 = to_soa(helix(p.L, p.N))
    spl = Splitting(p.τ, p.dt; order = p.order)
    tg = time_grid(p.T; per_decade = p.per_decade)
    λfp, canabsorb = _absorption_rule(p)
    ng, ne, ns = length(tg), length(p.epsB), p.nsamples

    ts_Sdiff = zeros(Float32, ng, ns); ts_ndef = zeros(Float32, ng, ns); ts_rho = zeros(Float32, ng, ns)
    ts_lam = zeros(Float32, ng, ns); ts_ndom = zeros(Int32, ng, ns)
    ts_fB = zeros(Float32, ng, ne, ns)
    ss_Sdiff = zeros(ns); ss_ndef = zeros(ns); ss_rho = zeros(ns); ss_ndom = zeros(ns); ss_lam = zeros(ns)
    ss_fB = zeros(ne, ns); t_abs = zeros(Int, ns)

    t0 = time()
    done = Threads.Atomic{Int}(0)
    Threads.@threads :dynamic for s in 1:ns
        r = run_sample(p, s; X0, spl, tg, λfp, canabsorb)
        ts_Sdiff[:, s] .= r.ts_Sdiff; ts_ndef[:, s] .= r.ts_ndef; ts_rho[:, s] .= r.ts_rho
        ts_lam[:, s] .= r.ts_lam; ts_ndom[:, s] .= r.ts_ndom
        ts_fB[:, :, s] .= r.ts_fB
        ss_Sdiff[s] = r.ss_Sdiff; ss_ndef[s] = r.ss_ndef; ss_rho[s] = r.ss_rho; ss_ndom[s] = r.ss_ndom
        ss_lam[s] = r.ss_lam; ss_fB[:, s] .= r.ss_fB; t_abs[s] = r.t_abs
        if verbose
            k = Threads.atomic_add!(done, 1) + 1
            k % max(1, ns ÷ 10) == 0 && @printf("    %d/%d samples (%.1f s)\n", k, ns, time() - t0)
        end
    end
    wall = time() - t0

    return Dict{String,Any}(
        "N" => p.N, "L" => p.L, "a" => p.a, "T" => p.T, "model" => String(p.model),
        "tau" => p.τ, "dt" => p.dt, "order" => p.order, "nsamples" => ns,
        "lyapunov" => p.lyapunov, "epsB" => p.epsB, "rho_eps" => p.rho_eps,
        "ss_frac" => p.ss_frac, "absorb_tol" => p.absorb_tol, "Tabs_max" => p.Tabs_max, "seed" => p.seed,
        "ac_linear" => ac_linear(p.N; τ = p.τ, L = p.L), "lambda_fp" => λfp,
        "tgrid" => tg,
        "ts_Sdiff" => ts_Sdiff, "ts_ndef" => ts_ndef, "ts_rho" => ts_rho, "ts_lam" => ts_lam,
        "ts_ndom" => ts_ndom, "ts_fB" => ts_fB,
        "ss_Sdiff" => ss_Sdiff, "ss_ndef" => ss_ndef, "ss_rho" => ss_rho, "ss_ndom" => ss_ndom,
        "ss_lam" => ss_lam, "ss_fB" => ss_fB, "t_abs" => t_abs,
        "walltime" => wall, "nthreads" => Threads.nthreads(),
        "julia" => string(VERSION), "code_version" => "0.1.0",
    )
end
