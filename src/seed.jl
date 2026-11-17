# ---------------------------------------------------------------------------
# Seed spreading: the helix on an (effectively) infinite chain with ONE random
# spin at the origin, evolved with exactly the same map (Hamiltonian step +
# control on every site). Standard DP "spreading" protocol (Grassberger–de la
# Torre): survival P(t) ~ t^-δ, active sites N(t) ~ t^θ, spread R²(t) ~ t^{2/z}.
#
# Efficiency: only the active region plus a light-cone margin is evolved, on a
# buffer that doubles when the activity approaches its edge — no ring, no
# finite-size effects, no discarded runs, cost ∝ (cluster size) × t.
# Exactness: in the discretised map information moves ≤ 1 site per sublattice
# sweep, so a margin of ⌈K/2⌉ + 2 rows (K sweeps per step) around the active
# region reproduces the infinite-chain evolution to round-off. The single
# approximation (as in ldp.c) is that sites with |S_j - S0_j| ≤ epsw are snapped
# exactly onto the helix, which defines the window and lets runs die. This is
# harmless where the helix is linearly stable (N = 2, 4; N > 4 below a_c^lin);
# where it is linearly unstable the leading edge is a pulled front and the cutoff
# slightly slows it (Brunet–Derrida ~1/ln²epsw) — keep epsw small there.
# ---------------------------------------------------------------------------

"""
    SeedParams(; N, a, T, kwargs...)

| field        | default   | meaning |
|:-------------|:----------|:--------|
| `N`, `a`, `T`|  –        | helix period, control parameter, number of steps |
| `model`      | `:static` | or `:random_signs` |
| `τ, dt, order` | 1, 0.1, 2 | as in RunParams |
| `nruns`      | 1600      | number of seeds |
| `ngroups`    | 16        | runs are split into groups (run r → group mod1(r, ngroups)) for jackknife errors |
| `thr`        | 0.05      | a site is *active* if |S_j - S0_j| > thr (for P, N, R²) |
| `epsB`       | 1e-3      | brighton threshold on |d_{j+1} - d_j|² (for N_B, n_dom) |
| `epsw`       | 1e-12     | snap |d_j| ≤ epsw onto the helix (window / death criterion) |
| `lyapunov`   | false     | tangent vector restricted to the window (approximate, see docs) |
| `per_decade` | 40        | log-spaced recording times per decade |
| `M0`         | 256       | initial buffer rows (2·M0 sites); doubles as needed |
| `Mmax`       | 2^22      | buffer cap (rows); runs that would exceed it stop and are flagged |
| `seed`       | 1         | master seed (same seeds for every a) |
"""
Base.@kwdef struct SeedParams
    N::Int
    a::Float64
    T::Int
    model::Symbol = :static
    τ::Float64 = 1.0
    dt::Float64 = 0.1
    order::Int = 2
    nruns::Int = 1600
    ngroups::Int = 16
    thr::Float64 = 0.05
    epsB::Float64 = 1e-3
    epsw::Float64 = 1e-12
    lyapunov::Bool = false
    per_decade::Int = 40
    M0::Int = 256
    Mmax::Int = 2^22
    seed::Int = 1
end

mutable struct _SeedBuf
    X::Matrix{Float64}
    X0::Matrix{Float64}
    D::Union{Nothing,Matrix{Float64}}
    off::Int            # absolute site index of local site j is j + off (off even)
end

function _grow!(B::_SeedBuf, N::Int)
    M = size(B.X, 1); s = M ÷ 2                    # new rows on each side
    M2 = 2M
    X0 = helix_soa!(Matrix{Float64}(undef, M2, 6), N, B.off - 2s)
    X = copy(X0)
    @views X[s+1:s+M, :] .= B.X
    if B.D !== nothing
        D = zeros(M2, 6); @views D[s+1:s+M, :] .= B.D; B.D = D
    end
    B.X = X; B.X0 = X0; B.off -= 2s
    return s
end

"""
    run_seed_single(p, r; spl, tg) -> NamedTuple

One seed run. Time series on `tg` (zeros after death): number of active sites
`N`, Σ_active (j - j_seed)² `R2`, Σ_j |d_j| `A`, Σ_j |d_j|²/4 `Ndef`, flagged
brighton bonds `NB`, brighton domains `ndom`, and the λ block averages if
requested. Scalars: `tlast` (last time with an active site), `tdead` (exact
absorption time, 0 = survived to T), `truncated`, `maxrows`.
"""
function run_seed_single(p::SeedParams, r::Int; spl = Splitting(p.τ, p.dt; order = p.order),
                         tg = time_grid(p.T; per_decade = p.per_decade))
    rng = Xoshiro(hash((p.seed, p.N, p.model, :seed, r)))
    M = p.M0
    off = -2 * (M ÷ 2) - 2                         # puts the seed (odd site of row mc) at absolute site -1
    X0 = helix_soa!(Matrix{Float64}(undef, M, 6), p.N, off)
    X = copy(X0)
    mc = M ÷ 2 + 1
    jseed = 2mc - 1 + off
    v = random_state!(zeros(3, 1), rng)
    X[mc, 1] = v[1]; X[mc, 2] = v[2]; X[mc, 3] = v[3]
    D = nothing
    if p.lyapunov
        D = zeros(M, 6)
        w = randn(rng, 3); w .-= dot(w, v[:, 1]) .* v[:, 1]; w ./= norm(w)
        D[mc, 1] = w[1]; D[mc, 2] = w[2]; D[mc, 3] = w[3]
    end
    B = _SeedBuf(X, X0, D, off)

    K = length(spl.ops)
    Mr = cld(K, 2) + 2                             # light-cone margin in rows
    ng = length(tg)
    tsN = zeros(Float32, ng); tsR2 = zeros(ng); tsA = zeros(Float32, ng); tsNdef = zeros(Float32, ng)
    tsNB = zeros(Float32, ng); tsdom = zeros(Float32, ng); tslam = fill(NaN32, ng)
    segs = UnitRange{Int}[mc:mc]                    # clusters of rows containing non-helix sites
    newsegs = UnitRange{Int}[]; wins = UnitRange{Int}[]
    gi = 1; tlast = 0; tdead = 0; truncated = false
    blocklam = 0.0; blockstart = 0
    J0 = Couplings()
    pmlo = mc; pmhi = mc
    for t in 1:p.T
        # grow the buffer if the padded activity approaches its edge
        while first(segs[1]) - Mr < 2 || last(segs[end]) + Mr > size(B.X, 1) - 1
            if 2 * size(B.X, 1) > p.Mmax
                truncated = true; break
            end
            s = _grow!(B, p.N)
            segs .= (r -> r .+ s).(segs); pmlo += s; pmhi += s
        end
        truncated && break
        # windows = active clusters ± light-cone margin, merged when they touch. Separate
        # windows cannot influence each other within one step, so this is exact.
        empty!(wins)
        if B.D !== nothing                          # λ mode: one window (tangent restricted to it)
            push!(wins, (first(segs[1]) - Mr):(last(segs[end]) + Mr))
            mlo, mhi = first(wins[1]), last(wins[1])
            @views B.D[pmlo:min(pmhi, mlo - 1), :] .= 0
            @views B.D[max(pmlo, mhi + 1):pmhi, :] .= 0
            pmlo = mlo; pmhi = mhi
        else
            for sg in segs
                w = (first(sg) - Mr):(last(sg) + Mr)
                if !isempty(wins) && first(w) <= last(wins[end]) + 1
                    wins[end] = first(wins[end]):max(last(wins[end]), last(w))
                else
                    push!(wins, w)
                end
            end
        end
        J = _couplings(p.model, rng, J0)
        for w in wins
            hamiltonian_step_range!(B.X, B.D, spl, J, first(w), last(w))
            control!(B.X, B.D, B.X0, p.a, w)
        end
        if B.D !== nothing
            n2 = sum(abs2, @view B.D[pmlo:pmhi, :])
            @views B.D[pmlo:pmhi, :] ./= sqrt(n2)
            blocklam += 0.5 * log(n2)
        end
        # measure + snap; collect the rows that still deviate from the helix
        X = B.X; X0 = B.X0
        nact = 0; r2 = 0.0; A = 0.0; nd = 0.0; nB = 0; ndom = 0
        empty!(newsegs)
        for w in wins
            prevflag = false
            dprev = _dvec(X, X0, first(w) - 1, 3)    # site just left of the window (helix)
            @inbounds for m in w
                rowact = false
                for k in (0, 3)
                    d = _dvec(X, X0, m, k)
                    dn2 = d[1]^2 + d[2]^2 + d[3]^2
                    if dn2 <= p.epsw^2
                        X[m, k+1] = X0[m, k+1]; X[m, k+2] = X0[m, k+2]; X[m, k+3] = X0[m, k+3]
                        d = (0.0, 0.0, 0.0); dn2 = 0.0
                    else
                        rowact = true
                    end
                    dn = sqrt(dn2)
                    A += dn; nd += dn2 / 4
                    if dn > p.thr
                        nact += 1
                        j = 2m - 1 + (k == 3) + B.off
                        r2 += Float64(j - jseed)^2
                    end
                    f = _g(dprev, d) > p.epsB              # bond (j-1, j)
                    nB += f; ndom += f & !prevflag; prevflag = f
                    dprev = d
                end
                if rowact
                    if !isempty(newsegs) && last(newsegs[end]) == m - 1
                        newsegs[end] = first(newsegs[end]):m
                    else
                        push!(newsegs, m:m)
                    end
                end
            end
        end
        nact > 0 && (tlast = t)
        if gi <= ng && t == tg[gi]
            tsN[gi] = nact; tsR2[gi] = r2; tsA[gi] = A; tsNdef[gi] = nd; tsNB[gi] = nB; tsdom[gi] = ndom
            if B.D !== nothing
                tslam[gi] = blocklam / (t - blockstart); blockstart = t; blocklam = 0.0
            end
            gi += 1
        end
        if isempty(newsegs)
            tdead = t; break
        end
        segs, newsegs = newsegs, segs
    end
    return (; tsN, tsR2, tsA, tsNdef, tsNB, tsdom, tslam, tlast, tdead, truncated,
              maxrows = size(B.X, 1))
end

"""
    run_seed(p::SeedParams) -> Dict{String,Any}

All runs (threaded). Per-run arrays (nt × nruns, Float32 except R2): `N`, `R2`,
`A`, `Ndef`, `NB`, `ndom`, `lam`; per run `tlast`, `tdead`, `truncated`, `maxrows`, `group`.
Derived (all runs): `P` survival = fraction with an active site at some t' ≥ t,
`Px` = fraction not exactly absorbed by t, `Nmean` = ⟨N(t)⟩ over all runs,
`R2mean` = Σ R2 / Σ N (mean-square distance of active sites from the seed).
"""
function run_seed(p::SeedParams; verbose::Bool = false)
    p.dt > 0 || throw(ArgumentError("dt must be positive"))
    spl = Splitting(p.τ, p.dt; order = p.order)
    tg = time_grid(p.T; per_decade = p.per_decade)
    ng, R = length(tg), p.nruns
    Nn = zeros(Float32, ng, R); R2 = zeros(ng, R); A = zeros(Float32, ng, R); Nd = zeros(Float32, ng, R)
    NB = zeros(Float32, ng, R); dom = zeros(Float32, ng, R); lam = fill(NaN32, ng, R)
    tlast = zeros(Int, R); tdead = zeros(Int, R); trunc = falses(R); maxrows = zeros(Int, R)
    t0 = time()
    Threads.@threads :dynamic for r in 1:R
        o = run_seed_single(p, r; spl, tg)
        Nn[:, r] .= o.tsN; R2[:, r] .= o.tsR2; A[:, r] .= o.tsA; Nd[:, r] .= o.tsNdef
        NB[:, r] .= o.tsNB; dom[:, r] .= o.tsdom; lam[:, r] .= o.tslam
        tlast[r] = o.tlast; tdead[r] = o.tdead; trunc[r] = o.truncated; maxrows[r] = o.maxrows
    end
    P = [mean(tlast .>= t) for t in tg]
    Px = [mean((tdead .== 0) .| (tdead .> t)) for t in tg]
    Nmean = vec(mean(Float64.(Nn); dims = 2))
    R2mean = vec(sum(R2; dims = 2)) ./ vec(sum(Float64.(Nn); dims = 2))
    return Dict{String,Any}(
        "mode" => "seed", "N" => p.N, "a" => p.a, "T" => p.T, "model" => String(p.model), "tau" => p.τ,
        "dt" => p.dt, "order" => p.order, "nruns" => R, "ngroups" => p.ngroups, "thr" => p.thr,
        "epsB" => p.epsB, "epsw" => p.epsw, "lyapunov" => p.lyapunov, "seed" => p.seed,
        "group" => [mod1(r, p.ngroups) for r in 1:R], "ac_linear" => ac_linear(p.N; τ = p.τ),
        "tgrid" => tg, "N_t" => Nn, "R2_t" => R2, "A_t" => A, "Ndef_t" => Nd, "NB_t" => NB, "ndom_t" => dom,
        "lam_t" => lam, "tlast" => tlast, "tdead" => tdead, "truncated" => collect(trunc), "maxrows" => maxrows,
        "P" => P, "Px" => Px, "Nmean" => Nmean, "R2mean" => R2mean,
        "walltime" => time() - t0, "nthreads" => Threads.nthreads(), "julia" => string(VERSION),
        "code_version" => "0.2.0")
end
