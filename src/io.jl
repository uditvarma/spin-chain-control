# ---------------------------------------------------------------------------
# Saving / loading, reductions, TOML configs.
# ---------------------------------------------------------------------------

"""
    point_filename(dir, N, L, a, model) -> path

One JLD2 file per (model, N, L, a): `dir/N4/L256/<model>_N4_L256_a0.62000.jld2`.
"""
point_filename(dir::AbstractString, N::Int, L::Int, a::Real, model) =
    joinpath(dir, "N$N", "L$L", @sprintf("%s_N%d_L%d_a%.5f.jld2", model, N, L, a))

"""
    seed_filename(dir, N, a, model) -> path      (`dir/N4/seed/<model>_N4_seed_a0.62000.jld2`)
"""
seed_filename(dir::AbstractString, N::Int, a::Real, model) =
    joinpath(dir, "N$N", "seed", @sprintf("%s_N%d_seed_a%.5f.jld2", model, N, a))

"""
    save_point(file, res::Dict)

Atomic write (tmp file + rename) so interrupted jobs never leave corrupt files.
"""
function save_point(file::AbstractString, res::Dict{String,Any})
    mkpath(dirname(file))
    tmp = file * ".tmp"
    jldopen(tmp, "w"; compress = true) do f
        for (k, v) in res
            f[k] = v
        end
    end
    mv(tmp, file; force = true)
    return file
end

"""
    load_point(file) -> Dict{String,Any}
"""
load_point(file::AbstractString) = jldopen(f -> Dict{String,Any}(k => f[k] for k in keys(f)), file, "r")

_sem(x) = length(x) > 1 ? std(x) / sqrt(length(x)) : NaN

"""
    lambda_window(res, t1, t2) -> Vector (per sample)

Per-sample Lyapunov exponent averaged over grid blocks fully inside (t1, t2]
(weighted by block length). Use e.g. (t_f/2, 2t_f] as in the paper.
"""
function lambda_window(res::Dict, t1::Real, t2::Real)
    tg = res["tgrid"]; lam = res["ts_lam"]
    num = zeros(size(lam, 2)); den = 0
    prev = 0
    for (k, t) in enumerate(tg)
        w = t - prev
        if prev >= t1 && t <= t2
            num .+= w .* Float64.(@view lam[k, :]); den += w
        end
        prev = t
    end
    return num ./ den
end

"""
    classify(res; λB = 0.15, s_ctrl = 1e-8) -> Vector{Symbol}

Per-sample phase label from the steady-state data:
  * `:control`  — absorbed onto the helix (steady-state S_diff < s_ctrl)
  * `:brighton` — not absorbed and |λ| ≤ λB  (non-chaotic coherent structures, λ ≈ 0)
  * `:chaos`    — not absorbed and λ > λB
  * `:other`    — not absorbed and λ < -λB (slow transients; should be rare)
Control is defined by absorption, not by λ, because the fixed-point exponent
λ_fp = log a + max_k λ_k is ≈ 0 near a_c^lin for N > 4. The chaotic λ scale is
N-dependent (≈ 0.3 for N = 4, ≈ 0.05–0.08 for N = 10), so look at the λ
histograms (analysis/plots.jl) before trusting a fixed λB.
"""
function classify(res::Dict; λB::Real = 0.15, s_ctrl::Real = 1e-8)
    return map(zip(res["ss_Sdiff"], res["ss_lam"])) do (sd, λ)
        sd < s_ctrl ? :control : abs(λ) <= λB ? :brighton : λ > λB ? :chaos : :other
    end
end

"""
    summarize(dir; λB = 0.15, s_ctrl = 1e-8) -> Vector{NamedTuple}

Walk `dir` and reduce each point to sample means, standard errors and
sample-to-sample standard deviations (susceptibility-like) of the steady-state
observables, plus the per-sample phase fractions from `classify` and the mean
f_B / n_dom restricted to brighton-classified samples. Rows are sorted by
(model, N, L, a). Write with `write_summary_csv`.
"""
function summarize(dir::AbstractString; λB::Real = 0.15, s_ctrl::Real = 1e-8)
    rows = NamedTuple[]
    for (root, _, files) in walkdir(dir), f in files
        endswith(f, ".jld2") || continue
        r = load_point(joinpath(root, f))
        get(r, "mode", "hom") == "seed" && continue
        cls = classify(r; λB, s_ctrl)
        isB = cls .== :brighton
        Sd = r["ss_Sdiff"]; λ = r["ss_lam"]; fB = r["ss_fB"]; ρ = r["ss_rho"]; nd = r["ss_ndom"]
        row = (model = r["model"], N = r["N"], L = r["L"], a = r["a"], T = r["T"],
               nsamples = r["nsamples"], dt = r["dt"], order = r["order"],
               Sdiff = mean(Sd), Sdiff_sem = _sem(Sd), Sdiff_std = std(Sd),
               ndef = mean(get(r, "ss_ndef", [NaN])), ndef_sem = _sem(get(r, "ss_ndef", [NaN])),
               ndef_std = std(get(r, "ss_ndef", [NaN])),
               lam = mean(λ), lam_sem = _sem(λ), lam_std = std(λ),
               fB1 = mean(fB[1, :]), fB1_sem = _sem(fB[1, :]), fB1_std = std(fB[1, :]),
               fB_all = vec(mean(fB; dims = 2)), epsB = r["epsB"],
               rho = mean(ρ), rho_sem = _sem(ρ), ndom = mean(nd), ndom_sem = _sem(nd),
               frac_absorbed = mean(r["t_abs"] .> 0),
               frac_control = mean(cls .== :control), frac_brighton = mean(isB),
               frac_chaos = mean(cls .== :chaos), frac_other = mean(cls .== :other),
               fB1_brighton = any(isB) ? mean(fB[1, isB]) : NaN,
               ndom_brighton = any(isB) ? mean(nd[isB]) : NaN,
               ac_linear = r["ac_linear"], walltime = r["walltime"])
        push!(rows, row)
    end
    sort!(rows; by = x -> (x.model, x.N, x.L, x.a))
    return rows
end

"""
    write_summary_csv(path, rows)
"""
function write_summary_csv(path::AbstractString, rows)
    cols = [:model, :N, :L, :a, :T, :nsamples, :dt, :order, :Sdiff, :Sdiff_sem, :Sdiff_std, :ndef, :ndef_sem, :ndef_std,
            :lam, :lam_sem, :lam_std, :fB1, :fB1_sem, :fB1_std, :rho, :rho_sem, :ndom, :ndom_sem,
            :frac_absorbed, :frac_control, :frac_brighton, :frac_chaos, :frac_other,
            :fB1_brighton, :ndom_brighton, :ac_linear, :walltime]
    open(path, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join((string(getfield(r, c)) for c in cols), ","))
        end
    end
    return path
end

"""
    summarize_seed(dir; b = 4) -> Vector{NamedTuple}

One row per seed-spreading file: survival P(T), exact survival, ⟨N(T)⟩, ⟨N_B(T)⟩,
R²(T), and the effective exponents δ, θ, 2/z at the last time where the
factor-b window fits (with jackknife errors).
"""
function summarize_seed(dir::AbstractString; b::Real = 4)
    rows = NamedTuple[]
    for (root, _, files) in walkdir(dir), f in files
        endswith(f, ".jld2") || continue
        r = load_point(joinpath(root, f))
        get(r, "mode", "hom") == "seed" || continue
        e = seed_exponents(r; b)
        k = findlast(isfinite, e.delta); k === nothing && (k = 1)
        kθ = findlast(isfinite, e.theta); kθ === nothing && (kθ = 1)
        kz = findlast(isfinite, e.twooverz); kz === nothing && (kz = 1)
        push!(rows, (model = r["model"], N = r["N"], a = r["a"], T = r["T"], nruns = r["nruns"],
                     P_T = r["P"][end], Px_T = r["Px"][end], N_T = r["Nmean"][end],
                     NB_T = mean(r["NB_t"][end, :]), R2_T = r["R2mean"][end],
                     t_eff = e.t[k], delta_eff = e.delta[k], delta_err = e.delta_err[k],
                     theta_eff = e.theta[kθ], theta_err = e.theta_err[kθ],
                     twooverz_eff = e.twooverz[kz], twooverz_err = e.twooverz_err[kz],
                     frac_truncated = mean(r["truncated"]), walltime = r["walltime"]))
    end
    sort!(rows; by = x -> (x.model, x.N, x.a))
    return rows
end

function write_seed_csv(path::AbstractString, rows)
    cols = [:model, :N, :a, :T, :nruns, :P_T, :Px_T, :N_T, :NB_T, :R2_T, :t_eff, :delta_eff, :delta_err,
            :theta_eff, :theta_err, :twooverz_eff, :twooverz_err, :frac_truncated, :walltime]
    open(path, "w") do io
        println(io, join(cols, ","))
        foreach(r -> println(io, join((string(getfield(r, c)) for c in cols), ",")), rows)
    end
    return path
end

# ---------------------------------------------------------------------------
# TOML configs (see configs/*.toml)
# ---------------------------------------------------------------------------

load_config(path::AbstractString) = TOML.parsefile(path)

_avals(x::AbstractVector) = Float64.(x)
function _avals(x::AbstractDict)
    haskey(x, "center") && return x      # window around the linear a_c, filled in by caller
    if haskey(x, "step")
        n = round(Int, (x["stop"] - x["start"]) / x["step"]) + 1
        return collect(range(Float64(x["start"]), Float64(x["stop"]); length = n))
    elseif haskey(x, "length")
        return collect(range(Float64(x["start"]), Float64(x["stop"]); length = x["length"]))
    end
    error("bad `a` specification: $x")
end

function _Tsteps(cfg_time::AbstractDict, L::Int)
    if haskey(cfg_time, "T")
        return Int(cfg_time["T"])
    end
    z = Float64(get(cfg_time, "z", 1.58)); c = Float64(get(cfg_time, "c", 1.0))
    Tmin = Int(get(cfg_time, "Tmin", 100)); Tmax = Int(get(cfg_time, "Tmax", typemax(Int)))
    return clamp(ceil(Int, c * L^z), Tmin, Tmax)
end

"""
    expand_config(cfg) -> Vector{RunParams}

Turn a parsed TOML config into the list of points to run (`RunParams` for
`[[sweep]]` blocks, `SeedParams` for `[[seed_sweep]]` blocks). Each `[[sweep]]`
block gives `N`, a list `L`, and `a` either as a list, as
`{start, stop, step}` / `{start, stop, length}`, or as
`{center = "linear", halfwidth, step}` (window around ac_linear(N), only for N ≠ 2, 4).
Block-level keys override the global ones (e.g. a per-block `nsamples`).
With `round_L = true`, each L is rounded to the nearest multiple of lcm(2, N).
`[time]`: either `T = …` or `T = clamp(ceil(c·L^z), Tmin, Tmax)` with keys z, c, Tmin, Tmax.
"""
function expand_config(cfg::AbstractDict)
    pts = Any[]
    glob = cfg
    for sw in get(cfg, "seed_sweep", [])
        get1s(k, d) = get(sw, k, get(glob, k, d))
        N = Int(sw["N"])
        avs = _avals(sw["a"])
        if avs isa AbstractDict
            ac = ac_linear(N); hw = Float64(avs["halfwidth"]); st = Float64(avs["step"])
            avs = collect(range(ac - hw, ac + hw; length = round(Int, 2hw / st) + 1))
        end
        for a in avs
            push!(pts, SeedParams(; N, a = round(a; digits = 6), T = Int(sw["T"]),
                                  model = Symbol(get1s("model", "static")), τ = Float64(get1s("tau", 1.0)),
                                  dt = Float64(get1s("dt", 0.1)), order = Int(get1s("order", 2)),
                                  nruns = Int(get1s("nruns", 1600)), ngroups = Int(get1s("ngroups", 16)),
                                  thr = Float64(get1s("thr", 0.05)), epsB = Float64(get1s("epsB_seed", 1e-3)),
                                  epsw = Float64(get1s("epsw", 1e-12)), lyapunov = Bool(get1s("lyapunov_seed", false)),
                                  per_decade = Int(get1s("per_decade", 40)), seed = Int(get1s("seed", 1))))
        end
    end
    for sw in get(cfg, "sweep", [])
        get1(k, d) = get(sw, k, get(glob, k, d))
        N = Int(sw["N"])
        model = Symbol(get1("model", "static"))
        tcfg = merge(get(glob, "time", Dict()), get(sw, "time", Dict()))
        avs = _avals(sw["a"])
        if avs isa AbstractDict
            ac = ac_linear(N)
            hw = Float64(avs["halfwidth"]); st = Float64(avs["step"])
            n = round(Int, 2hw / st) + 1
            avs = collect(range(ac - hw, ac + hw; length = n))
        end
        Ls = Int.(sw["L"])
        if Bool(get1("round_L", false))          # round each L to the nearest multiple of lcm(2, N)
            q = lcm(2, N)
            Ls = unique(max.(q, q .* round.(Int, Ls ./ q)))
        end
        for L in Ls, a in avs
            push!(pts, RunParams(; N, L, a = round(a; digits = 6),
                                 T = _Tsteps(tcfg, L), model,
                                 τ = Float64(get1("tau", 1.0)), dt = Float64(get1("dt", 0.1)),
                                 order = Int(get1("order", 2)),
                                 nsamples = Int(get1("nsamples", 100)),
                                 lyapunov = Bool(get1("lyapunov", true)),
                                 epsB = Float64.(get1("epsB", [1e-3, 1e-2, 5e-2])),
                                 rho_eps = Float64(get1("rho_eps", 0.1)),
                                 ss_frac = Float64(get1("ss_frac", 0.25)),
                                 per_decade = Int(get1("per_decade", 40)),
                                 absorb_tol = Float64(get1("absorb_tol", 1e-10)),
                                 Tabs_max = Int(get1("Tabs_max", 0)),
                                 seed = Int(get1("seed", 1))))
        end
    end
    return pts
end
