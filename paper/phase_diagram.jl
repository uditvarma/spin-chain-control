#!/usr/bin/env julia
#
# Phase diagram in (N, a): heatmap of the steady-state S_diff with the linear threshold a_c^lin(N).
#
#   julia --project=scripts paper/phase_diagram.jl [SCANDIR] [OUTDIR] [NMIN]
#       SCANDIR  output of scripts/phase_scan.sh   (default results/sdiff_phase_L2520; only read)
#       OUTDIR   where the figure goes             (default figures)
#       NMIN     smallest N drawn                  (default 3)
#
# Colour: sample mean over random initial states of S_diff = (1/L) Σ_j |S_j - S0_j|, averaged
# over the last 25 % of the steps. Also writes OUTDIR/phase_Sdiff_summary.csv
# (N, L, a, ns, Sdiff, Sdiff_sem).
#
# Analytic curve: λ(k,N)² = -4 cos q (cos k - cos q)(cos k - 1), q = 2π/N, a_c^lin = exp(-τ max_k λ).
# Line: continuous N ≥ 5 (guide); markers: lattice momenta k = 2πm/L for the simulated L.

using HeisenbergControl, Plots, LaTeXStrings, Printf, Statistics
gr()
default(framestyle = :box, grid = false, guidefontsize = 12, tickfontsize = 9,
        titlefontsize = 11, fontfamily = "Computer Modern")

const ROOT = normpath(joinpath(@__DIR__, ".."))
scandir = length(ARGS) >= 1 ? ARGS[1] : joinpath(ROOT, "results", "sdiff_phase_L2520")
outdir  = length(ARGS) >= 2 ? ARGS[2] : joinpath(ROOT, "figures")
nmin    = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
mkpath(outdir)

# ---- aggregate the per-sample CSVs of the C scan: (N, a) -> mean S_diff --------------------------
rows = NamedTuple[]
for d in sort(readdir(scandir))
    m = match(r"^N(\d+)$", d); m === nothing && continue
    N = parse(Int, m[1]); N < nmin && continue
    for f in sort(readdir(joinpath(scandir, d)))
        endswith(f, "_samples.csv") || continue
        ls = readlines(joinpath(scandir, d, f)); h = Dict(k => i for (i, k) in enumerate(split(ls[1], ',')))
        r = [split(l, ',') for l in ls[2:end]]
        x = [parse(Float64, q[h["Sdiff_ss"]]) for q in r]
        push!(rows, (N = N, L = parse(Int, r[1][h["L"]]), a = parse(Float64, r[1][h["a"]]), ns = length(x),
                     Sdiff = mean(x), Sdiff_sem = std(x) / sqrt(length(x))))
    end
end
isempty(rows) && error("no *_samples.csv under $scandir — run scripts/phase_scan.sh first")
open(joinpath(outdir, "phase_Sdiff_summary.csv"), "w") do io
    println(io, "N,L,a,ns,Sdiff,Sdiff_sem")
    for r in rows
        @printf(io, "%d,%d,%.3f,%d,%.10g,%.6g\n", r.N, r.L, r.a, r.ns, r.Sdiff, r.Sdiff_sem)
    end
end

NS    = sort(unique(r.N for r in rows))
avals = sort(unique(r.a for r in rows))
Z  = fill(NaN, length(avals), length(NS))                 # Z[a, N]
Ls = zeros(Int, length(NS))
for r in rows
    i = searchsortedfirst(avals, r.a); j = searchsortedfirst(NS, r.N)
    Z[i, j] = r.Sdiff; Ls[j] = r.L
end
any(isnan, Z) && @warn "missing (N, a) points (shown blank)" count(isnan, Z)
@printf("N = %d … %d, a = %.2f … %.2f (%d values), max S_diff = %.3f\n",
        first(NS), last(NS), first(avals), last(avals), length(avals), maximum(filter(!isnan, Z)))

# ---- analytic threshold ---------------------------------------------------------------------------
function ac_lin_cont(Nc; τ = 1.0, ks = range(0, π; length = 20_001))
    c = cos(2π / Nc)
    λmax = maximum(k -> (λ2 = -4c * (cos(k) - c) * (cos(k) - 1); λ2 > 0 ? sqrt(λ2) : 0.0), ks)
    return exp(-τ * λmax)
end
Ncont  = range(max(5, first(NS)), last(NS); length = 600)   # closed form spikes to 1 at N = 4
accont = ac_lin_cont.(Ncont)
aclat  = [ac_linear(N; L = L) for (N, L) in zip(NS, Ls)]

# ---- figure ---------------------------------------------------------------------------------------
P = heatmap(NS, avals, Z; color = :Blues, clims = (0, ceil(maximum(filter(!isnan, Z)); digits = 1)),
            xlabel = L"N", ylabel = L"a", colorbar_title = L"S_{\mathrm{diff}}",
            xticks = NS, label = "", size = (720, 520), dpi = 200,
            left_margin = 3Plots.mm, right_margin = 5Plots.mm, bottom_margin = 3Plots.mm)
plot!(P, Ncont, accont; color = "#eb6834", lw = 2.5, label = L"a_c^{\mathrm{lin}}(N)")
scatter!(P, NS, aclat; color = "#eb6834", markerstrokecolor = :white, markerstrokewidth = 1,
         markersize = 5, label = "")
da = length(avals) > 1 ? avals[2] - avals[1] : 0.01
plot!(P; xlims = (first(NS) - 0.5, last(NS) + 0.5), ylims = (first(avals) - da / 2, last(avals) + da / 2),
      widen = false, legend = :bottomright, legendfontsize = 10, foreground_color_legend = nothing)
f = joinpath(outdir, "phase_Sdiff_N$(first(NS))-$(last(NS)).pdf")
savefig(P, f)
println("wrote ", f)
