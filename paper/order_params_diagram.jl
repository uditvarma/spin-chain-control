#!/usr/bin/env julia
#
# Phase diagrams in (N, a) of the long-range tilt correlator G_x and the Lyapunov exponent λ
# (and S_diff) from the output of scripts/order_params_scan.jl.
#
#   julia --project=scripts paper/order_params_diagram.jl [SCANDIR] [OUTDIR]
#       SCANDIR  default results/order_params_L1200
#       OUTDIR   default figures
#
# Writes OUTDIR/order_param_{Gx,lambda,Sdiff}_N3-20_clean.pdf and the 3-panel order_params_N3-20_clean.pdf
# (the files in paper/; "clean" = without the front-race points used in the working notes).
# Colour: sample mean of the per-sample steady-state values.
#
#   G_x = 1/(L(b+1)) Σ_j Σ_{r=b}^{2b} S^x_j S^x_{j+r}      (b = 64 by default)
#
# G_x vanishes on the helix (S^x_j = 0) and, for b ≫ ξ, in the chaotic phase (~ e^{-b/ξ}); inside the
# intermediate cone phase it is s_+^2, the squared tilt of the cone.
# Lines: a_c^lin(N) (linear instability of the helix; continuous N ≥ 5 as a guide, markers = lattice values
# for the simulated L) and a_sn(N) = 1/(1 + 2 G_max), G_max = max_s sqrt(1-s²) sin(κ s/2),
# κ = 2τ(1 - cos 2π/N): saddle-node where the cone fixed point first exists.

using HeisenbergControl, Plots, LaTeXStrings, Printf
gr()
default(framestyle = :box, grid = false, guidefontsize = 12, tickfontsize = 9,
        titlefontsize = 11, fontfamily = "Computer Modern")

const ROOT = normpath(joinpath(@__DIR__, ".."))
scandir = length(ARGS) >= 1 ? ARGS[1] : joinpath(ROOT, "results", "order_params_L1200")
outdir  = length(ARGS) >= 2 ? ARGS[2] : joinpath(ROOT, "figures")
mkpath(outdir)

# ---- read the per-N CSVs: (N, a) -> sample means -------------------------------------------------
const KEYS = ("Sdiff", "Gxr", "lam")
files = sort(filter(f -> occursin(r"^N\d+\.csv$", f), readdir(scandir)))
isempty(files) && error("no N??.csv under $scandir — run scripts/order_params_scan.jl first")
data = Dict{Tuple{Int,Float64},Vector{Vector{Float64}}}(); Lof = Dict{Int,Int}()
for f in files
    ls = readlines(joinpath(scandir, f)); h = Dict(k => i for (i, k) in enumerate(split(ls[1], ',')))
    for l in ls[2:end]
        r = split(l, ','); N = parse(Int, r[h["N"]]); a = parse(Float64, r[h["a"]])
        Lof[N] = parse(Int, r[h["L"]])
        push!(get!(data, (N, a), Vector{Float64}[]), [parse(Float64, r[h[k]]) for k in KEYS])
    end
end
NS    = sort(collect(keys(Lof)))
avals = sort(unique(last.(keys(data))))
OP = Dict(k => fill(NaN, length(avals), length(NS)) for k in KEYS)
for ((N, a), v) in data
    i = searchsortedfirst(avals, a); j = searchsortedfirst(NS, N)
    for (n, k) in enumerate(KEYS)
        OP[k][i, j] = sum(x[n] for x in v) / length(v)
    end
end
OP["lam"] .= max.(OP["lam"], -10.0)                 # a = 0: λ = ln a = -Inf
Ls = [Lof[N] for N in NS]
@printf("N = %d … %d, a = %.2f … %.2f (%d values), max G_x = %.3f\n",
        first(NS), last(NS), first(avals), last(avals), length(avals), maximum(filter(!isnan, OP["Gxr"])))

# ---- analytic thresholds --------------------------------------------------------------------------
function ac_lin_cont(Nc; τ = 1.0, ks = range(0, π; length = 20_001))
    c = cos(2π / Nc)
    λmax = maximum(k -> (λ2 = -4c * (cos(k) - c) * (cos(k) - 1); λ2 > 0 ? sqrt(λ2) : 0.0), ks)
    return exp(-τ * λmax)
end
function a_sn(Nc; τ = 1.0, ss = range(0, 1; length = 20_001))
    κ = 2τ * (1 - cos(2π / Nc))
    return 1 / (1 + 2 * maximum(s -> sqrt(1 - s^2) * sin(κ * s / 2), ss))
end
Ncont  = range(max(5, first(NS)), last(NS); length = 600)   # closed form spikes to 1 at N = 4
accont = ac_lin_cont.(Ncont)
aclat  = [ac_linear(N; L = L) for (N, L) in zip(NS, Ls)]
Nsn    = range(first(NS), last(NS); length = 600)

# ---- figures --------------------------------------------------------------------------------------
# Each (N, a) cell is drawn as a block of pixels: GR embeds a heatmap as one pixel per cell, and PDF
# viewers smooth such small images when scaling them, blurring the cell edges.
function blocky(xs, ys, Z; rx = 24, ry = 6)
    dx = length(xs) > 1 ? xs[2] - xs[1] : 1.0; dy = length(ys) > 1 ? ys[2] - ys[1] : 0.01
    xf = [x - dx/2 + (k - 0.5) * dx / rx for x in xs for k in 1:rx]
    yf = [y - dy/2 + (k - 0.5) * dy / ry for y in ys for k in 1:ry]
    return xf, yf, repeat(Z; inner = (ry, rx))
end
da = length(avals) > 1 ? avals[2] - avals[1] : 0.01

function oppanel(key; cmap, clims, cb, legend = :bottomright)
    xf, yf, Zf = blocky(NS, avals, OP[key])
    p = heatmap(xf, yf, Zf; color = cmap, clims, colorbar_title = cb,
                xlabel = L"N", ylabel = L"a", xticks = NS, label = "", background_color_inside = :gray90)
    plot!(p, Ncont, accont; color = "#eb6834", lw = 2, label = L"a_c^{\mathrm{lin}}")
    scatter!(p, NS, aclat; color = "#eb6834", markerstrokecolor = :white, markerstrokewidth = 1, markersize = 4, label = "")
    plot!(p, Nsn, a_sn.(Nsn); color = :black, lw = 1.2, ls = :dash, label = L"a_{\mathrm{sn}}")
    scatter!(p, NS, a_sn.(NS); marker = :diamond, color = :white, markerstrokecolor = :black, markerstrokewidth = 1,
             markersize = 5, label = "")
    plot!(p; xlims = (first(NS) - 0.5, last(NS) + 0.5), ylims = (first(avals) - da/2, last(avals) + da/2),
          widen = false, legend, legendfontsize = 8)
    return p
end

panels = (("Gxr",   "Gx",     (cmap = :Blues, clims = (0, 1.0), cb = L"G_x")),
          ("lam",   "lambda", (cmap = cgrad(:RdBu, rev = true), clims = (-0.3, 0.3), cb = L"\lambda")),
          ("Sdiff", "Sdiff",  (cmap = :Blues, clims = (0, 1.4), cb = L"S_{\mathrm{diff}}")))
tag = "N$(first(NS))-$(last(NS))"
for (key, name, kw) in panels
    p = oppanel(key; kw...)
    plot!(p; size = (620, 480), dpi = 200, left_margin = 3Plots.mm, bottom_margin = 3Plots.mm, right_margin = 4Plots.mm)
    f = joinpath(outdir, "order_param_$(name)_$(tag)_clean.pdf"); savefig(p, f); println("wrote ", f)
end
P = plot(oppanel("Sdiff"; panels[3][3]...), oppanel("Gxr"; panels[1][3]..., legend = false),
         oppanel("lam"; panels[2][3]..., legend = false);
         layout = (1, 3), size = (1500, 470), dpi = 200, xtickfontsize = 6,
         left_margin = 5Plots.mm, bottom_margin = 6Plots.mm, right_margin = 3Plots.mm)
f = joinpath(outdir, "order_params_$(tag)_clean.pdf"); savefig(P, f); println("wrote ", f)
