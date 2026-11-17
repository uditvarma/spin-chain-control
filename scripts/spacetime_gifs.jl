#!/usr/bin/env julia
#
# Space-time heat-map GIFs of the local error |S_j(t) - S0_j| across the control transition.
#
#   julia -t auto --project=scripts scripts/spacetime_gifs.jl [OUTDIR]          the three paper GIFs
#   julia -t auto --project=scripts scripts/spacetime_gifs.jl OUTDIR N L amin amax nframes [T]
#
# One frame per a: a single trajectory (static model, τ = 1, dt = 0.1, 2nd-order splitting) from
# HeisenbergControl.spacetime, started from the same random state (sample 1) for every a, on a fixed
# colour scale [0, 2]. Defaults: T = L, every step recorded, 1 frame/s, 1500×1200 px.
# Paper GIFs:  N = 3, L = 1500, a = 0:0.05:1;  N = 4, L = 1600 and N = 5, L = 1500, a = 0.5:0.025:1.

using HeisenbergControl, Plots, LaTeXStrings, Printf
gr()
default(framestyle = :box, grid = false, guidefontsize = 11, tickfontsize = 8,
        titlefontsize = 11, fontfamily = "Computer Modern")

"""
    make_gif(N, L, avals; T = L, stride = 1, sample = 1, clims = (0, 2), fps = 1, dpi = 250, outfile)

One frame per value in `avals`. `L` must be even and a multiple of `N`.
Frames are about (6 dpi) × (4.8 dpi) pixels.
"""
function make_gif(N, L, avals; T = L, stride = 1, sample = 1, clims = (0.0, 2.0), fps = 1, dpi = 250,
                  outfile = "spacetime_N$(N)_L$(L).gif")
    iseven(L) && L % N == 0 || error("L = $L must be even and a multiple of N = $N")
    avals = collect(avals)
    runs = Vector{Any}(undef, length(avals))
    Threads.@threads for i in eachindex(avals)          # trajectories in parallel, frames in order
        runs[i] = spacetime(N, L, avals[i], T; stride, sample)
    end
    anim = Animation()
    for (a, s) in zip(avals, runs)
        sd = sum(@view s.D[:, end]) / L
        p  = heatmap(1:L, s.t, s.D'; color = :Blues, clims,
                     xlabel = L"\mathrm{site}\ j", ylabel = L"t",
                     colorbar_title = L"|\mathbf{S}_j(t)-\mathbf{S}^0_j|",
                     title = latexstring(@sprintf("N = %d,\\ L = %d,\\ a = %.3f,\\ S_{\\mathrm{diff}}(T) = %.3g",
                                                  N, L, a, sd)),
                     size = (600, 480), dpi, right_margin = 4Plots.mm)
        frame(anim, p)
        @printf("N = %d  a = %.3f   S_diff(T) = %.4g\n", N, a, sd)
    end
    return gif(anim, outfile; fps)
end

outdir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "figures")
mkpath(outdir)
if length(ARGS) >= 6
    N, L = parse(Int, ARGS[2]), parse(Int, ARGS[3])
    amin, amax = parse(Float64, ARGS[4]), parse(Float64, ARGS[5]); nf = parse(Int, ARGS[6])
    T = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : L
    make_gif(N, L, range(amin, amax; length = nf); T, outfile = joinpath(outdir, "spacetime_N$(N)_L$(L).gif"))
else
    for (N, L, avals) in ((3, 1500, range(0.0, 1.0; length = 21)),
                          (4, 1600, range(0.5, 1.0; length = 21)),
                          (5, 1500, range(0.5, 1.0; length = 21)))
        make_gif(N, L, avals; outfile = joinpath(outdir, "spacetime_N$(N)_L$(L).gif"))
    end
end
