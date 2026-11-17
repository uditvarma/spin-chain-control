# ---------------------------------------------------------------------------
# Universality analysis (no plotting dependencies):
#   * absorption-time quantiles τ_Q(L) with bootstrap errors (censored runs = ∞)
#   * local slopes d ln y / d ln t over a factor-b window, δ_n(t) = log_b[n(t)/n(bt)]
#   * seed-spreading exponents (δ, θ, 2/z) with jackknife errors over run groups
#   * finite-size-scaling collapse  L^b n(t, L) = G(t / L^z)  by grid search,
#     with errors from a bootstrap over groups of samples
# ---------------------------------------------------------------------------

"""
    quantile_boot(x, q; nboot = 500, rng = Xoshiro(1)) -> (value, stderr)

q-quantile of `x` (may contain `Inf` for censored runs) and its bootstrap error.
Returns (Inf, NaN) if fewer than a fraction q of the entries are finite.
"""
function quantile_boot(x::AbstractVector{<:Real}, q::Real; nboot::Int = 500, rng = Xoshiro(1))
    xs = sort(Float64.(x))
    v = quantile(xs, q; sorted = true)
    isfinite(v) || return (Inf, NaN)
    bs = [quantile(xs[rand(rng, 1:length(xs), length(xs))], q) for _ in 1:nboot]
    bs = filter(isfinite, bs)
    return v, length(bs) > 1 ? std(bs) : NaN
end

"""
    absorption_quantile(res, q; nboot = 500) -> (τ_q, stderr)

From a `run_point` result: time by which a fraction q of samples was absorbed
(t_abs = 0, i.e. not absorbed by max(T, Tabs_max), counts as ∞).
"""
absorption_quantile(res::Dict, q::Real; nboot::Int = 500) =
    quantile_boot([t == 0 ? Inf : Float64(t) for t in res["t_abs"]], q; nboot)

"""
    interp_loglog(t, y, tq) -> y(tq)

Linear interpolation in (ln t, ln y); `NaN` outside the range or where y ≤ 0.
"""
function interp_loglog(t::AbstractVector, y::AbstractVector, tq::Real)
    (tq < t[1] || tq > t[end]) && return NaN
    k = searchsortedlast(t, tq)
    k == length(t) && return Float64(y[end])
    y1, y2 = y[k], y[k+1]
    (y1 <= 0 || y2 <= 0) && return NaN
    w = (log(tq) - log(t[k])) / (log(t[k+1]) - log(t[k]))
    return exp((1 - w) * log(y1) + w * log(y2))
end

"""
    local_slopes(t, y; b = 4) -> Vector

Effective exponent d ln y / d ln t at each t, from y(t√b)/y(t/√b) (NaN where the
window leaves the data).
"""
local_slopes(t::AbstractVector, y::AbstractVector; b::Real = 4) =
    [log(interp_loglog(t, y, s * sqrt(b)) / interp_loglog(t, y, s / sqrt(b))) / log(b) for s in t]

"""
    delta_n(t, n; b = 2.5) -> (ts, δ_n)

Sierant–Turkeshi estimator δ_n(t) = log_b[n(t) / n(bt)]: constant (= δ) at the
critical point, bends down (up) in the absorbing (active) phase.
"""
function delta_n(t::AbstractVector, n::AbstractVector; b::Real = 2.5)
    keep = [s * b <= t[end] for s in t]
    ts = t[keep]
    return ts, [log(interp_loglog(t, n, s) / interp_loglog(t, n, b * s)) / log(b) for s in ts]
end

"""
    mean_series(res, key) -> Vector

Sample average of a per-sample time series (`"ts_Sdiff"`, `"ts_ndef"`, `"ts_rho"`,
`"ts_lam"`, …) of a `run_point` result; absorbed samples count as 0.
"""
mean_series(res::Dict, key::AbstractString) = vec(mean(Float64.(res[key]); dims = 2))

# ---------------------------------------------------------------------------
# Seed spreading
# ---------------------------------------------------------------------------

"""
    seed_curves(res; groups = all) -> (; t, P, N, R2)

Survival P(t) (an active site at some t' ≥ t), mean number of active sites N(t)
(dead runs count 0) and mean-square spread R²(t) = Σ R2 / Σ N, from the runs in
the given groups.
"""
function seed_curves(res::Dict; groups = 1:res["ngroups"])
    sel = in(Set(groups)).(res["group"])
    tg = res["tgrid"]; tl = res["tlast"][sel]
    Nn = Float64.(res["N_t"][:, sel]); R2 = res["R2_t"][:, sel]
    P = [mean(tl .>= t) for t in tg]
    N = vec(mean(Nn; dims = 2))
    R2m = vec(sum(R2; dims = 2)) ./ vec(sum(Nn; dims = 2))
    return (; t = tg, P, N, R2 = R2m)
end

"""
    seed_exponents(res; b = 4) -> NamedTuple

Effective exponents vs t with jackknife errors over the run groups:
`delta` = -d ln P/d ln t, `theta` = d ln N/d ln t, `twooverz` = d ln R²/d ln t
(DP 1+1: δ = 0.1595, θ = 0.3137, 2/z = 1.2652). Brighton phase: P, N → const,
2/z → 2 (ballistic).
"""
function seed_exponents(res::Dict; b::Real = 4)
    G = res["ngroups"]
    slopes(c) = (-local_slopes(c.t, c.P; b), local_slopes(c.t, c.N; b), local_slopes(c.t, c.R2; b))
    full = slopes(seed_curves(res))
    J = [slopes(seed_curves(res; groups = setdiff(1:G, [g]))) for g in 1:G]
    jk(i) = [sqrt((G - 1) / G * sum((J[g][i][k] - mean(J[h][i][k] for h in 1:G))^2 for g in 1:G))
             for k in eachindex(res["tgrid"])]
    return (; t = res["tgrid"], delta = full[1], delta_err = jk(1), theta = full[2], theta_err = jk(2),
              twooverz = full[3], twooverz_err = jk(3))
end

# ---------------------------------------------------------------------------
# Finite-size-scaling collapse
# ---------------------------------------------------------------------------

"""
    collapse_cost(curves, z, b; tmin = 10, npts = 40) -> Float64

`curves` = [(L, t, y), …]. Rescale x = t/L^z, Y = L^b y and return the mean
squared difference of ln Y between every pair of curves on their overlap in
ln x (interpolated on `npts` points). Large if the overlaps are empty.
"""
function collapse_cost(curves, z::Real, b::Real; tmin::Real = 10, npts::Int = 40)
    resc = map(curves) do (L, t, y)
        k = (t .>= tmin) .& (y .> 0)
        (log.(t[k] ./ L^z), log.(y[k]) .+ b * log(L))
    end
    cost = 0.0; npair = 0
    for i in 1:length(resc), j in i+1:length(resc)
        xi, yi = resc[i]; xj, yj = resc[j]
        (length(xi) < 2 || length(xj) < 2) && continue
        lo = max(xi[1], xj[1]); hi = min(xi[end], xj[end])
        hi > lo || continue
        for x in range(lo, hi; length = npts)
            cost += (_lininterp(xi, yi, x) - _lininterp(xj, yj, x))^2
        end
        npair += 1
    end
    return npair == 0 ? Inf : cost / (npair * npts)
end

function _lininterp(x, y, xq)
    k = clamp(searchsortedlast(x, xq), 1, length(x) - 1)
    w = (xq - x[k]) / (x[k+1] - x[k])
    return (1 - w) * y[k] + w * y[k+1]
end

"""
    grid_collapse(curves; zs = 1.0:0.005:2.0, bs = 0.0:0.002:0.5, tmin = 10) -> (z, b, cost)

Best (z, b = β/ν⊥) for the collapse L^b y(t, L) = G(t/L^z) by grid search.
"""
function grid_collapse(curves; zs = 1.0:0.005:2.0, bs = 0.0:0.002:0.5, tmin::Real = 10)
    best = (NaN, NaN, Inf)
    for z in zs, b in bs
        c = collapse_cost(curves, z, b; tmin)
        c < best[3] && (best = (z, b, c))
    end
    return best
end

"""
    collapse_fit(results; key = "ts_rho", ngroups = 20, nboot = 200, zs, bs, tmin) -> NamedTuple

FSS collapse of the sample-averaged time series `key` for a set of `run_point`
results at the same (N, a) and different L (use T ≈ 4 L^z). Errors from a
bootstrap over `ngroups` groups of samples per L. Returns z, b (= β/ν⊥), their
errors and correlation, the bootstrap samples, and `at_boundary` (optimum on the
grid edge ⇒ no collapse: off-critical, or widen the grid).
"""
function collapse_fit(results::AbstractVector; key::AbstractString = "ts_rho", ngroups::Int = 20,
                      nboot::Int = 200, zs = 1.0:0.01:2.0, bs = 0.0:0.005:0.5, tmin::Real = 10,
                      rng = Xoshiro(7))
    grp = map(results) do r
        Y = Float64.(r[key]); ns = size(Y, 2)
        g = [vec(mean(Y[:, mod1.(1:ns, ngroups) .== k]; dims = 2)) for k in 1:ngroups]
        (r["L"], Float64.(r["tgrid"]), g)
    end
    curves(pick) = [(L, t, reduce(+, g[pick]) ./ length(pick)) for (L, t, g) in grp]
    z, b, _ = grid_collapse(curves(1:ngroups); zs, bs, tmin)
    B = zeros(nboot, 2)
    for s in 1:nboot
        B[s, 1], B[s, 2], _ = grid_collapse(curves(rand(rng, 1:ngroups, ngroups)); zs, bs, tmin)
    end
    at_boundary = z in (first(zs), last(zs)) || b in (first(bs), last(bs))
    at_boundary && @warn "collapse optimum on the edge of the (z, b) grid — not a critical point, or widen zs/bs" z b
    return (; z, b, z_err = std(B[:, 1]), b_err = std(B[:, 2]), corr = cor(B[:, 1], B[:, 2]), boot = B, at_boundary)
end

"""
    fss_at(res, c, z; key = "ts_rho") -> (mean, sem)

Per-sample value of the time series `key` at t = c·L^z (interpolated), averaged.
"""
function fss_at(res::Dict, c::Real, z::Real; key::AbstractString = "ts_rho")
    tg = Float64.(res["tgrid"]); Y = Float64.(res[key]); tq = c * res["L"]^z
    v = [_lininterp(tg, Y[:, s], tq) for s in axes(Y, 2)]
    return mean(v), std(v) / sqrt(length(v))
end
