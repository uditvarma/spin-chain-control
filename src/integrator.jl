# ---------------------------------------------------------------------------
# Sublattice-splitting integrator for  dS_j/dt = -(h_j × S_j),
#   h_j = (Jx(Sx_{j-1}+Sx_{j+1}), Jy(…), Jz(…)).
#
# With one sublattice frozen, every spin of the other precesses about a constant
# field; that is integrated *exactly* (rotation). Properties (Krech, Bunker &
# Landau, Comput. Phys. Commun. 111, 1 (1998)):
#   * |S_j| = 1 preserved exactly,
#   * energy conserved exactly up to round-off (each bond touches one spin of
#     the active sublattice, and rotating about h_j keeps S_j·h_j fixed),
#   * time-reversible; 2nd order (Strang) or 4th order (Suzuki–Yoshida).
# Requires L even. Works on the SoA layout (see model.jl).
# ---------------------------------------------------------------------------

"""
    Splitting(τ, dt; order = 2)

Precomputed list of sublattice sweeps `(parity, duration)` that integrates the
Hamiltonian flow over a total time τ using substeps ≤ dt. Consecutive sweeps of
the same sublattice are merged.
"""
struct Splitting
    ops::Vector{Tuple{Int,Float64}}
    τ::Float64
    dt::Float64
    order::Int
end

function Splitting(τ::Real, dt::Real; order::Int = 2)
    n = max(1, ceil(Int, τ / dt - 1e-9))
    h = τ / n
    stages = if order == 2
        [h]
    elseif order == 4
        w1 = 1 / (2 - cbrt(2.0))
        w0 = -cbrt(2.0) / (2 - cbrt(2.0))
        [w1 * h, w0 * h, w1 * h]
    else
        throw(ArgumentError("order must be 2 or 4"))
    end
    raw = Tuple{Int,Float64}[]
    for _ in 1:n, c in stages              # Strang block: A(c/2) B(c) A(c/2)
        push!(raw, (1, c / 2)); push!(raw, (2, c)); push!(raw, (1, c / 2))
    end
    ops = Tuple{Int,Float64}[]
    for (p, d) in raw                       # merge A(x)A(y) → A(x+y)
        if !isempty(ops) && ops[end][1] == p
            ops[end] = (p, ops[end][2] + d)
        else
            push!(ops, (p, d))
        end
    end
    return Splitting(ops, Float64(τ), τ / n, order)
end

Base.show(io::IO, s::Splitting) =
    print(io, "Splitting(τ=$(s.τ), dt=$(round(s.dt, sigdigits=4)), order=$(s.order), $(length(s.ops)) sweeps)")

# ---------------------------------------------------------------------------
# Exact precession for constant h (rotation by -|h|dt about h):
#   s' = s cos x - (h × s) dt sinc(x) + h (h·s) dt² (1 - cos x)/x²,   x = |h| dt.
# Written in terms of x² only, so it is smooth (and differentiable for the
# tangent dynamics) at h = 0 — which matters: for the N = 4 helix the local
# field vanishes identically, S_{j-1} + S_{j+1} = 0.
# ---------------------------------------------------------------------------

# Branchless polynomials in x² for (cos x, sin x/x, (1-cos x)/x²): exact to machine
# precision for x² ≤ 1 (first omitted term ≤ 1/20! ≈ 4e-19). Since |h| ≤ 2 max|J|,
# this is used whenever 2·max|J|·|duration| ≤ 1 (checked once per sweep).
const _CC = ntuple(k -> (-1)^(k - 1) / factorial(big(2k - 2)), 10) .|> Float64
const _CS = ntuple(k -> (-1)^(k - 1) / factorial(big(2k - 1)), 10) .|> Float64
const _C2 = ntuple(k -> (-1)^(k - 1) / factorial(big(2k)), 10) .|> Float64
const _DCC = ntuple(k -> k * _CC[k+1], 9)          # d/dx² of the above
const _DCS = ntuple(k -> k * _CS[k+1], 9)
const _DC2 = ntuple(k -> k * _C2[k+1], 9)
@inline _rotfuns_poly(x2) = (evalpoly(x2, _CC), evalpoly(x2, _CS), evalpoly(x2, _C2))
@inline _drotfuns_poly(x2) = (evalpoly(x2, _DCC), evalpoly(x2, _DCS), evalpoly(x2, _DC2))

# Fallback for large substeps (trig functions, Taylor for tiny x²).
@inline function _rotfuns(x2)
    if x2 < 1e-3
        c  = 1 - x2 / 2 * (1 - x2 / 12 * (1 - x2 / 30))
        s1 = 1 - x2 / 6 * (1 - x2 / 20 * (1 - x2 / 42))
        s2 = (1 - x2 / 12 * (1 - x2 / 30 * (1 - x2 / 56))) / 2
    else
        x = sqrt(x2)
        sh, ch = sincos(x / 2)
        c  = 1 - 2 * sh * sh
        s1 = 2 * sh * ch / x
        s2 = 2 * sh * sh / x2
    end
    return c, s1, s2
end
@inline function _drotfuns(x2, c, s1, s2)
    x2 < 1e-3 && return (-1 / 2 + x2 / 12 - x2^2 / 240, -1 / 6 + x2 / 60 - x2^2 / 1680,
                         -1 / 24 + x2 / 360 - x2^2 / 13440)
    return (-s1 / 2, (c - s1) / (2x2), (s1 / 2 - s2) / x2)
end

@inline function _rot(sx, sy, sz, hx, hy, hz, dt, ::Val{P}) where {P}
    dt2 = dt * dt
    x2 = (hx * hx + hy * hy + hz * hz) * dt2
    c, s1, s2 = P ? _rotfuns_poly(x2) : _rotfuns(x2)
    a1 = dt * s1
    hd = (hx * sx + hy * sy + hz * sz) * (dt2 * s2)
    cx = hy * sz - hz * sy
    cy = hz * sx - hx * sz
    cz = hx * sy - hy * sx
    return (sx * c - cx * a1 + hx * hd,
            sy * c - cy * a1 + hy * hd,
            sz * c - cz * a1 + hz * hd)
end

# Rotation and its exact Jacobian-vector product: (s, u = δs; h, g = δh) → (s', u').
@inline function _rot_tan(sx, sy, sz, ux, uy, uz, hx, hy, hz, gx, gy, gz, dt, ::Val{P}) where {P}
    dt2 = dt * dt
    x2 = (hx * hx + hy * hy + hz * hz) * dt2
    δx2 = 2 * (hx * gx + hy * gy + hz * gz) * dt2
    if P
        c, s1, s2 = _rotfuns_poly(x2); cp, s1p, s2p = _drotfuns_poly(x2)
    else
        c, s1, s2 = _rotfuns(x2); cp, s1p, s2p = _drotfuns(x2, c, s1, s2)
    end
    a1 = dt * s1; δa1 = dt * s1p * δx2
    hs = hx * sx + hy * sy + hz * sz
    δhs = gx * sx + gy * sy + gz * sz + hx * ux + hy * uy + hz * uz
    hd = hs * dt2 * s2; δhd = dt2 * (δhs * s2 + hs * s2p * δx2)
    δc = cp * δx2
    cx = hy * sz - hz * sy; cy = hz * sx - hx * sz; cz = hx * sy - hy * sx
    ex = gy * sz - gz * sy + hy * uz - hz * uy        # δ(h × s)
    ey = gz * sx - gx * sz + hz * ux - hx * uz
    ez = gx * sy - gy * sx + hx * uy - hy * ux
    return (sx * c - cx * a1 + hx * hd,
            sy * c - cy * a1 + hy * hd,
            sz * c - cz * a1 + hz * hd,
            δc * sx + c * ux - δa1 * cx - a1 * ex + δhd * hx + hd * gx,
            δc * sy + c * uy - δa1 * cy - a1 * ey + δhd * hy + hd * gy,
            δc * sz + c * uz - δa1 * cz - a1 * ez + δhd * hz + hd * gz)
end

# Update target sublattice (columns k+1:k+3) at row m from neighbours at rows n1, n2
# of the other sublattice (columns q+1:q+3).
@inline function _site!(X, m, n1, n2, k, q, dt, Jx, Jy, Jz, poly)
    @inbounds begin
        hx = Jx * (X[n1, q+1] + X[n2, q+1])
        hy = Jy * (X[n1, q+2] + X[n2, q+2])
        hz = Jz * (X[n1, q+3] + X[n2, q+3])
        a, b, c = _rot(X[m, k+1], X[m, k+2], X[m, k+3], hx, hy, hz, dt, poly)
        X[m, k+1] = a; X[m, k+2] = b; X[m, k+3] = c
    end
    return nothing
end

@inline function _site!(X, D, m, n1, n2, k, q, dt, Jx, Jy, Jz, poly)
    @inbounds begin
        hx = Jx * (X[n1, q+1] + X[n2, q+1]); gx = Jx * (D[n1, q+1] + D[n2, q+1])
        hy = Jy * (X[n1, q+2] + X[n2, q+2]); gy = Jy * (D[n1, q+2] + D[n2, q+2])
        hz = Jz * (X[n1, q+3] + X[n2, q+3]); gz = Jz * (D[n1, q+3] + D[n2, q+3])
        s1, s2, s3, u1, u2, u3 = _rot_tan(X[m, k+1], X[m, k+2], X[m, k+3],
                                          D[m, k+1], D[m, k+2], D[m, k+3],
                                          hx, hy, hz, gx, gy, gz, dt, poly)
        X[m, k+1] = s1; X[m, k+2] = s2; X[m, k+3] = s3
        D[m, k+1] = u1; D[m, k+2] = u2; D[m, k+3] = u3
    end
    return nothing
end

# Odd site 2m-1 has even neighbours 2m-2 (row m-1, wraps to M) and 2m (row m).
# Even site 2m has odd neighbours 2m-1 (row m) and 2m+1 (row m+1, wraps to 1).
function _sweep!(X, ::Nothing, parity::Int, dt::Float64, J::Couplings, poly::Val)
    M = size(X, 1); Jx, Jy, Jz = J.Jx, J.Jy, J.Jz
    if parity == 1
        _site!(X, 1, M, 1, 0, 3, dt, Jx, Jy, Jz, poly)
        @inbounds @simd for m in 2:M
            _site!(X, m, m - 1, m, 0, 3, dt, Jx, Jy, Jz, poly)
        end
    else
        @inbounds @simd for m in 1:M-1
            _site!(X, m, m, m + 1, 3, 0, dt, Jx, Jy, Jz, poly)
        end
        _site!(X, M, M, 1, 3, 0, dt, Jx, Jy, Jz, poly)
    end
    return nothing
end

function _sweep!(X, D::AbstractMatrix, parity::Int, dt::Float64, J::Couplings, poly::Val)
    M = size(X, 1); Jx, Jy, Jz = J.Jx, J.Jy, J.Jz
    if parity == 1
        _site!(X, D, 1, M, 1, 0, 3, dt, Jx, Jy, Jz, poly)
        @inbounds @simd for m in 2:M
            _site!(X, D, m, m - 1, m, 0, 3, dt, Jx, Jy, Jz, poly)
        end
    else
        @inbounds @simd for m in 1:M-1
            _site!(X, D, m, m, m + 1, 3, 0, dt, Jx, Jy, Jz, poly)
        end
        _site!(X, D, M, M, 1, 3, 0, dt, Jx, Jy, Jz, poly)
    end
    return nothing
end

"""
    sweep!(X, [D,] parity, dt, J)

Exactly precess every spin of sublattice `parity` (1 = odd sites, 2 = even sites)
for time dt about its frozen neighbour field (and propagate the tangent vector D).
"""
function sweep!(X::AbstractMatrix, D, parity::Int, dt::Float64, J::Couplings)
    hmax = 2 * max(abs(J.Jx), abs(J.Jy), abs(J.Jz))
    hmax * abs(dt) <= 1 ? _sweep!(X, D, parity, dt, J, Val(true)) :
                          _sweep!(X, D, parity, dt, J, Val(false))
    return X
end
sweep!(X::AbstractMatrix, parity::Int, dt::Float64, J::Couplings) = sweep!(X, nothing, parity, dt, J)

"""
    hamiltonian_step!(X, [D,] spl::Splitting, J::Couplings)

Evolve the SoA state X (and tangent vector D, if given) for time spl.τ.
"""
function hamiltonian_step!(X::AbstractMatrix, D, spl::Splitting, J::Couplings)
    for (p, d) in spl.ops
        sweep!(X, D, p, d, J)
    end
    return X
end
hamiltonian_step!(X::AbstractMatrix, spl::Splitting, J::Couplings) = hamiltonian_step!(X, nothing, spl, J)

# ---------------------------------------------------------------------------
# Windowed (row-range) versions for seed spreading on an open buffer: only rows
# mlo:mhi are updated; requires 2 ≤ mlo, mhi ≤ M - 1 (no wrap-around).
# ---------------------------------------------------------------------------
function _sweep_range!(X, ::Nothing, parity::Int, dt::Float64, J::Couplings, poly::Val, mlo::Int, mhi::Int)
    Jx, Jy, Jz = J.Jx, J.Jy, J.Jz
    if parity == 1
        @inbounds @simd for m in mlo:mhi
            _site!(X, m, m - 1, m, 0, 3, dt, Jx, Jy, Jz, poly)
        end
    else
        @inbounds @simd for m in mlo:mhi
            _site!(X, m, m, m + 1, 3, 0, dt, Jx, Jy, Jz, poly)
        end
    end
    return nothing
end
function _sweep_range!(X, D::AbstractMatrix, parity::Int, dt::Float64, J::Couplings, poly::Val, mlo::Int, mhi::Int)
    Jx, Jy, Jz = J.Jx, J.Jy, J.Jz
    if parity == 1
        @inbounds @simd for m in mlo:mhi
            _site!(X, D, m, m - 1, m, 0, 3, dt, Jx, Jy, Jz, poly)
        end
    else
        @inbounds @simd for m in mlo:mhi
            _site!(X, D, m, m, m + 1, 3, 0, dt, Jx, Jy, Jz, poly)
        end
    end
    return nothing
end

"""
    hamiltonian_step_range!(X, D, spl, J, mlo, mhi)

Hamiltonian step restricted to rows mlo:mhi of an open (non-periodic) SoA buffer.
Information moves at most one site per sweep, so if every site outside
[mlo + ⌈K/2⌉, mhi - ⌈K/2⌉] rows (K = number of sweeps) is on the helix, the result
equals evolving the whole (infinite) chain.
"""
function hamiltonian_step_range!(X::AbstractMatrix, D, spl::Splitting, J::Couplings, mlo::Int, mhi::Int)
    hmax = 2 * max(abs(J.Jx), abs(J.Jy), abs(J.Jz))
    for (p, d) in spl.ops
        hmax * abs(d) <= 1 ? _sweep_range!(X, D, p, d, J, Val(true), mlo, mhi) :
                             _sweep_range!(X, D, p, d, J, Val(false), mlo, mhi)
    end
    return X
end
