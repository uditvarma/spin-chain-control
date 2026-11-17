# ---------------------------------------------------------------------------
# Linear stability of the period-N helix (isotropic J = 1), from the notes
# "Controlling the chaotic classical spin chain", Sec. 6.1.3:
#   λ(k,N)² = -4 cos(q) (cos k - cos q)(cos k - 1),   q = 2π/N.
# Growth rate of mode k is sqrt(max(λ², 0)). With control the fixed point is
# stable iff a · exp(τ max_k λ) < 1.
# ---------------------------------------------------------------------------

"""
    lambda_linear(k, N) -> real growth rate of mode k (0 if oscillatory/marginal)
"""
function lambda_linear(k::Real, N::Int)
    cq = cospi(2 / N)
    λ2 = -4 * cq * (cos(k) - cq) * (cos(k) - 1)
    return λ2 > 0 ? sqrt(λ2) : 0.0
end

"""
    lambda_max_linear(N; L = nothing)

max_k λ(k, N): over the lattice momenta k = 2πm/L if L is given, else over a
fine continuum grid (L → ∞).
"""
function lambda_max_linear(N::Int; L::Union{Nothing,Int} = nothing)
    ks = L === nothing ? range(0, π; length = 200_001) : (2π * m / L for m in 0:L÷2)
    return maximum(k -> lambda_linear(k, N), ks)
end

"""
    ac_linear(N; τ = 1, L = nothing) = exp(-τ max_k λ(k, N))

Linear-stability threshold of the helix. For N = 2, 4, ∞ all modes are marginal
and this returns 1 (the actual transition is then set by nonlinear effects).
Closed form for N > 4: exp[-√(cos q) (1 - cos q)].
"""
ac_linear(N::Int; τ::Real = 1.0, L = nothing) = exp(-τ * lambda_max_linear(N; L = L))

"""
    lambda_fixed_point(N, a; τ = 1, L = nothing)

Leading Lyapunov exponent of the controlled map *at* the fixed point (static,
isotropic model): log(a) + τ max_k λ(k, N). Used to fill in λ after a sample
has been absorbed.
"""
lambda_fixed_point(N::Int, a::Real; τ::Real = 1.0, L = nothing) =
    log(a) + τ * lambda_max_linear(N; L = L)
