# ---------------------------------------------------------------------------
# State layout, target helix, initial conditions, control map.
#
# Internally a chain of L spins is stored "struct-of-arrays" by sublattice as an
# M×6 matrix X (M = L/2), each column contiguous:
#     X[m, 1:3] = (Sx, Sy, Sz) of odd  site j = 2m-1
#     X[m, 4:6] = (Sx, Sy, Sz) of even site j = 2m
# so every sublattice sweep is a unit-stride loop that SIMD-vectorises.
# `to_soa` / `to_aos` convert from/to the conventional 3×L matrix.
# All kernels are generic in the element type (Float64, or dual numbers in tests).
# ---------------------------------------------------------------------------

"""
    Couplings(Jx, Jy, Jz)

Exchange couplings for one Hamiltonian step, H = Σ_j Jx Sx_j Sx_{j+1} + Jy … + Jz ….
"""
struct Couplings
    Jx::Float64
    Jy::Float64
    Jz::Float64
end
Couplings() = Couplings(1.0, 1.0, 1.0)

"""
    check_size(L, N)

L must be even (sublattice splitting) and a multiple of N (helix commensurate
with periodic boundary conditions; otherwise the target is not a fixed point).
"""
function check_size(L::Int, N::Int)
    (iseven(L) && L >= 4) || throw(ArgumentError("L = $L must be even and ≥ 4 (sublattice integrator)"))
    L % N == 0 || throw(ArgumentError("L = $L must be a multiple of N = $N (helix must close on the ring)"))
    return nothing
end

"""
    to_soa(S::AbstractMatrix) -> M×6 matrix      (S is 3×L, L even)
"""
function to_soa(S::AbstractMatrix{T}) where {T}
    L = size(S, 2); M = L ÷ 2
    X = Matrix{T}(undef, M, 6)
    @inbounds for m in 1:M, c in 1:3
        X[m, c] = S[c, 2m - 1]
        X[m, c + 3] = S[c, 2m]
    end
    return X
end

"""
    to_aos(X) -> 3×L matrix
"""
function to_aos(X::AbstractMatrix{T}) where {T}
    M = size(X, 1)
    S = Matrix{T}(undef, 3, 2M)
    @inbounds for m in 1:M, c in 1:3
        S[c, 2m - 1] = X[m, c]
        S[c, 2m] = X[m, c + 3]
    end
    return S
end

"""
    helix_soa!(X0, N, off)

Fill the M×6 SoA matrix with the helix at absolute sites j = off + 1, …, off + 2M
(off even keeps the sublattice assignment).
"""
function helix_soa!(X0::AbstractMatrix{Float64}, N::Int, off::Int)
    @inbounds for m in axes(X0, 1), (k, j) in ((0, off + 2m - 1), (3, off + 2m))
        s, c = sincospi(2 * mod(j, N) / N)
        X0[m, k+1] = 0.0; X0[m, k+2] = c; X0[m, k+3] = s
    end
    return X0
end

"""
    helix(L, N) -> 3×L Matrix

Target configuration S0_j = (0, cos(2πj/N), sin(2πj/N)).
"""
function helix(L::Int, N::Int)
    S0 = zeros(Float64, 3, L)
    @inbounds for j in 1:L
        s, c = sincospi(2 * mod(j, N) / N)   # same values as helix_soa! (seed mode)
        S0[2, j] = c
        S0[3, j] = s
    end
    return S0
end

"""
    random_state!(S, rng)

Fill the 3×L matrix S with independent spins uniform on the unit sphere.
"""
function random_state!(S::AbstractMatrix{Float64}, rng::AbstractRNG)
    @inbounds for j in axes(S, 2)
        z = 2rand(rng) - 1
        φ = 2π * rand(rng)
        r = sqrt(1 - z^2)
        sφ, cφ = sincos(φ)
        S[1, j] = r * cφ
        S[2, j] = r * sφ
        S[3, j] = z
    end
    return S
end

"""
    control!(X, X0, a)

Control map on every site (SoA layout): S ← [(1-a) S0 + a S] / |…|.
"""
function control!(X::AbstractMatrix{T}, X0::AbstractMatrix{Float64}, a::Float64,
                  rows::UnitRange{Int} = 1:size(X, 1)) where {T}
    b = 1 - a
    for k in (0, 3)
        @inbounds @simd for m in rows
            v1 = b * X0[m, k+1] + a * X[m, k+1]
            v2 = b * X0[m, k+2] + a * X[m, k+2]
            v3 = b * X0[m, k+3] + a * X[m, k+3]
            inv = 1 / sqrt(v1 * v1 + v2 * v2 + v3 * v3)   # v = 0 only on a measure-zero set
            X[m, k+1] = v1 * inv
            X[m, k+2] = v2 * inv
            X[m, k+3] = v3 * inv
        end
    end
    return X
end

"""
    control!(X, D, X0, a)

Control map on the trajectory X and its exact linearisation on the tangent vector D:
u = v/|v|, v = (1-a)S0 + aS  ⇒  δu = (a δS - u (u·a δS)) / |v|.
"""
function control!(X::Matrix{Float64}, D::Matrix{Float64}, X0::AbstractMatrix{Float64}, a::Float64,
                  rows::UnitRange{Int} = 1:size(X, 1))
    b = 1 - a
    for k in (0, 3)
        @inbounds @simd for m in rows
            v1 = b * X0[m, k+1] + a * X[m, k+1]
            v2 = b * X0[m, k+2] + a * X[m, k+2]
            v3 = b * X0[m, k+3] + a * X[m, k+3]
            inv = 1 / sqrt(v1 * v1 + v2 * v2 + v3 * v3)
            u1 = v1 * inv; u2 = v2 * inv; u3 = v3 * inv
            w1 = a * D[m, k+1]; w2 = a * D[m, k+2]; w3 = a * D[m, k+3]
            uw = u1 * w1 + u2 * w2 + u3 * w3
            X[m, k+1] = u1; X[m, k+2] = u2; X[m, k+3] = u3
            D[m, k+1] = (w1 - u1 * uw) * inv
            D[m, k+2] = (w2 - u2 * uw) * inv
            D[m, k+3] = (w3 - u3 * uw) * inv
        end
    end
    return X
end
control!(X::Matrix{Float64}, ::Nothing, X0::AbstractMatrix{Float64}, a::Float64,
         rows::UnitRange{Int} = 1:size(X, 1)) = control!(X, X0, a, rows)
