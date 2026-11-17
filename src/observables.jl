# ---------------------------------------------------------------------------
# Order parameters, all in a single O(L) pass over the chain (SoA layout).
# ---------------------------------------------------------------------------

_val(x::Real) = x                       # hook for generic (e.g. dual-number) states

"""
    Measurement

Result of `measure`: S_diff, defect density n_def = (1/L) Σ_j (1 - S_j·S0_j)/2 = (1/L) Σ_j |d_j|²/4, active density ρ, number of brighton domains for the
first ε, max_j |d_j|, and (in the caller's buffer) f_B(ε) for every ε.
"""
struct Measurement
    Sdiff::Float64
    ndef::Float64
    rho::Float64
    ndom::Int
    maxd::Float64
end

@inline _dvec(X, X0, m, k) = @inbounds (_val(X[m, k+1]) - X0[m, k+1],
                                        _val(X[m, k+2]) - X0[m, k+2],
                                        _val(X[m, k+3]) - X0[m, k+3])
@inline _g(d, n) = (n[1] - d[1])^2 + (n[2] - d[2])^2 + (n[3] - d[3])^2

"""
    measure(fB, X, X0, epsB, rho_eps) -> Measurement

`X`, `X0` in SoA layout. `fB` (length = length(epsB)) is overwritten with
    f_B(ε) = (1/L) Σ_j Θ(|d_{j+1} - d_j|² - ε),  d_j = S_j - S0_j  (periodic),
i.e. the fraction of bonds across which the error field jumps by more than √ε.
`ndom` = number of contiguous runs of flagged bonds for ε = epsB[1] (0 if none or all).
"""
function measure(fB::Vector{Float64}, X::AbstractMatrix, X0::AbstractMatrix{Float64},
                 epsB::Vector{Float64}, rho_eps::Float64)
    M = size(X, 1); L = 2M
    ne = length(epsB)
    fill!(fB, 0.0)
    ε1 = epsB[1]
    sumd = 0.0; sumd2 = 0.0; nact = 0; maxd = 0.0; ndom = 0
    dfirst = _dvec(X, X0, 1, 0)
    prevflag = _g(_dvec(X, X0, M, 3), dfirst) > ε1      # bond (L, 1)
    d = dfirst
    @inbounds for m in 1:M, half in 0:1
        # current site: (m, 3half); next site: (m, 3) if odd, else (m+1, 0) / wrap
        nxt = half == 0 ? _dvec(X, X0, m, 3) : (m < M ? _dvec(X, X0, m + 1, 0) : dfirst)
        dn2 = d[1]^2 + d[2]^2 + d[3]^2
        dn = sqrt(dn2)
        sumd += dn
        sumd2 += dn2
        nact += dn > rho_eps
        maxd = max(maxd, dn)
        g = _g(d, nxt)
        for k in 1:ne
            fB[k] += g > epsB[k]
        end
        f1 = g > ε1
        ndom += f1 & !prevflag
        prevflag = f1
        d = nxt
    end
    fB ./= L
    return Measurement(sumd / L, sumd2 / (4L), nact / L, ndom, maxd)
end

"""
    energy(S, J) = Σ_j (Jx Sx_j Sx_{j+1} + Jy … + Jz …)   for a 3×L matrix S (periodic)
"""
function energy(S::AbstractMatrix, J::Couplings = Couplings())
    L = size(S, 2); E = 0.0
    @inbounds for j in 1:L
        r = j == L ? 1 : j + 1
        E += J.Jx * _val(S[1, j]) * _val(S[1, r]) + J.Jy * _val(S[2, j]) * _val(S[2, r]) +
             J.Jz * _val(S[3, j]) * _val(S[3, r])
    end
    return E
end
