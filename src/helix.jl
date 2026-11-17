# ---------------------------------------------------------------------------
# Numerical (transverse) Lyapunov exponent of the helix fixed point under the
# controlled map: the tangent vector is propagated by the exact linearisation
# at S = S0 (the trajectory is reset to S0 every step), so this is the growth
# rate of infinitesimal perturbations of the target — the quantity whose sign
# decides linear stability, a_c^lin. Works for both models.
# ---------------------------------------------------------------------------

"""
    helix_is_fixed_point(N, model; L = 4lcm(2, N), tol = 1e-10) -> Bool

True if the helix is invariant under one Hamiltonian step for every coupling
choice the model can draw (for `:random_signs` only N = 2, 4 qualify).
"""
function helix_is_fixed_point(N::Int, model::Symbol; L::Int = 4lcm(2, N), tol::Real = 1e-10)
    X0 = to_soa(helix(L, N)); spl = Splitting(1.0, 0.1)
    Js = model === :random_signs ? [Couplings(sx, sy, 1.0) for sx in (-1.0, 1.0), sy in (-1.0, 1.0)] : [Couplings()]
    return all(J -> maximum(abs.(hamiltonian_step!(copy(X0), spl, J) .- X0)) < tol, Js)
end

"""
    lyapunov_helix(N, a; L, T = 3000, Tburn = 500, nruns = 4, model = :static,
                   τ = 1.0, dt = 0.1, order = 2, seed = 1) -> (λ, sem)

Leading exponent of the linearised controlled map at the helix. For the static
model this reproduces `lambda_fixed_point(N, a; L)` = log a + τ max_k λ(k, N);
for `:random_signs` (N = 2, 4) it gives the value the paper describes as
"slightly above log a". Default L: the smallest multiple of lcm(2, N) ≥ 120.
"""
function lyapunov_helix(N::Int, a::Real; L::Int = lcm(2, N) * cld(120, lcm(2, N)), T::Int = 3000,
                        Tburn::Int = 500, nruns::Int = 4, model::Symbol = :static, τ::Real = 1.0,
                        dt::Real = 0.1, order::Int = 2, seed::Int = 1)
    check_size(L, N)
    helix_is_fixed_point(N, model; L) ||
        throw(ArgumentError("helix (N = $N) is not a fixed point of the $model dynamics"))
    X0 = to_soa(helix(L, N)); spl = Splitting(τ, dt; order); J0 = Couplings()
    λs = zeros(nruns)
    Threads.@threads for r in 1:nruns
        rng = Xoshiro(hash((seed, N, L, model, :helix, r)))
        S0 = helix(L, N)
        Da = randn(rng, 3, L)
        for j in 1:L
            Da[:, j] .-= dot(Da[:, j], S0[:, j]) .* S0[:, j]
        end
        D = to_soa(Da); D ./= norm(D)
        X = similar(X0); acc = 0.0
        for t in 1:(Tburn + T)
            copyto!(X, X0)
            hamiltonian_step!(X, D, spl, _couplings(model, rng, J0))
            control!(X, D, X0, Float64(a))
            lt = _renorm_tangent!(D)
            t > Tburn && (acc += lt)
        end
        λs[r] = acc / T
    end
    return mean(λs), nruns > 1 ? std(λs) / sqrt(nruns) : NaN
end
