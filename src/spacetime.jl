# ---------------------------------------------------------------------------
# Space–time data for heatmaps (single trajectory).
# ---------------------------------------------------------------------------

"""
    spacetime(N, L, a, T; model = :static, dt = 0.1, order = 2, stride = 1,
              seed = 1, sample = 1, τ = 1.0) -> (; t, D, G)

Single trajectory from the same initial state that sample `sample` uses in
`run_point` (same seed convention). Returns
  * `t`  times 0:stride:T
  * `D`  L × nt Float32, |d_j(t)| = |S_j - S0_j|           (activity)
  * `G`  L × nt Float32, |d_{j+1}(t) - d_j(t)|²             (brighton field; f_B = mean(G .> ε))
"""
function spacetime(N::Int, L::Int, a::Real, T::Int; model::Symbol = :static, dt::Real = 0.1,
                   order::Int = 2, stride::Int = 1, seed::Int = 1, sample::Int = 1, τ::Real = 1.0)
    check_size(L, N)
    p = RunParams(; N, L, a = Float64(a), T, model, τ = Float64(τ), dt = Float64(dt), order,
                  nsamples = 1, lyapunov = false, seed)
    S0 = helix(L, N); X0 = to_soa(S0)
    spl = Splitting(τ, dt; order)
    rng = _sample_rng(p, sample)
    X = to_soa(random_state!(Matrix{Float64}(undef, 3, L), rng))
    randn(rng, 3)                # keep RNG stream identical to run_sample
    ts = collect(0:stride:T)
    D = zeros(Float32, L, length(ts)); G = zeros(Float32, L, length(ts))
    function rec!(k)
        S = to_aos(X)
        @inbounds for j in 1:L
            r = j == L ? 1 : j + 1
            dx = S[1, j] - S0[1, j]; dy = S[2, j] - S0[2, j]; dz = S[3, j] - S0[3, j]
            ex = S[1, r] - S0[1, r]; ey = S[2, r] - S0[2, r]; ez = S[3, r] - S0[3, r]
            D[j, k] = sqrt(dx^2 + dy^2 + dz^2)
            G[j, k] = (ex - dx)^2 + (ey - dy)^2 + (ez - dz)^2
        end
    end
    rec!(1)
    k = 1
    J0 = Couplings()
    for t in 1:T
        J = _couplings(model, rng, J0)
        hamiltonian_step!(X, spl, J)
        control!(X, X0, Float64(a))
        if t % stride == 0
            k += 1
            rec!(k)
        end
    end
    return (; t = ts, D, G)
end
