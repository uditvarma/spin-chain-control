# Validation suite:  julia --project=. -e 'using Pkg; Pkg.test()'
using HeisenbergControl
using HeisenbergControl: _init_tangent, _renorm_tangent!
using Test, Random, LinearAlgebra, Statistics
import ForwardDiff
using ForwardDiff: Dual, partials
struct TTag end
const DualF = Dual{TTag,Float64,1}

# Reference integrator: classic RK4 on the full ODE (3×L layout) with a tiny step.
function rhs!(dS, S, J::Couplings)
    L = size(S, 2)
    @inbounds for j in 1:L
        l = j == 1 ? L : j - 1; r = j == L ? 1 : j + 1
        hx = J.Jx * (S[1, l] + S[1, r]); hy = J.Jy * (S[2, l] + S[2, r]); hz = J.Jz * (S[3, l] + S[3, r])
        dS[1, j] = -(hy * S[3, j] - hz * S[2, j])
        dS[2, j] = -(hz * S[1, j] - hx * S[3, j])
        dS[3, j] = -(hx * S[2, j] - hy * S[1, j])
    end
end
function rk4!(S, τ, h, J)
    n = round(Int, τ / h); k1 = similar(S); k2 = similar(S); k3 = similar(S); k4 = similar(S)
    for _ in 1:n
        rhs!(k1, S, J); rhs!(k2, S .+ (h / 2) .* k1, J); rhs!(k3, S .+ (h / 2) .* k2, J)
        rhs!(k4, S .+ h .* k3, J)
        S .+= (h / 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
    end
    S
end
rnd(L, seed) = random_state!(zeros(3, L), Xoshiro(seed))
maxnormerr(S) = maximum(abs.(sqrt.(sum(abs2, S; dims = 1)) .- 1))
project!(D, S) = (for j in axes(S, 2); D[:, j] .-= dot(D[:, j], S[:, j]) .* S[:, j]; end; D)

@testset "HeisenbergControl" begin

@testset "layout round trip" begin
    S = rnd(10, 1)
    @test to_aos(to_soa(S)) == S
end

@testset "helix is a fixed point" begin
    for N in (2, 3, 4, 5, 6, 7, 10, 16)
        L = 2 * N * 4
        X0 = to_soa(helix(L, N)); X = copy(X0)
        for _ in 1:20
            hamiltonian_step!(X, Splitting(1.0, 0.1), Couplings())
            control!(X, X0, 0.2)        # a = 0.2 < ac_linear(N) for all N (ac_linear(3) ≈ 0.243)
        end
        @test maximum(abs.(X .- X0)) < 1e-12
    end
    @test_throws ArgumentError check_size(30, 4)   # 30 not a multiple of 4
    @test_throws ArgumentError check_size(15, 5)   # odd
end

@testset "exact norm and energy conservation" begin
    for J in (Couplings(), Couplings(-1, 1, 1), Couplings(1, -1, 1)), order in (2, 4), dt in (0.1, 0.4)
        S = rnd(256, 7); E0 = energy(S, J)
        X = to_soa(S); spl = Splitting(1.0, dt; order)
        for _ in 1:2000
            hamiltonian_step!(X, spl, J)
        end
        S = to_aos(X)
        @test maxnormerr(S) < 1e-12
        @test abs(energy(S, J) - E0) / 256 < 1e-11
    end
end

@testset "convergence to the exact flow (order 2 and 4)" begin
    S_init = rnd(64, 3)
    for J in (Couplings(), Couplings(1, -1, 1))
        Sref = rk4!(copy(S_init), 1.0, 1e-4, J)
        for (order, p_expected) in ((2, 2), (4, 4))
            dts = [0.2, 0.1, 0.05]
            errs = [maximum(abs.(to_aos(hamiltonian_step!(to_soa(S_init), Splitting(1.0, dt; order), J)) .- Sref))
                    for dt in dts]
            slopes = diff(log.(errs)) ./ diff(log.(dts))
            println("  J=$J order $order: errors $(round.(errs, sigdigits=3)), observed order $(round.(slopes, digits=2))")
            @test all(abs.(slopes .- p_expected) .< 0.3)
        end
    end
end

@testset "tangent dynamics: hand-coded = dual numbers = finite difference" begin
    for (N, a, J, dt, order) in ((4, 0.7, Couplings(), 0.1, 4), (5, 0.8, Couplings(-1, 1, 1), 0.25, 2),
                                 (4, 0.6, Couplings(), 0.9, 2))      # last: trig (non-polynomial) branch
        L = 60
        X0 = to_soa(helix(L, N)); spl = Splitting(1.0, dt; order)
        Sf = rnd(L, 11)
        Da = project!(randn(Xoshiro(3), 3, L), Sf)
        X = to_soa(Sf); D = to_soa(Da)
        Xd = to_soa([DualF(Sf[c, j], ForwardDiff.Partials((Da[c, j],))) for c in 1:3, j in 1:L])
        h = 1e-6
        Xp = to_soa(Sf .+ h .* Da); Xm = to_soa(Sf .- h .* Da)
        for _ in 1:5
            hamiltonian_step!(X, D, spl, J); control!(X, D, X0, a)
            for Y in (Xd, Xp, Xm)
                hamiltonian_step!(Y, spl, J); control!(Y, X0, a)
            end
        end
        fd = (Xp .- Xm) ./ (2h)
        dual = partials.(Xd, 1)
        @test norm(D .- dual) / norm(dual) < 1e-10
        @test norm(fd .- D) / norm(D) < 1e-5
        @test maximum(abs.(X .- ForwardDiff.value.(Xd))) < 1e-12
    end
end

@testset "linear theory" begin
    @test ac_linear(4) ≈ 1.0
    @test ac_linear(2) ≈ 1.0
    for N in (5, 6, 10, 20)
        q = 2π / N
        @test isapprox(ac_linear(N), exp(-sqrt(cos(q)) * (1 - cos(q))); rtol = 1e-6)
    end
    @test isapprox(ac_linear(3), exp(-sqrt(2)); rtol = 1e-6)
    # Tangent λ measured at the fixed point equals log(a) + max_k λ_k (N = 6, a in control phase)
    N = 6; L = 120; a = 0.55
    S0 = helix(L, N); X0 = to_soa(S0); spl = Splitting(1.0, 0.05)
    Sf = S0 .+ 1e-12 .* randn(Xoshiro(1), 3, L); Sf ./= sqrt.(sum(abs2, Sf; dims = 1))
    X = to_soa(Sf); D = to_soa(project!(randn(Xoshiro(5), 3, L), Sf))
    λs = Float64[]
    for t in 1:300
        hamiltonian_step!(X, D, spl, Couplings()); control!(X, D, X0, a)
        lt = _renorm_tangent!(D)
        t > 100 && push!(λs, lt)
    end
    println("  N=6 fixed-point λ: measured $(round(mean(λs), digits=5)), theory $(round(lambda_fixed_point(N, a; L), digits=5))")
    @test isapprox(mean(λs), lambda_fixed_point(N, a; L); atol = 2e-3)
end

@testset "brighton domains" begin
    L = 40; N = 4; S0 = helix(L, N); S = copy(S0)
    S[:, 10] .= [1.0, 0, 0]; S[:, 25] .= [1.0, 0, 0]; S[:, 26] .= [1.0, 0, 0]
    fB = zeros(2)
    m = measure(fB, to_soa(S), to_soa(S0), [1e-3, 10.0], 0.1)
    @test m.ndom == 2                        # two separated defects
    @test fB[1] ≈ 5 / 40                     # bonds (9,10),(10,11),(24,25),(25,26),(26,27)
    @test fB[2] == 0
    @test m.rho ≈ 3 / 40
    @test m.Sdiff ≈ sum(norm(S[:, j] - S0[:, j]) for j in 1:L) / L
    # defect straddling the wrap-around bond counts once
    S = copy(S0); S[:, 1] .= [1.0, 0, 0]
    @test measure(fB, to_soa(S), to_soa(S0), [1e-3], 0.1).ndom == 1
end

@testset "absorption shortcut is exact" begin
    p1 = RunParams(N = 4, L = 64, a = 0.45, T = 600, nsamples = 8)
    p2 = RunParams(N = 4, L = 64, a = 0.45, T = 600, nsamples = 8, absorb_tol = 0.0)
    r1 = run_point(p1); r2 = run_point(p2)
    @test all(r1["t_abs"] .> 0)
    @test maximum(abs.(r1["ss_lam"] .- r2["ss_lam"])) < 5e-3
    @test maximum(abs.(r1["ss_lam"] .- log(0.45))) < 5e-3
    @test maximum(r2["ss_Sdiff"]) < 1e-12
    @test r1["ts_Sdiff"][1:5, :] == r2["ts_Sdiff"][1:5, :]
    println("  control phase N=4 a=0.45: λ = $(round(mean(r1["ss_lam"]), digits=5)) vs log a = $(round(log(0.45), digits=5))")
end

@testset "spacetime reproduces run_point sample; λ on/off identical trajectories" begin
    p = RunParams(N = 4, L = 64, a = 0.66, T = 50, nsamples = 2, lyapunov = false, per_decade = 200)
    r = run_point(p)
    rl = run_point(RunParams(N = 4, L = 64, a = 0.66, T = 50, nsamples = 2, lyapunov = true, per_decade = 200))
    @test r["ts_Sdiff"] == rl["ts_Sdiff"]
    st = spacetime(4, 64, 0.66, 50; sample = 2, order = p.order, dt = p.dt)
    @test isapprox(mean(st.D[:, end]), r["ts_Sdiff"][end, 2]; rtol = 1e-5)
end

@testset "threads do not change results" begin
    p = RunParams(N = 5, L = 40, a = 0.75, T = 200, nsamples = 6)
    @test run_point(p)["ss_lam"] == run_point(p)["ss_lam"]
end

@testset "n_def and run-until-absorbed" begin
    L = 40; S0 = helix(L, 4); S = copy(S0); S[:, 7] .= [1.0, 0, 0]
    m = measure(zeros(1), to_soa(S), to_soa(S0), [1e-3], 0.1)
    @test m.ndef ≈ (1 - dot(S[:, 7], S0[:, 7])) / 2 / L
    p = RunParams(N = 4, L = 32, a = 0.6, T = 50, nsamples = 8, lyapunov = false, Tabs_max = 20_000)
    r = run_point(p)
    @test any(r["t_abs"] .> 50)                        # absorbed after T thanks to the extension
    @test r["ts_Sdiff"] == run_point(RunParams(N = 4, L = 32, a = 0.6, T = 50, nsamples = 8, lyapunov = false))["ts_Sdiff"]
end

@testset "numerical helix Lyapunov exponent" begin
    for (N, a) in ((4, 0.7), (6, 0.6), (10, 0.8))
        λ, _ = lyapunov_helix(N, a; T = 2000)
        @test isapprox(λ, lambda_fixed_point(N, a; L = lcm(2, N) * cld(120, lcm(2, N))); atol = 2e-3)
    end
    @test helix_is_fixed_point(4, :random_signs)
    @test !helix_is_fixed_point(5, :random_signs)
    λr, _ = lyapunov_helix(4, 0.75; model = :random_signs, T = 2000, nruns = 4)
    @test log(0.75) - 1e-3 < λr < log(0.75) + 5e-3   # "slightly above log a" (paper)
    # absorption shortcut now also active for random signs at N = 4
    r = run_point(RunParams(N = 4, L = 64, a = 0.5, T = 800, nsamples = 4, model = :random_signs))
    @test all(r["t_abs"] .> 0)
end

@testset "seed spreading: windowed = full ring" begin
    using HeisenbergControl: _dvec
    function ring_ref(p, r, L)
        rng = Xoshiro(hash((p.seed, p.N, p.model, :seed, r)))
        v = random_state!(zeros(3, 1), rng)
        S0 = helix(L, p.N); S = copy(S0)
        j0 = L ÷ 2 - 1
        while mod(j0 + 1, p.N) != 0 || iseven(j0); j0 += 1; end   # same helix phase as absolute site -1
        S[:, j0] .= v[:, 1]
        X = to_soa(S); X0 = to_soa(S0); spl = Splitting(p.τ, p.dt; order = p.order)
        tg = time_grid(p.T; per_decade = p.per_decade); A = Float64[]; Nact = Int[]; R2 = Float64[]
        for t in 1:p.T
            hamiltonian_step!(X, spl, Couplings()); control!(X, X0, p.a)
            a = 0.0; n = 0; r2 = 0.0
            for m in 1:size(X, 1), k in (0, 3)
                d = _dvec(X, X0, m, k); dn = sqrt(sum(abs2, d))
                dn <= p.epsw && (X[m, k+1:k+3] .= X0[m, k+1:k+3]; dn = 0.0)
                a += dn
                if dn > p.thr
                    n += 1; r2 += (2m - 1 + (k == 3) - j0)^2
                end
            end
            t in tg && (push!(A, a); push!(Nact, n); push!(R2, r2))
            a == 0 && break
        end
        A, Nact, R2
    end
    for (N, a, T) in ((4, 0.66, 40), (4, 0.61, 400), (5, 0.68, 40), (10, 0.86, 40), (4, 0.55, 200))
        p = SeedParams(N = N, a = a, T = T, nruns = 1, per_decade = 100, M0 = 16)
        for r in 1:3
            o = run_seed_single(p, r)
            A, Nact, R2 = ring_ref(p, r, 1600)
            k = length(A)
            @test maximum(abs.(o.tsA[1:k] .- A) ./ max.(A, 1e-300)) < 1e-6     # (tsA is Float32)
            @test o.tsN[1:k] == Nact
            @test o.tsR2[1:k] ≈ R2
        end
    end
    r = run_seed(SeedParams(N = 4, a = 0.62, T = 300, nruns = 32, ngroups = 4))
    @test r["P"][1] <= 1 && all(diff(r["P"]) .<= 0)
    @test run_seed(SeedParams(N = 4, a = 0.62, T = 300, nruns = 32, ngroups = 4))["N_t"] == r["N_t"]
end

@testset "analysis helpers" begin
    t = Float64.(time_grid(10_000)); y = 3 .* t .^ -0.16
    s = local_slopes(t, y; b = 4)
    @test all(x -> isnan(x) || abs(x + 0.16) < 1e-6, s)
    ts, dn = delta_n(t, y; b = 2.5)
    @test all(x -> isnan(x) || abs(x - 0.16) < 1e-6, dn)
    # synthetic collapse: y = L^-b G(t/L^z)
    G(x) = x^-0.3 / (1 + x)
    curves = [(L, t, L^-0.25 .* G.(t ./ L^1.6)) for L in (64, 128, 256)]
    z, b, _ = grid_collapse(curves; zs = 1.4:0.01:1.8, bs = 0.15:0.01:0.35, tmin = 1)
    @test abs(z - 1.6) < 0.011 && abs(b - 0.25) < 0.011
    v, e = quantile_boot([1.0, 2, 3, 4, Inf, Inf], 0.5)
    @test isfinite(v) && 3 <= v <= 4 && isfinite(e)
    @test isinf(quantile_boot([1.0, Inf, Inf, Inf], 0.5)[1])
end

end
