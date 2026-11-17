"""
    HeisenbergControl

Classical Heisenberg chain with a deterministic (or temporally random-sign)
Hamiltonian step alternating with a contractive control map onto a period-N helix

    S0_j = (0, cos(2πj/N), sin(2πj/N)).

One time step = Hamiltonian flow for duration τ, then
    S_j → [(1-a) S0_j + a S_j] / |(1-a) S0_j + a S_j|.

Order parameters (all computed per sample, then stored):
  * S_diff  = (1/L) Σ_j |S_j - S0_j|
  * f_B(ε)  = (1/L) Σ_j Θ(|d_{j+1} - d_j|² - ε),   d_j = S_j - S0_j   (brighton fraction)
  * n_dom   = number of contiguous brighton domains (for the first ε)
  * ρ       = (1/L) Σ_j Θ(|d_j| - ρ_ε)  (active-site density, the DP order parameter)
  * λ       = leading Lyapunov exponent from exact tangent-space dynamics

See README.md in the sandbox root for usage.
"""
module HeisenbergControl

using Random
using Statistics
using LinearAlgebra
using Printf
using TOML
using JLD2

include("model.jl")
include("integrator.jl")
include("observables.jl")
include("theory.jl")
include("simulate.jl")
include("helix.jl")
include("seed.jl")
include("spacetime.jl")
include("io.jl")
include("analysis.jl")

export Couplings, helix, random_state!, control!, check_size, to_soa, to_aos
export Splitting, hamiltonian_step!, sweep!
export Measurement, measure, energy
export lambda_linear, ac_linear, lambda_fixed_point
export RunParams, run_point, run_sample, time_grid
export SeedParams, run_seed, run_seed_single, hamiltonian_step_range!, helix_soa!
export spacetime
export save_point, load_point, point_filename, summarize, write_summary_csv, lambda_window, load_config, expand_config
export lambda_max_linear, classify, lyapunov_helix, helix_is_fixed_point, summarize_seed, seed_filename
export quantile_boot, absorption_quantile, interp_loglog, local_slopes, delta_n, mean_series
export seed_curves, seed_exponents, collapse_cost, grid_collapse, collapse_fit, fss_at

end # module
