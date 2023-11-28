# External libraries.
using Flux
using Plots, StatsPlots
using LaTeXStrings
using Turing
using Optim, Optimization, OptimizationOptimJL, OptimizationBBO
using GlobalSensitivity
using Copulas
using Statistics, StatsBase
using Distributions
using LinearAlgebra
using Random
using BSON: @load

# I/O features.
include("./src/InOut.jl")
using .InOut
# Utilities.
include("./src/BLNM.jl")
using .BLNM
include("./src/Utils.jl")
using .Utils

# Set random seed for reproducibility.
Random.seed!(1)
Random.rand(1)
rng = Random.MersenneTwister(1)

### User-defined parameters ###
# Path to dataset with the full-order model numerical simulations.
dataset_file = "data/ECGs_HLHS.pkl"
# Path to the trained Branched Latent Neural Map.
BLNM_file = "NNs/BLNM_19-19-19-19-19-19-19_states10_disentanglement2_train-indices1:150_HLHS_ECGs.bson"
# Path to the figures folder.
figs_folder = "figs/"
# Path to the patient-specific 12-lead electrocardiogram.
clinical_ECGs = "data/ECGs_HLHS_clinical/" .* ["V1.csv", "V2.csv", "V3.csv" , "V4.csv" , "V5.csv" , "V6.csv",
                                               "I.csv" , "II.csv", "III.csv", "aVR.csv", "aVL.csv", "aVF.csv"]
num_leads = length(clinical_ECGs)
# Time step (in milliseconds).
dt = 5.0
# Settings for Nelder-Mead.
num_trials = 100
# Flag to compute Shapley values in sensitivity analysis.
compute_shapley = true
num_permutations = 2000
num_boostrapped = 500
num_outer = 50
num_inner = 3
# Settings for Hamiltonian Monte Carlo.
num_samples = 1000
num_warmup = 1000
num_chains = 4
acceptance_rate = 0.9
# Flag to compute Maximum a Posteriori estimation to initialize Hamiltonian Monte Carlo.
compute_MAP = false
# Flag to compute Maximum Likelihood Estimation to initialize Hamiltonian Monte Carlo.
compute_MLE = false
# Threshold for marginal prior distributions in sensitivity analysis and Hamiltonian Monte Carlo.
threshold = 0.2

# Load trained Branched Latent Neural Map.
@load BLNM_file NN

# Read dataset.
dataset = InOut.read_pkl(dataset_file)
num_simulations = Base.size(dataset["parameters"])[3]
# Import times in the original range [0, 600] milliseconds.
times = Base.range(dataset["times"][1], dataset["times"][end], step = dt)
tspan = (dataset["times"][1], dataset["times"][end])
num_times = Base.length(times)
times_adim = Base.zeros(1, num_times, 1)
times_adim[1, :, 1] = times / tspan[2]
times_adim = times_adim[:, :, 1]
# Import parameters in the adimensional range [-1, 1].
params_adim = Utils.interpolate_linear(dataset["parameters"], dataset["times"], times)
num_params = Base.size(params_adim)[1]
# Maximum and minimum of [tLVstim, GNa, GCaL, GKr, Dpurk, Dani, Diso] in the original range.
params_min = Array{Float64, 3}(undef, num_params, 1, 1) .= [0.1950 , 7.5400 , 2.0365e-5, 0.0771, 1.0027, 0.0084, 0.0028]
params_max = Array{Float64, 3}(undef, num_params, 1, 1) .= [99.3059, 29.4130, 7.9358e-5, 0.3057, 3.4826, 0.0331, 0.0110]
params = BLNM.dimensionalize(params_adim, params_min, params_max)
# Import outputs in the adimensional range [-1, 1].
outputs_adim = Utils.interpolate_linear(dataset["outputs"], dataset["times"], times)
num_outs = Base.size(outputs_adim)[1]
# Maximum and minimum of [V1, V2, V3, V4, V5, V6, RA, LA, F] in the original range.
out_min = Array{Float64, 3}(undef, num_outs, 1, 1) .= [-4.3731, -4.2820, -3.0456, -3.1056, -4.6857, -2.0958, -1.2211, -0.9557, -0.7815]
out_max = Array{Float64, 3}(undef, num_outs, 1, 1) .= [2.2964 , 3.8095 , 7.9580 , 5.8659 , 4.7494 , 2.9059 , 0.6399 , 0.3444 , 1.8498 ]
outputs = BLNM.dimensionalize(outputs_adim, out_min, out_max)
outputs = Utils.add_bipolar_and_augmented_limb_leads(outputs)
outputs_adim = Utils.add_bipolar_and_augmented_limb_leads(outputs_adim)

# Compilation (first run).
@time NN((times_adim, params_adim[:, :, 1]))

# Import clinical 12-lead electrocardiogram.
observations = Base.zeros(num_leads, num_times)
for idx in 1 : num_leads
    ECG_lead = InOut.read_csv(clinical_ECGs[idx], false)[:, :]
    observations[idx, :] = Utils.interpolate_spline(ECG_lead[:, 2], ECG_lead[:, 1], times)
end

# Find initial guess of model parameters for Hamiltonian Monte Carlo.
function NN_optim(p, dummy)
    p[7] = p[6] # Anisotropic and isotropic conductivities encode the same adimensional input.
    p_repeated = p;
    for index in 1 : (num_times - 1)
        p_repeated = Base.cat(p_repeated, p, dims = 2)
    end
    NN_out = NN((times_adim, p_repeated))[1 : num_outs, :, 1]
    NN_out_dim = BLNM.dimensionalize(NN_out, out_min[:, :, 1], out_max[:, :, 1])
    NN_out_dim = Utils.add_bipolar_and_augmented_limb_leads(NN_out_dim)

    return Flux.mse(NN_out_dim, observations) 
end
p_found = Base.zeros(num_params)
left = -Base.ones(num_params)
right = Base.ones(num_params)
@time for index in 1 : num_trials
    p_0 = (right .- left) .* rand!(rng, zeros(num_params)) + left
    NN_optim_AD = OptimizationFunction(NN_optim, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(NN_optim_AD, p_0, lb = Base.repeat([-1], num_params), ub = Base.repeat([1], num_params))
    # Two possible alternatives: Optim.BFGS(), OptimizationBBO.BBO_adaptive_de_rand_1_bin()
    p_trial = OptimizationOptimJL.solve(prob, Optim.NelderMead())
    p_found = p_found .+ p_trial
end
p_found = p_found ./ num_trials

# Compute Shapley effects starting from the initial guess given by Nelder-Mead.
if compute_shapley
    function NN_optim(p)
        p[7] = p[6] # Anisotropic and isotropic conductivities encode the same adimensional input.
        p_repeated = p;
        for index in 1 : num_params
            if p[index] > 1.0
                p[index] = 1.0
            elseif p[index] < -1.0
                p[index] = -1.0
            end
        end
        for index in 1 : (num_times - 1)
            p_repeated = Base.cat(p_repeated, p, dims = 2)
        end
        NN_out = NN((times_adim, p_repeated))[1 : num_outs, :, 1]
        NN_out_dim = BLNM.dimensionalize(NN_out, out_min[:, :, 1], out_max[:, :, 1])
        NN_out_dim = Utils.add_bipolar_and_augmented_limb_leads(NN_out_dim)
        return Flux.mse(NN_out_dim, observations) 
    end

    mu = p_found
    Covmat = Matrix(1.0 * I, num_params, num_params)
    marginals = [Distributions.Normal(mu[i], threshold) for i in 1 : num_params]
    copula = Copulas.GaussianCopula(Covmat)
    input_distribution = Copulas.SklarDist(copula, marginals)

    @time shapley_effects = GlobalSensitivity.gsa(NN_optim, Shapley(;n_perms = num_permutations,
                                                                     n_var = num_boostrapped,
                                                                     n_outer = num_outer,
                                                                     n_inner = num_inner),
                                                                     input_distribution, batch = false)
end

# Define model for Bayesian parameter estimation.
p_min = Base.zeros(num_params - 1)
p_max = Base.zeros(num_params - 1)
for index in 1 : (num_params - 1)
    if (p_found[index] - threshold < -1.0 && p_found[index] + threshold > 1.0)
        p_min[index] = -1.0
        p_max[index] = 1.0
    elseif (p_found[index] - threshold < -1.0)
        p_min[index] = -1.0
        p_max[index] = p_found[index] + threshold
    elseif (p_found[index] + threshold > 1.0)
        p_min[index] = p_found[index] - threshold
        p_max[index] = 1.0
    else
        p_min[index] = p_found[index] - threshold
        p_max[index] = p_found[index] + threshold
    end
end
@model function fitECGs(x, y, p_min, p_max, σ_meas = 0.1)
    # Prior distributions.
    tLV ~ Uniform(p_min[1], p_max[1])
    GNa ~ Uniform(p_min[2], p_max[2])
    GCaL ~ Uniform(p_min[3], p_max[3])
    GKr ~ Uniform(p_min[4], p_max[4])
    DPurk ~ Uniform(p_min[5], p_max[5])
    Dani ~ Uniform(p_min[6], p_max[6]) # Anisotropic and isotropic conductivities encode the same adimensional input.
    l_GP ~ Uniform(0.01, 1.0)
    σ_GP ~ Uniform(0.01, 1.0)

    # ECGs mapping via Branched Latent Neural Map.
    p = [tLV, GNa, GCaL, GKr, DPurk, Dani, Dani]
    p = transpose(repeat(transpose(p), num_times))[:, :]
    predicted = NN((times_adim, p))[1 : num_outs, :]
    predicted_dim = BLNM.dimensionalize(predicted, out_min[:, :, 1], out_max[:, :, 1])

    # Observations (measument and surrogate modeling errors for each time point).
    predicted_dim = Utils.add_bipolar_and_augmented_limb_leads(predicted_dim)
    for index in 1 : length(num_leads)
        y[index, :] ~ MvNormal(predicted_dim[index, :], σ_meas * I + Utils.EQkernel(x, x, l_GP, σ_GP))
    end
end
model = fitECGs(times_adim[1, :], observations, p_min, p_max)

# Compute maximum likelihood and/or maximum a posteriori estimation.
if compute_MAP
    map_est = optimize(model, Turing.MAP(), Optim.NelderMead(), Optim.Options(iterations = 10000, allow_f_increases = true))
elseif compute_MLE
    mle_est = optimize(model, Turing.MLE(), Optim.NelderMead(), Optim.Options(iterations = 10000, allow_f_increases = true))
end

# Run Hamiltonian Monte Carlo with No-U-Turn Sampler for inverse uncertainty quantification.
if compute_MAP
    @time chain = Turing.sample(model, Turing.NUTS(num_warmup, acceptance_rate), Turing.MCMCThreads(), num_samples, num_chains, init_theta = map_est.values.array; progress = true)
elseif compute_MLE
    @time chain = Turing.sample(model, Turing.NUTS(num_warmup, acceptance_rate), Turing.MCMCThreads(), num_samples, num_chains, init_theta = mle_est.values.array; progress = true)
else
    @time chain = Turing.sample(model, Turing.NUTS(num_warmup, acceptance_rate), Turing.MCMCThreads(), num_samples, num_chains; progress = true)
end
# Plot density estimation for all model parameters and chains.
StatsPlots.plot(chain)