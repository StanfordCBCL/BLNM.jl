# External packages.
using Plots
using BSON: @load

# I/O features.
include("./src/InOut.jl")
using .InOut
# Utilities.
include("./src/BLNM.jl")
using .BLNM

### User-defined parameters ###
# Path to dataset with the ECGs from electrophysiology simulations.
dataset_file = "data/ECGs_HLHS.pkl"
# Path to the trained Branched Latent Neural Map.
BLNM_file = "NNs/BLNM_19-19-19-19-19-19-19_states10_disentanglement2_train-indices1:150_HLHS_ECGs.bson"
# Path to the figures folder.
figs_folder = "figs/"
# Indices of the testing set.
indices = Base.range(151, 200)
# Testing time step (in milliseconds).
dt = 5.0

# Load trained Branched Latent Neural Map.
@load BLNM_file NN

# Number of testing samples.
num_samples = Base.length(indices)

# Read dataset.
dataset = InOut.read_pkl(dataset_file)
num_simulations = Base.size(dataset["parameters"])[3]
# Import times in the original range [0, 600] milliseconds.
times = Base.range(dataset["times"][1], dataset["times"][end], step = dt)
tspan = (dataset["times"][1], dataset["times"][end])
num_times = Base.length(times)
times_adim = Base.zeros(1, num_times, num_simulations)
for idx in 1 : num_simulations
  times_adim[1, :, idx] = times / tspan[2]
end
# Import parameters in the adimensional range [-1, 1].
params_adim = BLNM.interpolate(dataset["parameters"], dataset["times"], times)
num_params = Base.size(params_adim)[1]
# Maximum and minimum of [tLVstim, GNa, GCaL, GKr, Dpurk, Dani, Diso] in the original range.
params_min = Array{Float64, 3}(undef, num_params, 1, 1) .= [0.1950 , 7.5400 , 2.0365e-5, 0.0771, 1.0027, 0.0084, 0.0028]
params_max = Array{Float64, 3}(undef, num_params, 1, 1) .= [99.3059, 29.4130, 7.9358e-5, 0.3057, 3.4826, 0.0331, 0.0110]
# Import outputs in the adimensional range [-1, 1].
outputs_adim = BLNM.interpolate(dataset["outputs"], dataset["times"], times)
num_outs = Base.size(outputs_adim)[1]
# Maximum and minimum of [V1, V2, V3, V4, V5, V6, RA, LA, F] in the original range.
out_min = Array{Float64, 3}(undef, num_outs, 1, 1) .= [-4.3731, -4.2820, -3.0456, -3.1056, -4.6857, -2.0958, -1.2211, -0.9557, -0.7815]
out_max = Array{Float64, 3}(undef, num_outs, 1, 1) .= [2.2964 , 3.8095 , 7.9580 , 5.8659 , 4.7494 , 2.9059 , 0.6399 , 0.3444 , 1.8498 ]

# Splitting between training and testing set.
times_test = times_adim[:, :, indices]
params_test = params_adim[:, :, indices]
outputs_test = BLNM.dimensionalize(outputs_adim[:, :, indices], out_min, out_max)

# Generate all predictions.
history_adim = NN((times_test, params_test))
num_states = Base.size(history_adim)[1]
outputs_BLNM = BLNM.dimensionalize(history_adim[1 : num_outs, :, :], out_min, out_max)

# Compute testing loss function (MSE).
BLNM.loss_MSE_BLNM(NN, times_test, params_test, outputs_adim[:, :, indices])

# Plot all physical variables over time for the testing set.
for idx_out in 1 : num_outs
  plots = []
  for idx in range(1, num_samples)
    p = Plots.plot(times, outputs_BLNM[idx_out, :, idx], label = "Predicted")
    Plots.plot!(times, outputs_test[idx_out, :, idx], label = "True")
    Base.push!(plots, p)
  end
  Plots.plot(plots...)
  Plots.plot!(size = (1800, 1800))
  fig_file = figs_folder * "test_physical" * Base.string(idx_out) * "_indices" * Base.string(indices[1]) * ":" * Base.string(indices[end]) * "_HLHS_ECGs.pdf"
  Plots.savefig(fig_file)
end

# Plot all latent variables over time for the testing set.
for idx_state in (num_outs + 1) : num_states
  plots = []
  for idx in range(1, num_samples)
    p = Plots.plot(times, history_adim[idx_state, :, idx], label = "Predicted")
    Base.push!(plots, p)
  end
  Plots.plot(plots...)
  Plots.plot!(size = (1800, 1800))
  fig_file = figs_folder * "test_latent" * Base.string(idx_state) * "_indices" * Base.string(indices[1]) * ":" * Base.string(indices[end]) * "_HLHS_ECGs.pdf"
  Plots.savefig(fig_file)
end