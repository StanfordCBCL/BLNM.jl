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
dataset_file = "data/ToR-ORd_dynCI_epi.pkl"
# Path to the trained Branched Latent Neural Map.
BLNM_file = "NNs/BLNM_5-5-5-5-5_states1_disentanglement1_train-indices1:400_ToR-ORd.bson"
# Path to the figures folder.
figs_folder = "figs/"
# Indices of the testing set.
indices = Base.range(401, 500)
# Testing time step (in milliseconds).
dt = 0.1

# Load trained Branched Latent Neural Map.
@load BLNM_file NN

# Number of testing samples.
num_samples = Base.length(indices)

# Read dataset.
dataset = InOut.read_pkl(dataset_file)
num_simulations = Base.size(dataset["parameters"])[3]
# Times.
times = Base.range(dataset["times"][1], dataset["times"][end], step = dt)
tspan = (dataset["times"][1], dataset["times"][end])
num_times = Base.length(times)
times_adim = Base.zeros(1, num_times, num_simulations)
for idx in 1 : num_simulations
  times_adim[1, :, idx] = times / tspan[2]
end
# Parameters (cell-level conductances: [GNa, Gto, GNaL, GKr, GKs, GK1, GKb, Gncx, Pnak, PNab, PCab, GpCa, GClCa, GClb]).
params = Utils.interpolate_linear(dataset["parameters"], dataset["times"], times)
num_params = Base.size(params)[1]
params_min = Base.findmin(params, dims = (2, 3, 4))[1]
params_max = Base.findmax(params, dims = (2, 3, 4))[1]
params_adim = BLNM.adimensionalize(params, params_min, params_max)
# Outputs (action potential at limit cycle).
outputs = Utils.interpolate_linear(dataset["outputs"], dataset["times"], times)
num_outs = Base.size(outputs)[1]
out_min = Base.findmin(outputs, dims = (2, 3, 4))[1]
out_max = Base.findmax(outputs, dims = (2, 3, 4))[1]
outputs_adim = BLNM.adimensionalize(outputs, out_min, out_max)

# Splitting between training and testing set.
times_test = times_adim[:, :, indices]
params_test = params_adim[:, :, indices]
outputs_test = BLNM.dimensionalize(outputs_adim[:, :, indices], out_min, out_max)

# Generate all predictions.
history_adim = NN((times_test, params_test))
num_states = Base.size(history_adim)[1]
outputs_BLNM = BLNM.dimensionalize(history_adim[1 : num_outs, :, :], out_min, out_max)

# Compute testing loss function (MSE).
@time BLNM.loss_MSE_BLNM(NN, times_test, params_test, outputs_adim[:, :, indices])

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
  fig_file = figs_folder * "test_physical" * Base.string(idx_out) * "_indices" * Base.string(indices[1]) * ":" * Base.string(indices[end]) * "_ToR-ORd.pdf"
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
  fig_file = figs_folder * "test_latent" * Base.string(idx_state) * "_indices" * Base.string(indices[1]) * ":" * Base.string(indices[end]) * "_ToR-ORd.pdf"
  Plots.savefig(fig_file)
end