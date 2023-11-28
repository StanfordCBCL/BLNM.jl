# External packages.
using Flux
using Zygote
using Optim, Optimization, OptimizationOptimJL, Optimisers, Statistics
using Plots
using Random
using BSON: @save
using Plots
using LaTeXStrings

# I/O features.
include("./src/InOut.jl")
using .InOut
# Branched Latent Neural Maps.
include("./src/BLNM.jl")
using .BLNM
# Utilities.
include("./src/Utils.jl")
using .Utils

# Set random seed for reproducibility.
seed = 1
Random.seed!(seed)
Random.rand(seed)

### User-defined parameters ###
# Path to dataset with the ECGs from electrophysiology simulations.
dataset_file = "data/ToR-ORd_dynCI_epi.pkl"
# Output folder for the trained Branched Latent Neural Map.
output_folder = "NNs/"
# Training indices from the dataset.
train_indices = range(1, 400)
# Test indices from the dataset.
test_indices = range(401, 500)
# Number of neurons for each hidden layer.
neurons_per_layer = [5, 5, 5, 5, 5]
# Number of states (physical and latent variables).
num_states = 1
# Disentanglement level.
disentanglement_level = 1
# Number of epochs for (preliminary) Adam optimization.
num_epochs_Adam = 1000
# List of (deterministic) optimizers for each optimization phase, following Adam.
# Available options: [NelderMead(), GradientDescent(), ConjugateGradient(), BFGS(), LBFGS(), Newton()]
optimizers = [BFGS()]
# List of optimizers labels for each optimization phase, following Adam (needed for printing only).
labels = ["BFGS"]
# Maximum number of epochs for each optimization phase, following Adam.
num_epochs = [2000]
# Training wall time (in seconds) for each optimization phase, following Adam.
max_time = 100000
# Output frequency for the optimizer callback.
out_freq = 100
# Time step (in milliseconds).
dt = 0.1

# Training/Testing samples.
num_train = Base.length(train_indices)
num_test = Base.length(test_indices)
num_samples = num_train + num_test

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
params_train = params_adim[:, :, train_indices]
times_train = times_adim[:, :, train_indices]
outputs_train = outputs_adim[:, :, train_indices]
params_test = params_adim[:, :, test_indices]
times_test = times_adim[:, :, test_indices]
outputs_test = outputs_adim[:, :, test_indices]

# Define the Branched Latent Neural Map for Float64 operations.
NN = BLNM.build_BLNM(neurons_per_layer, 1, num_params, num_states, disentanglement_level)

# Compile Branched Latent Neural Map (first run).
@time NN((times_train, params_train))
# Compile MSE loss function (first run).
@time BLNM.loss_MSE_BLNM(NN, times_train, params_train, outputs_train)

# Reset and collect neural network tunable parameters.
Flux.loadparams!(NN, map(p -> p .= randn.(), Flux.params(NN)))
ps = Flux.params(NN)

# Collect history of training and testing loss.
epochs_history = []
training_history = []
testing_history = []

# Preliminary Adam optimization.
println("***** ADAM OPTIMIZER FOR ", num_epochs_Adam, " EPOCHS ", " *****")
function Adam_train!(ps, opt)
  local training_loss
  ps = Params(ps)
  gs = gradient(ps) do
    training_loss = BLNM.loss_MSE_BLNM(NN, times_train, params_train, outputs_train)

    return training_loss
  end
  Flux.update!(opt, ps, gs)
end
opt = Flux.Optimiser(Flux.Optimise.WeightDecay(1.0f-4), Flux.Adam(1.0f-3))
for idx_e in 1 : num_epochs_Adam
  Adam_train!(ps, opt)
  if (mod(idx_e, out_freq) == 0)
    training_loss = BLNM.loss_MSE_BLNM(NN, times_train, params_train, outputs_train)
    testing_loss = BLNM.loss_MSE_BLNM(NN, times_test, params_test, outputs_test)

    Base.push!(epochs_history, idx_e)
    Base.push!(training_history, training_loss)
    Base.push!(testing_history, testing_loss)

    println("Epoch = ", idx_e,
            ", training loss = ", training_loss,
            ", testing loss = ", testing_loss)
  end
end

# Callback function.
function cb(x)
  epoch = -1
  if Base.hasproperty(x, :iteration)
    epoch = x.iteration
  else
    epoch = x
  end
  training_loss = BLNM.loss_MSE_BLNM(NN, times_train, params_train, outputs_train)
  testing_loss = BLNM.loss_MSE_BLNM(NN, times_test, params_test, outputs_test)

  Base.push!(epochs_history, epoch + num_epochs_Adam)
  Base.push!(training_history, training_loss)
  Base.push!(testing_history, testing_loss)

  println("Epoch = ", epoch,
          ", training loss = ", training_loss,
          ", testing loss = ", testing_loss)

  return false
end

# Train the neural network using one or multiple (deterministic) optimizers combined together.
for (label, optimizer, n_epochs) in zip(labels, optimizers, num_epochs)
  Zygote.refresh()
  local ps = Flux.params(NN)

  loss_fun, grad_fun, p0 = BLNM.loss_grad_initparams_BLNM(BLNM.loss_MSE_BLNM, NN, times_train, params_train, outputs_train, ps)

  println("***** ", label, " OPTIMIZER FOR ", n_epochs, " EPOCHS OR A MAXIMUM TRAINING TIME OF ", (max_time ÷ 60), " MINUTES *****")
  @time res = Optim.optimize(loss_fun,
                             grad_fun,
                             p0,
                             optimizer,
                             Optim.Options(iterations = n_epochs, time_limit = max_time, store_trace = false, show_trace = false, allow_f_increases = true, show_every = out_freq, callback = cb))
end

# Save training history.
output_file = "figs/BLNM_"
for neurons in neurons_per_layer
  global output_file = output_file * Base.string(neurons) * "-"
end
output_file = Base.chop(output_file, tail = 1)
output_file = output_file * "_states" * Base.string(num_states) * "_disentanglement" * Base.string(disentanglement_level) * "_train-indices" * Base.string(train_indices[1]) * ":" * Base.string(train_indices[end]) * "_Adam" * Base.string(num_epochs_Adam) * "_" * labels[1] * Base.string(num_epochs[1]) * "_seed" * Base.string(seed) * "_ToR-ORd.pdf"
Plots.plot(epochs_history, training_history, label = "Train", xlabel = L"\mathrm{Epochs}", ylabel = L"\mathrm{Loss}", c = :black, marker =:circle, yaxis =:log)
Plots.plot!(epochs_history, testing_history, label = "Test", c = :red, marker =:circle, yaxis =:log)
Plots.title!(L"\mathrm{History}")
Plots.savefig(output_file) 

# Save neural network.
output_file = output_folder * "BLNM_"
for neurons in neurons_per_layer
  global output_file = output_file * Base.string(neurons) * "-"
end
output_file = Base.chop(output_file, tail = 1)
output_file = output_file * "_states" * Base.string(num_states) * "_disentanglement" * Base.string(disentanglement_level) * "_train-indices" * Base.string(train_indices[1]) * ":" * Base.string(train_indices[end]) * "_ToR-ORd.bson"
@save output_file NN