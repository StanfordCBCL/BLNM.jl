# External packages.
using Flux
using Zygote
using Optim, Optimization, OptimizationOptimJL, Optimisers, Statistics
using Plots
using Random
using BSON: @save

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
dataset_file = "data/ECGs_HLHS.pkl"
# Output folder for the trained Branched Latent Neural Map.
output_folder = "NNs/"
# Training indices from the dataset.
train_indices = range(1, 150)
# Test indices from the dataset.
test_indices  = range(151, 200)
# Number of neurons for each hidden layer.
neurons_per_layer = [19, 19, 19, 19, 19, 19, 19]
# Number of states (physical and latent variables).
num_states = 10
# Disentanglement level.
disentanglement_level = 2
# List of (deterministic) optimizers for each optimization phase.
# Available options: [NelderMead(), GradientDescent(), ConjugateGradient(), BFGS(), LBFGS(), Newton()]
optimizers = [BFGS()]
# List of optimizers labels for each optimization phase (needed for printing only).
labels = ["BFGS"]
# Maximum number of epochs for each optimization phase.
num_epochs = [50000]
# Output frequency for the optimizer callback.
out_freq = 100
# Training wall time (in seconds) for each optimization phase.
max_time = 18000
# Time step (in milliseconds).
dt = 5.0

# Training/Testing samples.
num_train = Base.length(train_indices)
num_test = Base.length(test_indices)
num_samples = num_train + num_test

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
params_adim = Utils.interpolate_linear(dataset["parameters"], dataset["times"], times)
num_params = Base.size(params_adim)[1]
# Maximum and minimum of [tLVstim, GNa, GCaL, GKr, Dpurk, Dani, Diso] in the original range.
params_min = Array{Float64, 3}(undef, num_params, 1, 1) .= [0.1950 , 7.5400 , 2.0365e-5, 0.0771, 1.0027, 0.0084, 0.0028]
params_max = Array{Float64, 3}(undef, num_params, 1, 1) .= [99.3059, 29.4130, 7.9358e-5, 0.3057, 3.4826, 0.0331, 0.0110]
# Import outputs in the adimensional range [-1, 1].
outputs_adim = Utils.interpolate_linear(dataset["outputs"], dataset["times"], times)
num_outs = Base.size(outputs_adim)[1]
# Maximum and minimum of [V1, V2, V3, V4, V5, V6, RA, LA, F] in the original range.
out_min = Array{Float64, 3}(undef, num_outs, 1, 1) .= [-4.3731, -4.2820, -3.0456, -3.1056, -4.6857, -2.0958, -1.2211, -0.9557, -0.7815]
out_max = Array{Float64, 3}(undef, num_outs, 1, 1) .= [2.2964 , 3.8095 , 7.9580 , 5.8659 , 4.7494 , 2.9059 , 0.6399 , 0.3444 , 1.8498 ]

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

# Callback function.
function cb(x)
  epoch = -1
  if Base.hasproperty(x, :iteration)
    epoch = x.iteration
  else
    epoch = x
  end

  println("Training loss = ", BLNM.loss_MSE_BLNM(NN, times_train, params_train, outputs_train),
          ", testing loss = ", BLNM.loss_MSE_BLNM(NN, times_test, params_test, outputs_test),
          ", epoch = ", epoch)

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

# Save neural network.
output_file = output_folder * "BLNM_"
for neurons in neurons_per_layer
  global output_file = output_file * Base.string(neurons) * "-"
end
output_file = Base.chop(output_file, tail = 1)
output_file = output_file * "_states" * Base.string(num_states) * "_disentanglement" * Base.string(disentanglement_level) * "_train-indices" * Base.string(train_indices[1]) * ":" * Base.string(train_indices[end]) * "_HLHS_ECGs.bson"
@save output_file NN