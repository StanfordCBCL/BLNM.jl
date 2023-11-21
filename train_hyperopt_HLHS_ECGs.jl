# External packages.
using Hyperopt
using Flux
using Zygote
using Optim, Optimization, OptimizationOptimJL, Optimisers, Statistics
using MLBase
using Plots
using DataFrames
using Random
using BSON: @save

# I/O features.
include("./src/InOut.jl")
using .InOut
# Utilities.
include("./src/BLNM.jl")
using .BLNM

# MPI-based K-fold cross validation.
using MPI

# Initialize MPI.
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)
root = 0

# Only rank zero outputs will be displayed.
if rank != root
  redirect_stdout(devnull)
end

# Set random seed for reproducibility.
seed = 1
Random.seed!(seed)
Random.rand(seed)

### User-defined parameters ###
# Path to dataset with the ECGs from electrophysiology simulations.
dataset_file = "data/ECGs_HLHS.pkl"
# Output folder for the trained Branched Latent Neural Map(s).
output_folder = "NNs/"
# Training and validation indices from the dataset.
train_valid_indices = collect(1 : 150)
# K-fold cross validation size.
K = size
# List of optimizers for each optimization phase.
# Available options: [NelderMead(), GradientDescent(), ConjugateGradient(), BFGS(), LBFGS(), Newton()]
optimizers = [BFGS()]
# Maximum number of epochs for each optimization phase.
num_epochs = [10000]
# Training wall time (in seconds) for each optimization phase.
max_time = 18000
# Time step (in milliseconds).
dt = 5.0
# Number of iterations for the hyperparameters optimizer.
num_iters = 50
# Range of neurons for the hyperparameters optimizer.
neurons_range = collect(10 : 30)
# Range of layers for the hyperparameters optimizer.
layers_range = collect(1 : 8)
# Range of states for the hyperparameters optimizer.
# Note that the minimum number must be greater or equal than the number of outputs.
states_range = collect(9 : 12)

# Training/Validation samples.
num_samples = Base.length(train_valid_indices)

# Read dataset.
dataset = InOut.read_pkl(dataset_file)
# Import times in the original range [0, 600] milliseconds.
times = Base.range(dataset["times"][1], dataset["times"][end], step = dt)
tspan = (dataset["times"][1], dataset["times"][end])
num_times = Base.length(times)
times_adim = Base.zeros(1, num_times, num_samples)
for idx in 1 : num_samples
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

# Initialize K-fold cross validation.
KF = Base.collect(MLBase.Kfold(num_samples, K))
train_indices = KF[rank + 1]
valid_indices = setdiff(train_valid_indices, train_indices)

# Initialize DataFrame for CSV output.
df = DataFrame(neurons = Int64[], layers = Int64[], states = Int64[], disentanglement_level = Int64[], loss_KFold = Float64[])
df_rank = DataFrame(neurons = Int64[], layers = Int64[], states = Int64[], disentanglement_level = Int64[], loss = Float64[])

# Hyperparameters optimizer loop.
ho = @phyperopt for resources = num_iters,
                    sampler = Hyperopt.CLHSampler(dims = [Hyperopt.Categorical(length(neurons_range)),
                                                          Hyperopt.Categorical(length(layers_range)),
                                                          Hyperopt.Categorical(length(states_range))]),
                    neuron = neurons_range,
                    layer = layers_range,
                    num_states = states_range
  
  # Random disentanglement level.
  disentanglement_level = rand(1 : layer)

  println("*****", " ", K, "-FOLD CROSS VALIDATION (ITERATION = ", resources, ", NEURONS = ", neuron, ", LAYERS = ", layer,", STATES = ", num_states, ", DISENTANGLEMENT = ", disentanglement_level, ") *****")
  flush(stdout)

  # Synchronize processes before the next hyperparameters iteration.
  MPI.Barrier(comm)

  neurons_per_layer = Base.repeat([neuron], layer)

  # Training and validation samples.
  num_train = Base.length(train_indices)
  num_valid = Base.length(valid_indices)

  # Splitting between training and validation set.
  params_train = params_adim[:, :, train_indices]
  times_train = times_adim[:, :, train_indices]
  outputs_train = outputs_adim[:, :, train_indices]
  params_valid = params_adim[:, :, valid_indices]
  times_valid = times_adim[:, :, valid_indices]
  outputs_valid = outputs_adim[:, :, valid_indices]

  # Define the Branched Latent Neural Map for Float64 operations.
  NN = BLNM.build_BLNM(neurons_per_layer, 1, num_params, num_states, disentanglement_level)

  # Compile Branched Latent Neural Map (first run).
  NN((times_train, params_train))
  # Compile MSE loss function (first run).
  BLNM.loss_MSE_BLNM(NN, times_train, params_train, outputs_train)

  # Train the neural network using one or multiple (deterministic) optimizers combined together.
  for (optimizer, n_epochs) in zip(optimizers, num_epochs)
    Zygote.refresh()
    ps = Flux.params(NN)
    loss_fun, grad_fun, p0 = BLNM.loss_grad_initparams_BLNM(BLNM.loss_MSE_BLNM, NN, times_train, params_train, outputs_train, ps)

    res = Optim.optimize(loss_fun,
                         grad_fun,
                         p0,
                         optimizer,
                         Optim.Options(iterations = n_epochs, time_limit = max_time, store_trace = false, show_trace = false))
  end

  # Compute final validation errors.
  loss_valid = BLNM.loss_MSE_BLNM(NN, times_valid, params_valid, outputs_valid)

  # Save neural network.
  output_file_BLNM = output_folder * "BLNM_"
  for neurons in neurons_per_layer
    output_file_BLNM = output_file_BLNM * Base.string(neurons) * "-"
  end
  output_file_BLNM = Base.chop(output_file_BLNM, tail = 1)
  output_file_BLNM = output_file_BLNM * "_states" * Base.string(num_states) * "_disentanglement" * Base.string(disentanglement_level) * "_trainvalid-indices" * Base.string(train_valid_indices[1]) * ":" * Base.string(train_valid_indices[end]) * "_K" * Base.string(rank + 1) * "_HLHS_ECGs.bson"
  @save output_file_BLNM NN

  # Synchronize processes before hyperparameters tuning.
  MPI.Barrier(comm)

  # Compute generalization errors.
  sum_loss_valid  = MPI.Allreduce(loss_valid, +, comm)
  mean_loss_valid = sum_loss_valid / K

  # Save output of hyperparameters tuner.
  if (rank == root)
    push!(df, [neuron layer num_states disentanglement_level mean_loss_valid])
    InOut.write_csv(output_folder * "hyperparameters_seed" * Base.string(seed) * ".csv", df)
  end
  push!(df_rank, [neuron layer num_states disentanglement_level loss_valid])
  InOut.write_csv(output_folder * "hyperparameters_rank" * Base.string(rank) * "_seed" * Base.string(seed) * "_HLHS_ECGs.csv", df_rank)

  # Synchronize processes after hyperparameters tuning.
  MPI.Barrier(comm)

  # Return generalization error to the hyperparameters tuner.
  mean_loss_valid
end

if (rank == root)
  Plots.plot(ho)
  Plots.savefig(output_folder * "hypertuning_seed" * Base.string(seed) * "_HLHS_ECGs.png")
end