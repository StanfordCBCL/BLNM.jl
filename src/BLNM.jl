module BLNM

using Flux
using Fluxperimental
using Interpolations
using Zygote

"""
Transform a 3D `tensor` into the adimensional range [-1, 1].
"""
function adimensionalize(tensor::Array{Float64, 3}, tensor_min::Array{Float64, 3}, tensor_max::Array{Float64, 3})::Array{Float64, 3}
  return (2.0 .* tensor .- tensor_min .- tensor_max) ./ (tensor_max .- tensor_min)
end

"""
Transform a 2D `matrix` into the adimensional range [-1, 1].
"""
function adimensionalize(matrix::Matrix{Float64}, matrix_min::Matrix{Float64}, matrix_max::Matrix{Float64})::Matrix{Float64}
  return (2.0 .* matrix .- matrix_min .- matrix_max) ./ (matrix_max .- matrix_min)
end

"""
Transform a 3D `tensor` into the original range [`tensor_min`, `tensor_max`].
"""
function dimensionalize(tensor::Array{Float64, 3}, tensor_min::Array{Float64, 3}, tensor_max::Array{Float64, 3})::Array{Float64, 3}
  return (tensor_min .+ tensor_max .+ (tensor_max .- tensor_min) .* tensor) ./ 2
end
"""
Transform a 2D `matrix` into the original range [`matrix_min`, `matrix_max`].
"""
function dimensionalize(matrix::Matrix{Float64}, matrix_min::Matrix{Float64}, matrix_max::Matrix{Float64})::Matrix{Float64}
  return (matrix_min .+ matrix_max .+ (matrix_max .- matrix_min) .* matrix) ./ 2
end

"""
Interpolate a generic 3D `tensor`, defined on a `times` vector, onto a different `times_interp` vector.
"""
function interpolate(tensor::Array{Float64, 3},
                     times::Vector{Float64},
                     times_interp::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64})::Array{Float64, 3}
  (num_variables, num_times, num_samples) = Base.size(tensor)
  tensor_interp = Array{Float64, 3}(undef, num_variables, Base.length(times_interp), num_samples)
  Threads.@threads for idx_s in 1 : num_samples
    Threads.@threads for idx_v in 1 : num_variables
      linear_interp = Interpolations.linear_interpolation(times, tensor[idx_v, :, idx_s])
      tensor_interp[idx_v, :, idx_s] = linear_interp(times_interp)
    end
  end

  return tensor_interp
end

"""
Build a Branched Latent Neural Map with (`num_inps_1` + `num_inps_2`) inputs and `num_outputs` outputs.
The number of neurons per hidden layer are specified in the `neurons_per_layer` vector.
The number of hidden layers where the first and second inputs are separated is specified by `disentanglement_level`.
"""
function build_BLNM(neurons_per_layer::Vector{Int64},
                    num_inps_1::Int64, num_inps_2::Int64,
                    num_outs::Int64,
                    disentanglement_level::Int64 = 1)::Chain{Vector{Any}}
  n_layers = Base.length(neurons_per_layer)

  if (disentanglement_level < 1 || disentanglement_level > n_layers)
    Base.throw("The disentanglement level must be between 1 and the total number of hidden layers")
  end

  BLNM_architecture = []

  BLNM_1_architecture = []
  BLNM_2_architecture = []
  chain_1 = []
  chain_2 = []
  inps_1 = num_inps_1
  inps_2 = num_inps_2
  branch_inps_1 = 0
  branch_inps_2 = 0
  for index in 1 : disentanglement_level
    branch_inps_1 = neurons_per_layer[index] ÷ 2
    branch_inps_2 = neurons_per_layer[index] - branch_inps_1
      
    Base.push!(BLNM_1_architecture, Flux.Dense(inps_1 => branch_inps_1, Flux.tanh_fast))
    Base.push!(BLNM_2_architecture, Flux.Dense(inps_2 => branch_inps_2, Flux.tanh_fast))

    inps_1 = branch_inps_1
    inps_2 = branch_inps_2
  end

  chain_1 = Flux.Chain(BLNM_1_architecture)
  chain_2 = Flux.Chain(BLNM_2_architecture)
  chain_1 = Flux.f64(chain_1)
  chain_2 = Flux.f64(chain_2)
  Base.push!(BLNM_architecture, Fluxperimental.Join(vcat,
                                                    # First branch (e.g. space or time).
                                                    chain_1,
                                                    # Second branch (e.g. scalar model parameters).
                                                    chain_2
                                                   ))

  for index in disentanglement_level : (n_layers - 1)
    Base.push!(BLNM_architecture, Flux.Dense(neurons_per_layer[index] => neurons_per_layer[index + 1], Flux.tanh_fast))
  end
  Base.push!(BLNM_architecture, Flux.Dense(neurons_per_layer[end] => num_outs, Flux.identity))

  BLNM = Flux.Chain(BLNM_architecture)
  BLNM = Flux.f64(BLNM)

  return BLNM
end

"""
Mean Square Error (MSE) loss function for a `BLNM` receiving `input_1` and `input_2` as inputs
, comparing the predictions with `output`, as 3D tensors.
"""
function loss_MSE_BLNM(BLNM::Chain{Vector{Any}},
                       input_1::Array{Float64, 3}, input_2::Array{Float64, 3},
                       output::Array{Float64, 3})::Float64
  return Flux.mse(BLNM((input_1, input_2))[1 : Base.size(output)[1], :, :], output)
end
"""
Mean Square Error (MSE) loss function for a `BLNM` receiving `input_1` and `input_2` as inputs
, comparing the predictions with `output`, as 2D matrices.
"""
function loss_MSE_BLNM(BLNM::Chain{Vector{Any}},
                       input_1::Matrix{Float64}, input_2::Matrix{Float64},
                       output::Matrix{Float64})::Float64
  return Flux.mse(BLNM((input_1, input_2))[1 : Base.size(output)[1], :, :], output)
end

veclength(grads::Zygote.Grads) = Base.sum(Base.length(grads[p]) for p in grads.params)
veclength(params::Flux.Params) = Base.sum(length, params.params)
veclength(x) = Base.length(x)
zeros(grads::Zygote.Grads) = Base.zeros(veclength(grads))
zeros(pars::Flux.Params) = Base.zeros(veclength(pars))
"""
Compile `loss` function and its gradients given a set of tunable parameters `pars` for a Branched Latent Neural Map `BLNM`.
The `BLNM` receives `input_1` and `input_2` as inputs and compares the predictions with `output`.
"""
function loss_grad_initparams_BLNM(loss::Function,
                                   BLNM::Chain{Vector{Any}},
                                   input_1::Array{Float64, 3}, input_2::Array{Float64, 3}, output::Array{Float64, 3},
                                   pars::Union{Flux.Params})
  grads = Zygote.gradient(pars) do
    loss(BLNM, input_1, input_2, output)
  end

  p0 = zeros(pars)
  Base.copy!(p0, pars)

  grad_fun = function (g, w)
    Base.copy!(pars, w)
    grads = Zygote.gradient(pars) do
      loss(BLNM, input_1, input_2, output)
    end
    Base.copy!(g, grads)
  end

  loss_fun = function (w)
    Base.copy!(pars, w)
    loss(BLNM, input_1, input_2, output)
  end

  return loss_fun, grad_fun, p0
end

end