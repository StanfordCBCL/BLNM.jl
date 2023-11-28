module BLNM

using Flux
using Fluxperimental
using Interpolations
using Zygote

"""
Transform a 3D `tensor` or a 2D `matrix` into the adimensional range [-1, 1].
"""
function adimensionalize(data::Union{Matrix, Array{Float64, 3}},
                         data_min::Union{Matrix, Array{Float64, 3}},
                         data_max::Union{Matrix, Array{Float64, 3}})::Union{Matrix, Array{Float64, 3}}
  return (2.0 .* data .- data_min .- data_max) ./ (data_max .- data_min)
end

"""
Transform a 3D tensor or a 2D matrix `data` into the original range [`data_min`, `data_max`].
"""
function dimensionalize(data::Union{Matrix, Array{Float64, 3}},
                        data_min::Union{Matrix, Array{Float64, 3}},
                        data_max::Union{Matrix, Array{Float64, 3}})::Union{Matrix, Array{Float64, 3}}
  return (data_min .+ data_max .+ (data_max .- data_min) .* data) ./ 2
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
, comparing the predictions with `output`, as 3D tensors or 2D matrices.
"""
function loss_MSE_BLNM(BLNM::Chain{Vector{Any}},
                       input_1::Union{Matrix, Array{Float64, 3}},
                       input_2::Union{Matrix, Array{Float64, 3}},
                       output::Union{Matrix, Array{Float64, 3}})::Float64
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