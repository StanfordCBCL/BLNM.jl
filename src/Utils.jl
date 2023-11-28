module Utils

using Interpolations
using BSplineKit

"""
Exponentiated quadratic kernel for gaussian process emulators.
`t1` and `t2` define the two input vectors.
`l` and `σ` represent amplitude and correlation length, respectively.
"""
function EQkernel(t1, t2, l, σ)
    t₁ = sum(t1 .^ 2, dims = 2)
    t₂ = sum(t2 .^ 2, dims = 2)
    t₃ = 2 * t1 * t2'
    t = (t₁ .+ t₂') - t₃
    return σ^2 * exp.(-0.5 / l^2 * t)
  end

"""
Transform RA, LA and LL `sources` into bipolar (I, II, III) and augmented (aVR, aVL, aVF) leads.
`sources` is a generic 2D `matrix` or 3D `tensor`.
"""
function add_bipolar_and_augmented_limb_leads(sources::Union{Matrix, Array{Float64, 3}})::Union{Matrix, Array{Float64, 3}}
    if isa(sources, Matrix)
      one   = sources[8:8, :] - sources[7:7, :] # I   = LA - RA
      two   = sources[9:9, :] - sources[7:7, :] # II  = LL - RA
      three = sources[9:9, :] - sources[8:8, :] # III = LL - LA
    else
      one   = sources[8:8, :, :] - sources[7:7, :, :] # I   = LA - RA
      two   = sources[9:9, :, :] - sources[7:7, :, :] # II  = LL - RA
      three = sources[9:9, :, :] - sources[8:8, :, :] # III = LL - LA
    end
    aVR = -0.5 * (one + two)   # aVR = - (I + II) / 2
    aVL = 0.5  * (one - three) # aVL = (I - III) / 2
    aVF = 0.5  * (two + three) # aVF = (II + III) / 2
    
    if isa(sources, Matrix)
      sources[7:7, :] = one
      sources[8:8, :] = two
      sources[9:9, :] = three
    else
      sources[7:7, :, :] = one
      sources[8:8, :, :] = two
      sources[9:9, :, :] = three
    end

    sources = Base.cat(sources, aVR, dims = 1)
    sources = Base.cat(sources, aVL, dims = 1)
    sources = Base.cat(sources, aVF, dims = 1)
  
    return sources
end

"""
Interpolate a generic 3D `tensor`, defined on a `times` vector, onto a different `times_interp` vector by using linear polynomials.
"""
function interpolate_linear(tensor::Array{Float64, 3},
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
Interpolate a generic 1D `vector`, defined on a `times` vector, onto a different `times_interp` vector by using cubic splines.
"""
function interpolate_spline(vector::Vector{Float64},
                            times::Vector{Float64},
                            times_interp::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64})::Vector{Float64}
  spline_interp = BSplineKit.interpolate(times, vector, BSplineOrder(3));
  vector_interp = spline_interp.(times_interp);

  return vector_interp;
end

end