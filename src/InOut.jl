module InOut

# Pkl handler.
using PyCall
@pyimport pickle
# Read raw file with `pkl` extension.
function read_pkl(path_to_file::String)::Dict{Any, Any}
  r = nothing
  @pywith pybuiltin("open")(path_to_file, "rb") as f begin
    r = pickle.load(f)
  end
  return r
end
# Read dict to `pkl` file.
function save_pkl(path_to_file::String, dataset::Dict{Any, Any})
  @pywith pybuiltin("open")(path_to_file, "wb") as f begin
    pickle.dump(dataset, f)
  end
end

# CSV handler.
using CSV
using DataFrames
# Save dataframe in a file with `csv` extension.
function write_csv(path_to_file::String, object::DataFrame, header::Bool = true)
  CSV.write(path_to_file, object, writeheader = header)
end
# Read file with `csv` extension into a 2D `matrix`.
function read_csv(path_to_file::String, header::Bool = true)::Matrix{Float64}
  return Matrix(CSV.read(path_to_file, DataFrame, header = header))
end

end