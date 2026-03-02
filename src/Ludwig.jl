module Ludwig

include("constants.jl") # Physical constants

### Submodules ###
include("utilities.jl")
using .Utilities

include("GeometryUtilities.jl")
using .GeometryUtilities

include("Groups.jl");
using .Groups

include("Lattices.jl"); 
using .Lattices

include(joinpath("mesh","MarchingSquares.jl"))
using .MarchingSquares

include(joinpath("mesh", "FSMesh.jl"))
using .FSMesh

include("Integration.jl")
using .Integration

export Lattices, Groups, MarchingSquares, FSMesh

import StaticArrays: SVector
using LinearAlgebra
using ForwardDiff
using Interpolations
using IterativeSolvers

include("properties.jl")

end # module Ludwig
