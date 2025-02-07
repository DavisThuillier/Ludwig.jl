module Ludwig

include("constants.jl") # Physical constants

### Submodules ###
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

export Lattices, Groups, MarchingSquares

import StaticArrays: SVector, MVector, MMatrix, SMatrix
using LinearAlgebra
using ForwardDiff
using Interpolations
using IterativeSolvers

include("utilities.jl")
include("mesh/mesh.jl")
include("properties.jl")
include("vertex_factors.jl")

    
end # module Ludwig
