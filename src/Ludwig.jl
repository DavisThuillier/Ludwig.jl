module Ludwig

include("constants.jl") # Physical constants

### Submodules ###
include("Groups.jl");
using .Groups

include("Lattices.jl"); 
using .Lattices

include(joinpath("mesh","MarchingSquares.jl"))
using .MarchingSquares


export Lattices, Groups, MarchingSquares

import StaticArrays: SVector, MVector, MMatrix, SMatrix
using LinearAlgebra
using ForwardDiff
using Interpolations
using IterativeSolvers

include("utilities.jl")
include("mesh/mesh.jl")
include("integration.jl")
include("properties.jl")
include("vertex_factors.jl")

    
end # module Ludwig
