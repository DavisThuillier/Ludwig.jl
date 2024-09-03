module Ludwig

include("constants.jl") # Physical constants

### Submodules ###
include("Groups.jl");
using .Groups

include("Lattices.jl"); 
using .Lattices

export Lattices, Groups

import StaticArrays: SVector, MVector, MMatrix, SMatrix
using LinearAlgebra
using ForwardDiff
import DataStructures: OrderedDict
using Interpolations
using IterativeSolvers

include("utilities.jl")
include("mesh/marching_squares.jl")
include("mesh/mesh.jl")
include("integration.jl")
include("properties.jl")
include("vertex_factors.jl")

    
end # module Ludwig
