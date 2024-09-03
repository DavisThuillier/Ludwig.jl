using Ludwig
using Test

@testset "Lattice Tests" begin
    include("lattice_tests.jl")
end

@testset "Group Tests" begin
    include("group_tests.jl")
end