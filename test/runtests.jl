using Ludwig
using Test
using Random

@testset "Lattice Tests" begin
    include("lattice_tests.jl")
end

@testset "Group Tests" begin
    include("group_tests.jl")
end

@testset "IBZ Fill Tests" begin
    include("ibz_fill_test.jl")
end