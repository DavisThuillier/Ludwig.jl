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

@testset "Band Tests" begin
    include("band_test.jl")
end

@testset "Mesh Tests" begin
    include("mesh_tests.jl")
end

@testset "Scattering Tests" begin
    include("scattering_tests.jl")
end

@testset "Properties Tests" begin
    include("properties_tests.jl")
end

@testset "Marching Squares Tests" begin
    include("marching_tests.jl")
end
