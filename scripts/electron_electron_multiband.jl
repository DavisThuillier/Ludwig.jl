using Ludwig
using HDF5
using Interpolations
using StaticArrays
import LinearAlgebra: dot

function main(T::Real, n_ε::Int, n_θ::Int)
    T = kb * T # Convert K to eV

    mesh, Δε = Ludwig.multiband_mesh(bands, orbital_weights, T, n_ε, n_θ)

    ℓ = length(mesh.patches)

    for i in 1:10:ℓ, j in 1:10:ℓ
        Lij = Ludwig.electron_electron(mesh.patches, i, j, bands, Δε, T, vertex_pp, vertex_pk, mesh.n_bands)
        Lji = Ludwig.electron_electron(mesh.patches, j, i, bands, Δε, T, vertex_pp, vertex_pk, mesh.n_bands)
    end

end

T   = 12.0
n_ε = 12
n_θ = 38

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))

main(T, n_ε, n_θ)