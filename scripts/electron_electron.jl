using Ludwig
using HDF5
using Interpolations
using StaticArrays
import LinearAlgebra: eigvals

function main(T::Real, n_ε::Int, n_θ::Int, outfile::String)
    T = kb * T # Convert K to eV

    mesh, Δε = Ludwig.multiband_mesh(hamiltonian, T, n_ε, n_θ)

    ℓ = length(mesh.patches)

    # Initialize file - will error if
    h5open(outfile, "cw") do fid
        g = create_group(fid, "data")
        write_attribute(g, "n_e", n_ε)
        write_attribute(g, "n_theta", n_θ)
        g["corners"] = copy(transpose(reduce(hcat, mesh.corners)))
        g["momenta"] = copy(transpose(reduce(hcat, map(x -> x.momentum, mesh.patches))))
        g["velocities"] = copy(transpose(reduce(hcat, map(x -> x.v, mesh.patches))))
        g["energies"] = map(x -> x.energy, mesh.patches) 
        g["dVs"] = map(x -> x.dV, mesh.patches)
        g["corner_ids"] = copy(transpose(reduce(hcat, map(x -> x.corners, mesh.patches))))
    end 

    L = zeros(Float64, ℓ, ℓ) # Scattering operator

    N = 1001 # Interpolation dimension for energies
    Ludwig.electron_electron!(L, mesh.patches, Δε, T, hamiltonian, N, eigenvecs!)

    # Write scattering operator out to file
    h5open(outfile, "cw") do fid
        g = fid["data"]
        g["L"] = L
    end
end

function argument_handling()
    T   = parse(Float64, ARGS[1])
    n_ε = parse(Int, ARGS[2])
    n_θ = parse(Int, ARGS[3])
    band_file = ARGS[4]
    out_dir = ARGS[5]
    return T, n_ε, n_θ, band_file, out_dir
end

T, n_ε, n_θ, band_file, dir = argument_handling()
include(joinpath(@__DIR__, band_file))
outfile = joinpath(@__DIR__, dir, "$(material)_$(T)_$(n_ε)x$(n_θ).h5")
main(T, n_ε, n_θ, outfile)