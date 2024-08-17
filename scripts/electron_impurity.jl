using Ludwig
using HDF5
using Interpolations
using StaticArrays

function main(T::Real, n_ε::Int, n_θ::Int, outfile::String)
    T = kb * T # Convert K to eV

    mesh = Ludwig.multiband_mesh(bands, orbital_weights, T, n_ε, n_θ)
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


    V_squared(k) = 1.0

    L = zeros(Float64, ℓ, ℓ) # Scattering operator

    V_squared = Matrix{Float64}(undef, 3, 3)
    Vs = [133.12, 42.209, 8.1114]
    for i in 1:3
        for j in 1:3
            V_squared[i,j] = sqrt(Vs[i] * Vs[j]) # Geometric mean
        end
    end

    Ludwig.electron_impurity!(L, mesh.patches, V_squared)

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

dir = joinpath("..", "data", "Sr2RuO4", "model_1")
band_file = joinpath("materials", "Sr2RuO4.jl")
include(joinpath(@__DIR__, band_file))
n_ε = 12
n_θ = 38
for T in 2.0:0.5:14.0
    @show T
    outfile = joinpath(@__DIR__, dir, "$(material)_unitary_imp_$(T)_$(n_ε)x$(n_θ).h5")
    main(T, n_ε, n_θ, outfile)
end 