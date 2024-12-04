using Ludwig
using HDF5
using Interpolations
using StaticArrays
using LinearAlgebra

function main(T::Real, n_ε::Int, n_θ::Int, outfile::String, α::Real = 6.0)

    mesh = Ludwig.multiband_mesh(bands, orbital_weights, kb * T, n_ε, n_θ; α = α)
    ℓ = length(mesh.patches)

    # Initialize file - will error if
    h5open(outfile, "cw") do f
        g = create_group(f, "data")
        g["n_ε"] = n_ε
        g["n_θ"] = n_θ
        g["T"] = T
        g["corners"] = copy(transpose(reduce(hcat, mesh.corners)))
        g["momenta"] = copy(transpose(reduce(hcat, map(x -> x.momentum, mesh.patches))))
        g["velocities"] = copy(transpose(reduce(hcat, map(x -> x.v, mesh.patches))))
        g["energies"] = map(x -> x.energy, mesh.patches) 
        g["dVs"] = map(x -> x.dV, mesh.patches)
        g["corner_ids"] = copy(transpose(reduce(hcat, map(x -> x.corners, mesh.patches))))
    end 

    T = kb * T # Convert K to eV

    N = 1001
    x = LinRange(-0.5, 0.5, N)
    E = Array{Float64}(undef, N, N)
    itps = Vector{ScaledInterpolation}(undef, length(bands))
    for μ in eachindex(bands)
        for i in 1:N, j in 1:N
            E[i, j] = bands[μ]([x[i], x[j]]) # Get eigenvalues (bands) of each k-point
        end

        itp = interpolate(E, BSpline(Cubic(Line(OnGrid()))))
        itps[μ] = scale(itp, x, x)
    end

    L = zeros(Float64, ℓ, ℓ) # Scattering operator
    f0s = map(x -> f0(x.energy, T), mesh.patches) # Fermi-Dirac Grid

    Threads.@threads for i in 1:ℓ
        for j in 1:ℓ
            L[i,j] = Ludwig.electron_electron(mesh.patches, f0s, i, j, itps, T, vertex_pp, vertex_pk, 36)
        end
    end

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
    α = parse(Float64, ARGS[4])
    band_file = ARGS[4]
    out_dir = ARGS[5]
    return T, n_ε, n_θ, α, band_file, out_dir
end

T, n_ε, n_θ, α, band_file, dir = argument_handling()
include(joinpath(@__DIR__, band_file))
outfile = joinpath(@__DIR__, dir, "$(material)_$(T)_$(n_ε)x$(n_θ)_$(α).h5")
main(T, n_ε, n_θ, outfile, α)