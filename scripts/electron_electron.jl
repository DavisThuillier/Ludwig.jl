using Ludwig
using HDF5
using Interpolations
using StaticArrays

function main(T::Real, n_ε::Int, n_θ::Int, outfile::String)
    T = kb * T # Convert K to eV
    N = 1001 # Interpolation dimension

    grid    = Vector{Patch}(undef, 0)
    corners = Vector{SVector{2, Float64}}(undef, 0)
    sitps = Vector{ScaledInterpolation}(undef, length(bands))
    Δε = 0
    for (i, h) in enumerate(bands)
        mesh, Δε = Ludwig.generate_mesh(h, T, n_ε, n_θ, 2000, 1.0)
        grid = vcat(grid, mesh.patches)
        corners = vcat(corners, mesh.corners)

        E = Matrix{Float64}(undef, N, N)
        for kx in 1:N
            for ky in 1:N
                E[kx, ky] = h([-0.5 + (kx-1)/(N-1), -0.5 + (ky-1)/(N-1)])
            end
        end
        itp = interpolate(E, BSpline(Cubic(Line(OnGrid()))))
        sitps[i] = scale(itp, -0.5:1/(N-1):0.5, -0.5:1/(N-1):0.5)
    end

    dVs = map(x -> x.dV, grid)
    ℓ = length(grid)

    h5open(outfile, "cw") do fid
        g = create_group(fid, "data") # In units of t0
        write_attribute(g, "n_e", n_ε)
        write_attribute(g, "n_theta", n_θ)
        g["corners"] = copy(transpose(reduce(hcat, corners)))
        g["momenta"] = copy(transpose(reduce(hcat, map(x -> x.momentum, grid))))
        g["dVs"] = dVs
        g["corner_ids"] = copy(transpose(reduce(hcat, map(x -> x.corners, grid))))
    end 

    Γ = zeros(Float64, ℓ, ℓ) # Scattering operator to be iterated
    Ludwig.electron_electron!(Γ, grid, Δε, T, sitps)

    # Write scattering operator out to file
    h5open(outfile, "cw") do fid
        g = fid["data"]
        g["Γ"] = Γ 
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
outfile = joinpath(@__DIR__, dir, "$(mat)_$(T)_$(n_ε)x$(n_θ).h5")
main(T, n_ε, n_θ, outfile)