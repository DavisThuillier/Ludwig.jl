using Ludwig
using HDF5
using Interpolations
using StaticArrays

function main(T::Real, n_ε::Int, n_θ::Int, outfile::String)
    T = kb * T # Convert K to eV

    grid    = Vector{Patch}(undef, 0)
    corners = Vector{SVector{2, Float64}}(undef, 0)
    Δε = 0
    for h in bands
        mesh, Δε = Ludwig.generate_mesh(h, T, n_ε, n_θ, 2000)
        grid = vcat(grid, map(x -> Patch(
                                    x.momentum, 
                                    x.energy,
                                    x.v,
                                    x.dV,
                                    x.de,
                                    x.jinv, 
                                    x.djinv,
                                    x.corners .+ length(corners)
                                ), mesh.patches)
        )
        corners = vcat(corners, mesh.corners)
    end

    ℓ = length(grid)

    h5open(outfile, "cw") do fid
        g = create_group(fid, "data")
        write_attribute(g, "n_e", n_ε)
        write_attribute(g, "n_theta", n_θ)
        g["corners"] = copy(transpose(reduce(hcat, corners)))
        g["momenta"] = copy(transpose(reduce(hcat, map(x -> x.momentum, grid))))
        g["velocities"] = copy(transpose(reduce(hcat, map(x -> x.v, grid))))
        g["energies"] = map(x -> x.energy, grid) 
        g["dVs"] = map(x -> x.dV, grid)
        g["corner_ids"] = copy(transpose(reduce(hcat, map(x -> x.corners, grid))))
    end 

    Γ = zeros(Float64, ℓ, ℓ) # Scattering operator
    V_squared(k) = 1.0
    Ludwig.electron_impurity!(Γ, grid, Δε, (x,y) -> 1.0)

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

dir = joinpath("..", "data", "Sr2RuO4")
band_file = joinpath("materials", "Sr2RuO4.jl")
include(joinpath(@__DIR__, band_file))
n_ε = 12
n_θ = 38
for T in 2.0:0.5:14.0
    @show T
    outfile = joinpath(@__DIR__, dir, "$(material)_imp_$(T)_$(n_ε)x$(n_θ).h5")
    main(T, n_ε, n_θ, outfile)
end 