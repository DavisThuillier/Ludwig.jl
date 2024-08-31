using Ludwig
using HDF5
using Interpolations
using StaticArrays
using LinearAlgebra
using ProgressBars

function main(T::Real, n_ε::Int, n_θ::Int, outfile)
    T = kb * T # Convert K to eV

    mesh = Ludwig.multiband_mesh(bands, x -> [1.0], T, n_ε, n_θ)
    f0s = map(x -> f0(x.energy, T), mesh.patches)
    ℓ = length(mesh.patches)

    # vertex_model = ""
    vertex_model = "Constant"

    if vertex_model == "Constant" 
        Fpp(p1, p2) = 1.0
        Fpk(p1, k, μ) = 1.0
    else
        Fpp = vertex_pp
        Fpk = vertex_pk
    end

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

    for i in ProgressBar(1:ℓ)
        for j in 1:ℓ
            L[i,j] = Ludwig.electron_electron(mesh.patches, f0s, i, j, itps, T, Fpp, Fpk)
        end
    end

    # Write scattering operator out to file
    h5open(outfile, "cw") do fid
        g = fid["data"]
        g["L"] = L
    end

end

n_ε = 10
n_θ = 24

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))

bands = [bands[3]]

for T in 12.0:2.0:12.0
    outfile = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "single_band_test", "$(material)_$(T)_$(n_ε)x$(n_θ).h5")
    main(T, n_ε, n_θ, outfile)
end