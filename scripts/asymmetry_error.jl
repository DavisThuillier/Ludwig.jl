using Ludwig
using HDF5
using Interpolations
using StaticArrays
import LinearAlgebra: dot
using CairoMakie
using LsqFit
using StatsBase
using ProgressBars

function main(T::Real, n_ε::Int, n_θ::Int, α)
    T = kb * T # Convert K to eV

    mesh, Δε = Ludwig.multiband_mesh(bands, orbital_weights, T, n_ε, n_θ; α = α)
    f0s = map(x -> f0(x.energy, T), mesh.patches)
    ℓ = length(mesh.patches)

    vertex_model = ""
    # vertex_model = "Constant"

    if vertex_model == "Constant" 
        Fpp(p1, p2) = 1.0
        Fpk(p1, k, μ) = 1.0
        title = L"F_{\mathbf{k}_1\mathbf{k}_2}^{\mu_1\mu_2} = 1"
    else
        Fpp = vertex_pp
        Fpk = vertex_pk
        title = L"F_{\mathbf{k}_1\mathbf{k}_2}^{\mu_1\mu_2} = (W_{\mathbf{k}_1}^\dagger W_{\mathbf{k}_1})^{\mu_1\mu_2}"
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

    N = 2000
    errors = Vector{Float64}(undef, N)
    counter = 0
    @time while true
        i,j = rand(1:ℓ, 2)
        Lij = Ludwig.electron_electron(mesh.patches, f0s, i, j, itps, T, Fpp, Fpk)
        Lji = Ludwig.electron_electron(mesh.patches, f0s, j, i, itps, T, Fpp, Fpk)

        counter += 1
        errors[counter] = abs( 2 * (Lij - Lji) / (Lij + Lji) )
        mod(counter, 100) == 0 && @show counter
        counter == N && break
    end

    return nothing

    ## Fit histogram of errors ##
    x_lower = 0.0
    x_upper = 0.5
    n_bins  = 100
    step    = (x_upper - x_lower) / n_bins 
    bins = LinRange(x_lower, x_upper, n_bins)

    h = StatsBase.fit(Histogram, errors, bins)
    h = StatsBase.normalize(h, mode = :pdf) 

    midpoints = ((h.edges[1] .+ circshift(h.edges[1], -1)) / 2.0)[1:end-1]
    mean = dot(midpoints, h.weights) * step

    # expmodel(t, p) = p[1] .* exp.(-p[1] .* t)
    # fit = LsqFit.curve_fit(expmodel, midpoints, h.weights, [0.1])
    println("Average Asymmetry Error: ", mean)

    f = Figure()
    ax = Axis(f[1,1], ylabel = "PDF", xlabel = "Asymmetry Error", title = title)
    # xlims!(ax, x_lower, x_upper)
    hist!(ax, errors, bins = bins, normalization = :pdf)
    # domain = LinRange(x_lower, x_upper, 100)
    # lines!(ax, domain, expmodel(domain, fit.param), color = :red)
    display(f)

end

T   = 12.0
n_ε = 12
n_θ = 38

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))

main(T, n_ε, n_θ, 6.0)