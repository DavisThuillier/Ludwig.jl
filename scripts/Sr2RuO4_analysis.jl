using Ludwig
using HDF5
using StaticArrays
using CairoMakie, LaTeXStrings, Colors
using LinearAlgebra
using LsqFit

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))

function load(file)
    h5open(file, "r") do fid
        g = fid["data"]

        Γ = 2pi * read(g, "Γ") 
        k = reinterpret(SVector{2,Float64}, vec(transpose(read(g, "momenta"))))
        v = reinterpret(SVector{2,Float64}, vec(transpose(read(g, "velocities"))))
        E = read(g, "energies")
        dV = read(g, "dVs")
        corners = read(g, "corners")
        corner_ids = read(g, "corner_ids")

        # Enforce particle conservation
        for i in 1:size(Γ)[1]
            Γ[i,i] -= sum(Γ[i, :])
        end

        return Γ, k, v, E, dV, corners, corner_ids
    end
end

function modes(file, T, bands)
    Γ, k, v, E, dV, corners, corner_ids = load(file)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd) # Energy derivative of FD on grid points

    M = diagm(sqrt.(w .* dV)) * Γ * (diagm(1 ./ sqrt.(w .* dV)))

    quads = Vector{Vector{SVector{2, Float64}}}(undef, 0)
    for i in 1:size(corner_ids)[1]
        push!(quads, map(x -> SVector{2}(corners[x, :]), corner_ids[i, :]))
    end

    τ = hbar

    eigenvalues  = 1e-12 * eigvals(Γ) / τ
    eigenvectors = eigvecs(Γ)
    
    colors = Vector{RGB{Float64}}(undef, length(eigenvalues))
    for i in eachindex(eigenvalues)
        # println("λ_$(i) = ", eigenvalues[i])
        weights = Ludwig.band_weight(eigenvectors[:, i], bands)
        # for j in eachindex(bands)
            # println("   ", bands[j], ": ", weights[j])
        # end
        colors[i] = RGB{Float64}(weights...)

        # if i == 1
        #     f = Figure(size = (1000,1000), fontsize = 30)
        #     ax  = Axis(f[1,1], aspect = 1.0, title = latexstring("\$ \\tau_{$(i -1)} = $(round(real(eigenvalues[i]), digits = 6)) \\text{ fs}\$"), limits = (-0.5, 0.5, -0.5, 0.5))
        #     p = poly!(ax, quads, color = norm.(v), colormap = :viridis, 
        #     )
        #     Colorbar(f[1,2], p)
        #     display(f)
        # end
    end

    N = 1600
    f = Figure(size = (2000, 1000), fontsize = 42)
    ax  = Axis(f[1,1], 
               aspect = 2.4,
               title  = latexstring("\$\\mathrm{Sr_2RuO_4}\$ collision operator spectrum"),
               ylabel = L"\gamma (\mathrm{ps}^{-1})",
               xlabel = "Mode"
    )
    scatter!(ax, real.(eigenvalues)[1:N], color = colors[1:N], markersize = 4)


    α = MarkerElement(color = RGB(1,0,0), marker = :circle, markersize = 15)
    β = MarkerElement(color = RGB(0,1,0), marker = :circle, markersize = 15)
    γ = MarkerElement(color = RGB(0,0,1), marker = :circle, markersize = 15)
    axislegend(ax, [α, β, γ], [L"\alpha", L"\beta", L"\gamma"], valign = :bottom)
    display(f)

    # outfile = joinpath(@__DIR__, "..", "plots", "Sr2RuO4_spectrum.png")
    # save(outfile, f)
end

function convergence()
    ρ = [1870 1870 1880 1885 1887;
         1883 1887 1891 1890 1898;
         1903 1906 1911 1912 1912]

    n_ε = [32, 34, 36, 38, 40]
    n_θ = [7, 8, 9]

    f = Figure()
    ax = Axis(f[1,1])
    m(t, p) = p[1] .+ p[2] * t
    domain = 0.0:0.01:0.15
    for i in 1:size(ρ)[2]
        fit = curve_fit(m, 1 ./ n_θ, ρ[:, i], [1900, -0.2])
        scatter!(ax, 1 ./ n_θ, ρ[:, i], label = "$(n_ε[i])")
        lines!(ax, domain, m(domain, fit.param))
    end
    axislegend(ax, title = L"n_\theta")
    display(f)

end

function main(T, n_ε, n_θ, bands)
    file = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "Sr2RuO4_$(T)_$(n_ε)x$(n_θ).h5")
    T = kb * T

    for band in bands
        @show 1 / (2pi)^2 / Ludwig.get_n(band)
    end
    # convergence()
    return nothing

    Γ, k, v, E, dV, corners, corner_ids = load(file)
    fd = f0.(E, T) # Fermi dirac on grid points

    # w = fd .* (1 .- fd) # Energy derivative of FD on grid points
    # M = diagm(sqrt.(w .* dV)) * Γ * (diagm(1 ./ sqrt.(w .* dV)))
    # ℓ = size(Γ)[1]
    # f = Figure(size = (4000, 4000))
    # ax = Axis(f[1,1], aspect = 1.0)
    # h = heatmap!(ax, -M, colorrange = (-0.002, 0.002), colormap = :lisbon)
    # # Colorbar(f[1,2], h)
    # display(f)

    σ = Ludwig.conductivity(Γ, v, E, dV, T)
    @show 1 / σ[1,1]
    # modes(file, T, bands)
end

T = 12.0
n_ε = 10
n_θ = 36
main(T, n_ε, n_θ, bands)