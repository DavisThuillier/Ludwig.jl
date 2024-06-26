using Ludwig
using HDF5
using StaticArrays
using CairoMakie, LaTeXStrings, Colors
using LinearAlgebra
using LsqFit
using DelimitedFiles

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
        weights = vcat([0.0], Ludwig.band_weight(eigenvectors[:, i], length(bands)))
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

    N = length(eigenvalues)
    f = Figure(size = (2000, 1000), fontsize = 42)
    ax  = Axis(f[1,1], 
               aspect = 2.4,
               title  = latexstring("\$\\mathrm{Sr_2RuO_4}\$ collision operator spectrum"),
               ylabel = L"\gamma (\mathrm{ps}^{-1})",
               xlabel = "Mode"
    )
    scatter!(ax, real.(eigenvalues)[1:N] / 8, color = colors[1:N], markersize = 4)

    α = MarkerElement(color = RGB(1,0,0), marker = :circle, markersize = 15)
    β = MarkerElement(color = RGB(0,1,0), marker = :circle, markersize = 15)
    γ = MarkerElement(color = RGB(0,0,1), marker = :circle, markersize = 15)
    axislegend(ax, [α, β, γ], [L"\alpha", L"\beta", L"\gamma"], valign = :bottom)
    display(f)

    # outfile = joinpath(@__DIR__, "..", "plots", "Sr2RuO4_spectrum_αβγ.png")
    # save(outfile, f)
end

function convergence()
    # -1 indicates missing data
    ρ = #[4.7430 4.7674 4.7792 4.7844 -1     -1
         [4.7862 4.7964 4.7928 4.8121 -1     -1
         4.8340 4.8451 4.8478 4.8500 4.8533 4.8702;
         4.8623 4.8759 4.8662 4.8843 4.8729 4.8886;
         4.8853 4.8969 4.8933 4.8965 4.9004 4.9056;
         4.9006 4.9150 4.9085 4.9152 -1     -1    ;
         -1     -1 4.9218 -1     4.9216 4.93332]

    n_θ = [34, 36, 38, 40, 42, 44]
    n_ε = [8, 9, 10, 11, 12, 13]

    f = Figure(fontsize = 20, size = (1600, 600))
    ax = Axis(f[1,1], 
        xlabel = L"n_\theta^{-1}",
        ylabel= L"\rho (\mathrm{\mu\Omega cm})",
        xlabelsize = 28,
        ylabelsize = 28)
    m(t, p) = p[1] .+ p[2] * t
    domain = 0.0:0.01:0.04
    ρ∞ = Vector{Float64}(undef, length(n_ε))
    for i in eachindex(n_ε)
        θs = n_θ[ρ[i, :] .> 0.0]
        ρs = filter(x -> x>0, ρ[i,:])
        @show θs, ρs
        fit = curve_fit(m, 1 ./ θs, ρs, [4, -0.2])
        scatter!(ax, 1 ./ θs, ρs, label = "$(n_ε[i])")
        lines!(ax, domain, m(domain, fit.param))
        ρ∞[i] = fit.param[1]
    end
    axislegend(ax, L"n_\theta")

    ε_fit = curve_fit(m, 1 ./ n_ε, ρ∞, [5, -0.2])
    ax2 = Axis(f[1,2],
        xlabel = L"n_ε^{-1}",
        ylabel= L"\rho |_{n_\theta \to \infty} (\mathrm{\mu\Omega cm})",
        xlabelsize = 28,
        ylabelsize = 28
    )
    domain = 0.0:0.01:0.15
    scatter!(ax2, 1 ./ n_ε, ρ∞, color = Makie.wong_colors()[1:length(ρ∞)])
    lines!(ax2, domain, m(domain, ε_fit.param))
    display(f)
    @show ε_fit.param

    # outfile = joinpath(@__DIR__, "..", "plots", "interband_convergence.png")
    # save(outfile, f)
end

function get_property(prop::String, T::Float64, n_ε::Int, n_θ::Int, Uee::Real, Vimp::Real; include_impurity::Bool = true)
    eefile = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "Sr2RuO4_$(T)_$(n_ε)x$(n_θ).h5")
    
    Γ, k, v, E, dV, _, _= load(eefile)
    Γ *= Uee^2

    if include_impurity
        impfile = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "Sr2RuO4_imp_$(T)_$(n_ε)x$(n_θ).h5")
        Γimp, _, _, _, _, _, _ = load(impfile)
        Γ += Γimp * Vimp^2
    end

    T = kb*T

    if prop == "σ"
        σ = Ludwig.conductivity(Γ, v, E, dV, T)
        return σ[1,1] / c
    elseif prop == "η"
        return Ludwig.viscosity(Γ, k, E, dV, dxx, dyy, dxy, T) ./ (a^2 * c)
    else
        return nothing
    end
end

function display_heatmap(file, T)
    Γ, k, v, E, dV, corners, corner_ids = load(file)
    fd = f0.(E, T) # Fermi dirac on grid points

    w = fd .* (1 .- fd) # Energy derivative of FD on grid points
    M = diagm(sqrt.(w .* dV)) * Γ * (diagm(1 ./ sqrt.(w .* dV)))
    ℓ = size(Γ)[1]
    f = Figure(size = (1000, 1000))
    ax = Axis(f[1,1], aspect = 1.0)
    h = heatmap!(ax, -M, colorrange = (-0.002, 0.002), colormap = :lisbon)
    # Colorbar(f[1,2], h)
    display(f)
    # save(joinpath(@__DIR__,"..", "plots","Sr2RuO4_interband_heatmap.png"), f)
end

function ρ_fit_with_impurities(n_ε, n_θ)
    # Fit to Lupien's data
    lupien_file = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "rhovT_Lupien_digitized.dat")
    lupien_data = readdlm(lupien_file)
    
    lupien_T = lupien_data[11:60,1]
    lupien_ρ = lupien_data[11:60,2]

    lupien_model(t, p) = p[1] .+ p[2] * t.^p[3]
    lupien_fit = curve_fit(lupien_model, lupien_T, lupien_ρ, [0.12, 0.003, 1.9])
    @show lupien_fit.param 

    temps = 2.0:0.5:14.0
    lupien_ρ = lupien_model(temps, lupien_fit.param)
    @show lupien_ρ

    Vfit_model(t, p) = begin 
        @show p
        σ = Vector{Float64}(undef, length(t))
        for i in eachindex(t)
            eefile = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "Sr2RuO4_$(t[i])_$(n_ε)x$(n_θ).h5")
            impfile = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "Sr2RuO4_imp_$(t[i])_$(n_ε)x$(n_θ).h5")
            Γee, k, v, E, dV, _, _= load(eefile)
            Γimp, _, _, _, _, _, _ = load(impfile)

            σ[i] = real(Ludwig.conductivity(p[1]^2 * Γee + p[2]^2 * Γimp, v, E, dV, kb * t[i])[1,1])
        end

        ρ = (c ./ σ) * 10^8 # Convert to μΩ-cm
        @show ρ

        χ = sqrt(sum((ρ .- lupien_ρ).^2) / length(t))
        @show χ
        ρ
    end

    Uee = 0.033
    Vimp = 4e-4

    guess = [Uee, Vimp]
    Vfit = curve_fit(Vfit_model, temps, lupien_ρ, guess, lower = [0.0, 0.0])
    @show Vfit.param
end

function plot_ρ(n_ε, n_θ)
    t = 2.0:0.5:14.0
    Uee = 0.033381071857926596
    Vimp = 0.0003953923115459219

    σ = Vector{Float64}(undef, length(t))
    for i in eachindex(t)
        eefile = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "Sr2RuO4_$(t[i])_$(n_ε)x$(n_θ).h5")
        impfile = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "Sr2RuO4_imp_$(t[i])_$(n_ε)x$(n_θ).h5")
        Γee, k, v, E, dV, _, _= load(eefile)
        Γimp, _, _, _, _, _, _ = load(impfile)

        σ[i] = real(Ludwig.conductivity(Uee^2 * Γee + Vimp^2 * Γimp, v, E, dV, kb * t[i])[1,1])
    end

    ρ = (c ./ σ) * 10^8 # Convert to μΩ-cm

    model(t, p) = p[1] .+ p[2] * t.^p[3]
    fit = curve_fit(model, t, ρ, [0.12, 0.1, 2.0], lower = [0.0, 0.0, 0.0])
    @show fit.param


    f = Figure()
    ax = Axis(f[1,1])
    # lines!(ax, t, ρ)
    lupien_file = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "rhovT_Lupien_digitized.dat")
    lupien_data = readdlm(lupien_file)
    scatter!(ax, lupien_data[:, 1], lupien_data[:, 2], color = :black)
    domain = 0.0:0.1:30.0
    lines!(ax, domain, model(domain, fit.param))
    display(f)

end

function ρ_matthiesen_fit()
    data_file = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "rho_ee.dat")

    Ts = Float64[]
    ρs = Float64[]
    open(data_file, "r") do file
        for line in readlines(file)
            startswith(line, "#") && continue
            T, ρ = split(line, " ")
            push!(Ts, parse(Float64, T))
            push!(ρs, parse(Float64, ρ))
        end
    end

    model(t, p) = p[1] * t.^p[2]
    fit = curve_fit(model, Ts, ρs, [1e-3, 2.0])
    @show fit.param[2]

    # Fit to Lupien's data
    lupien_file = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "rhovT_Lupien_digitized.dat")
    lupien_data = readdlm(lupien_file)
    
    lupien_T = lupien_data[11:60,1]
    lupien_ρ = lupien_data[11:60,2]

    lupien_model(t, p) = p[1] .+ p[2] * t.^fit.param[2]
    lupien_fit = curve_fit(lupien_model, lupien_T, lupien_ρ, [0.1, 1.0])
    @show lupien_fit.param 

    scale = lupien_fit.param[2] / fit.param[1]

    Uee = sqrt(scale) * 0.1
    @show Uee

    f = Figure(fontsize = 20)
    ax = Axis(f[1,1],
              xlabel = L"T(\mathrm{K})",
              ylabel = L"\rho (\mathrm{\mu\Omega \,cm})")
    scatter!(ax, lupien_data[:,1], lupien_data[:,2], color = :black)
    domain = 0.0:0.1:30
    lines!(ax, domain, lupien_fit.param[1] .+ scale * model(domain, fit.param), color = :red)
    display(f)

    # outfile = joinpath(@__DIR__, "..", "plots", "test.png")
    # save(outfile, f)
end

function main(n_ε, n_θ)
    Uee = 0.033381071857926596
    Vimp = 0.0003953923115459219

    for T in 2.0:14.0
        @show get_property("η", T, n_ε, n_θ, Uee, Vimp)
    end

end

main(12, 38)


# open(joinpath(@__DIR__, "..", "data", "rho_ee.dat"), "w") do file
#     write(file, "#T(K) ρ(μΩ-cm)")
#     for T in 2.0:0.5:14.0
#         ρ = real(main(T, n_ε, n_θ, bands))
#         write(file, "\n$(T) $(ρ)")
#     end
# end