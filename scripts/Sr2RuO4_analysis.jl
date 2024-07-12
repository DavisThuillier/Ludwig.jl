using Ludwig
using HDF5
using StaticArrays
using CairoMakie, LaTeXStrings, Colors
using LinearAlgebra
using LsqFit
using DelimitedFiles
using ProgressBars

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))
data_dir = joinpath(@__DIR__, "..", "data", "Sr2RuO4")
plot_dir = joinpath(@__DIR__, "..", "plots", "Sr2RuO4")

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
    eefile = joinpath(data_dir, "Sr2RuO4_$(T)_$(n_ε)x$(n_θ).h5")
    
    Γ, k, v, E, dV, _, _= load(eefile)
    Γ *= Uee^2
    ℓ = size(Γ)[1]

    if include_impurity
        impfile = joinpath(data_dir, "Sr2RuO4_unitary_imp_$(T)_$(n_ε)x$(n_θ).h5")
        Γimp, _, _, _, _, _, _ = load(impfile)
        Γ += Γimp * Vimp^2
    end

    T = kb*T

    if prop == "σ"
        σ = Ludwig.conductivity(Γ, v, E, dV, T)
        return σ[1,1] / c
    elseif prop == "ηB1g"
        Dxx = zeros(Float64, ℓ)
        Dyy = zeros(Float64, ℓ)
        for i in 1:ℓ
            μ = (i-1)÷(ℓ÷3) + 1 # Band index
            if μ < 3 # α and β bands
                Dxx[i] = dii_μ(k[i], 1, μ)
                Dyy[i] = dii_μ(k[i], 2, μ)
            else
                Dxx[i] = dxx_γ(k[i])
                Dyy[i] = dyy_γ(k[i])
            end
        end

        return Ludwig.ηB1g(Γ, E, dV, Dxx, Dyy, T) ./ (a^2 * c)
    elseif prop == "τ_eff"
        # Effective lifetime as calculated from the conductivity
        τ = Ludwig.σ_lifetime(Γ, v, E, dV, T)
        return hbar * τ
    elseif prop == "spectrum"
        return Ludwig.spectrum(Γ, E, dV, T) ./ hbar
    else
        return nothing
    end
end

function spectrum(n_ε, n_θ, Uee, Vimp)
    T = 2.0:0.5:14.0
    ℓ = 12 * (n_ε - 1) * (n_θ - 1) # Dimension of matrix

    info_file = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "spectrum.aux")
    open(info_file, "w") do file
        write(file, "#About spectrum.csv")
        write(file, "\n#Uee = $(Uee) eV")
        write(file, "\n#n_ε = $(n_ε), n_θ = $(n_θ)")
        write(file, "\n#T(K) is row index\n")
        writedlm(file, T', ",")
    end

    return nothing

    spectrum = Matrix{Float64}(undef, length(T), ℓ)
    for i in ProgressBar(eachindex(T))
        spectrum[i, :] .= real.(get_property("spectrum", T[i], n_ε, n_θ, Uee, Vimp))
    end

    file = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "spectrum.csv")
    writedlm(file, spectrum, ",")

end

function τ_eff(n_ε, n_θ, Uee, Vimp)
    T = 2.0:0.5:14.0

    file = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "τ_eff.dat")
    open(file, "w") do f
        write(f, "#τ_eff from conductivity")
        write(f, "\n#n_ε = $(n_ε), n_θ = $(n_θ)")
        write(f, "\n#Uee = $(Uee) eV")
        write(f, "\n#√n * Vimp = $(Vimp) eV m^{-3/2}")
        write(f, "\n#T(K) τ_eff (ps)")
        for i in eachindex(T)
            τ = get_property("τ_eff", T[i], n_ε, n_θ, Uee, Vimp, include_impurity = false) * 1e12
            write(f, "\n$(T[i]) $(τ)")
        end
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

function modes(T, n_ε, n_θ, Uee, bands)
    file = joinpath(data_dir, "Sr2RuO4_$(T)_$(n_ε)x$(n_θ).h5")

    Γ, k, v, E, dV, corners, corner_ids = load(file)
    Γ *= Uee^2
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd) # Energy derivative of FD on grid points

    M = diagm(sqrt.(w .* dV)) * Γ * (diagm(1 ./ sqrt.(w .* dV)))

    quads = Vector{Vector{SVector{2, Float64}}}(undef, 0)
    for i in 1:size(corner_ids)[1]
        push!(quads, map(x -> SVector{2}(corners[x, :]), corner_ids[i, :]))
    end

    N = 20
    
    @time eigenvalues  = eigvals(M)
    @time eigenvectors = eigvecs(M)

    F = eigen(M)
    eigenvalues *= 1e-12 / hbar
    
    colors = Vector{RGB{Float64}}(undef, length(eigenvalues))
    for i in eachindex(eigenvalues)
        # println("λ_$(i) = ", eigenvalues[i])
        weights = vcat(Ludwig.band_weight(eigenvectors[:, i], length(bands)))
        # for j in eachindex(bands)
            # println("   ", bands[j], ": ", weights[j])
        # end
        colors[i] = RGB{Float64}(weights...)

        if i < N
            f = Figure(size = (1000,1000), fontsize = 30)
            ax  = Axis(f[1,1], aspect = 1.0, title = latexstring("\$ \\tau_{$(i -1)} = $(round(real(eigenvalues[i]), digits = 6)) \\text{ fs}\$"), limits = (-0.5, 0.5, -0.5, 0.5))
            p = poly!(ax, quads, color = real.(eigenvectors[:, i]), colormap = :viridis, 
            )
            Colorbar(f[1,2], p)
            display(f)
        end
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

    outfile = joinpath(plot_dir, "Sr2RuO4_spectrum_αβγ.png")
    # save(outfile, f)
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

    Vfit_model(t, p) = begin 
        @show p
        σ = Vector{Float64}(undef, length(t))
        for i in eachindex(t)
            σ[i] = get_property("σ", t[i], n_ε, n_θ, p[1], p[2])
            @show σ[i]
        end

        ρ = 1e8 ./ σ # Convert to μΩ-cm

        χ = sqrt(sum((ρ .- lupien_ρ).^2) / length(t))
        @show χ
        ρ
    end

    Uee = 0.033
    Vimp = 4e-6

    guess = [Uee, Vimp]
    Vfit = curve_fit(Vfit_model, temps, lupien_ρ, guess, lower = [0.0, 0.0])
    @show Vfit.param
end

function plot_ρ(n_ε, n_θ, Uee, Vimp)
    t = 2.0:0.5:14.0

    # σ = Vector{Float64}(undef, length(t))
    # for i in eachindex(t)
    #     σ[i] = get_property("σ", t[i], n_ε, n_θ, Uee, Vimp)
    # end

    # ρ = 10^8  ./ σ # Convert to μΩ-cm
    # @show ρ

    ρ = [0.1150196207561388, 0.12451619211769277, 0.1375536322369924, 0.151331465720305, 0.16678337503360013, 0.18390660051654195, 0.20259655450950417, 0.22294204160296502, 0.24481574576635556, 0.26880111695227943, 0.2943530798780751, 0.3221427793466389, 0.35144701512387355, 0.3830058933824082, 0.41646153824899235, 0.4516984084562476, 0.4888882328571111, 0.5280041011251837, 0.5690052123933946, 0.6120857271933049, 0.6576240047597802, 0.7049349407316307, 0.7544758877211121, 0.8067026603183666, 0.8604543201283871]

    model(t, p) = p[1] .+ p[2] * t.^p[3]
    fit = curve_fit(model, t, ρ, [0.12, 0.1, 2.0], lower = [0.0, 0.0, 0.0])
    @show fit.param


    f = Figure(fontsize = 20)
    ax = Axis(f[1,1],
              xlabel = L"T(\mathrm{K})",
              ylabel = L"\rho (\mathrm{\mu\Omega \,cm})")
    # lines!(ax, t, ρ)
    lupien_file = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "rhovT_Lupien_digitized.dat")
    lupien_data = readdlm(lupien_file)
    scatter!(ax, lupien_data[:, 1], lupien_data[:, 2], color = :black)
    domain = 0.0:0.1:30.0
    lines!(ax, domain, model(domain, fit.param))
    display(f)

    outfile = joinpath(plot_dir, "ρ_unitary_scattering.png")
    save(outfile, f)

end

function plot_η()
    data_file = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "viscosity_unitary.dat")

    T = Float64[]
    ηB1g = Float64[]
    open(data_file, "r") do file
        for line in readlines(file)
            startswith(line, "#") && continue
            t, η1 = split(line, " ")
            push!(T, parse(Float64, t))
            push!(ηB1g, parse(Float64, η1))
        end
    end

    # Fit to Brad's data
    visc_file = joinpath(@__DIR__, "..", "data", "Sr2RuO4","B1gViscvT_new.dat")
    data = readdlm(visc_file)
    rmodel(t, p) = p[1] .+ 1 ./ (p[2] .+ p[3] * t.^p[4])
    rfit = curve_fit(rmodel, data[1:341, 1], data[1:341, 2], [0.1866, 1.2, 0.1, 2.0])
    @show rfit.param
    @show rfit.param[2] / rfit.param[3]
    data[:, 2] .-= rfit.param[1] # Subtract off systematic offset


    model(t, p) = 1 ./ (p[1] .+ p[2] * t.^p[3])
    fit = curve_fit(model, T, ηB1g, [1.0, 0.1, 2.0])
    @show fit.param
    @show fit.param[1] / fit.param[2]

    f = Figure()
    ax = Axis(f[1,1], 
            aspect = 1.0,
            ylabel = L"\eta^{-1} ((\mathrm{Pa\,s})^{-1})",
            xlabel = L"T^2(\mathrm{K^2})",
            xticks = [0.0, 36, 81, 144, 225, 324],
            xtickformat = values -> [L"%$(Int(sqrt(x)))^2" for x in values]
            )
            xlims!(0.0, 14^2)
    ylims!(ax, 0.0, 40)
    xlims!(ax, 0.0, 14^2)
    
    domain = 0.0:0.1:14.0
    scatter!(ax, data[:, 1].^2, 1 ./ data[:, 2],color = :black)
    lines!(ax, domain.^2, 1 ./ (rmodel(domain, rfit.param) .- rfit.param[1]))
    lines!(ax, domain.^2, 1 ./ (model(domain, fit.param)), color = :red)
    scatter!(ax, T.^2, 1 ./ ηB1g)
    
    ax2 = Axis(f[1,2],
        aspect = 1.0,
        ylabel = L"\eta_\text{exp} / \eta_\text{theory} ",
            xlabel = L"T(\mathrm{K})",
    )
    lines!(ax2, domain, (rfit.param[3]/fit.param[2]) * (rmodel(domain, rfit.param) .- rfit.param[1]) ./ (model(domain, fit.param)))

    display(f)
    outfile = joinpath(@__DIR__, "..", "plots", "η_unitary.png")
    save(outfile, f)

    return nothing

    f = Figure()
    ax = Axis(f[1,1], 
            aspect = 1.0,
            ylabel = L"\eta^{-1} ((\mathrm{Pa\,s})^{-1})",
            xlabel = L"T^2(\mathrm{K^2})"
            )
            xlims!(0.0, 14^2)
    ylims!(ax, 0.0, 1)
    xlims!(ax, 0.0, 14)
    
    domain = 0.0:0.1:14.0
    scatter!(ax, data[:, 1], data[:, 2],color = :black)
    lines!(ax, domain, (rmodel(domain, rfit.param) .- rfit.param[1]))
    lines!(ax, domain, model(domain, fit.param), color = :red)
    # scatter!(ax, T.^2, 1 ./ ηB1g)
    display(f)

end

function plot_spectrum(n_ε, n_θ, Uee, Vimp)
    τ_eff_file = joinpath(data_dir, "τ_eff.dat")
   
    T = Float64[]
    τ_eff = Float64[]
    open(τ_eff_file, "r") do file
        for line in readlines(file)
            startswith(line, "#") && continue
            t, τ = split(line, " ")
            push!(T, parse(Float64, t))
            push!(τ_eff, parse(Float64, τ))
        end
    end

    model(t,p) = p[1] .+ p[2] * t.^p[3]
    fit = curve_fit(model, T, τ_eff.^-1, [0.0, 1.0, 2.0], lower = [0.0, 0.0, 1.0])
    @show fit.param

    f = Figure(fontsize = 20)
    ax = Axis(f[1,1], 
            xlabel = L"T^2(K^2)", 
            ylabel = L"\tau^{-1}_\mathrm{eff}\, (\mathrm{ps}^{-1})")
    # scatter!(ax, T.^2, 1 ./ τ_eff)
    domain = 0.0:0.1:14
    
    
    # Plot spectrum from file
    aux_file = joinpath(data_dir, "spectrum.aux")
    spectrum_T = vec(readdlm(aux_file, ',', Float64, comments = true, comment_char = '#'))

    N = 4000
    spectrum_file = joinpath(data_dir, "spectrum.csv")
    spectrum = readdlm(spectrum_file, ',', Float64)[:, 1:N]

    for i in 1:N
        scatter!(ax, spectrum_T.^2, 1e-12 * spectrum[:, i], color = :black)
        mode_fit = curve_fit(model, spectrum_T, 1e-12 * spectrum[:, i], [0.0, 1e-3, 2.0], lower = [0.0, 0.0, 0.0])
        @show mode_fit.param
    end

    lines!(ax, domain.^2, model(domain, fit.param), color = :red)
    display(f)
   

    outfile = joinpath(plot_dir, "τ_eff.png")
    save(outfile, f)
end

function main(n_ε, n_θ)    
    # Unitary
    Uee = 0.03295629587089841
    Vimp = 7.560563025815616e-5

    # τ_eff(n_ε, n_θ, Uee, Vimp)
    # plot_spectrum(n_ε, n_θ, Uee, Vimp)

    # modes(12.0, n_ε, n_θ, Uee, bands)

    plot_η()
    # open(joinpath(data_dir, "viscosity_unitary.dat"), "w") do file
    #     write(file, "#Uee = $(Uee) eV")
    #     write(file, "\n#√n = $(Vimp) m^{-3/2}")
    #     write(file, "\n#T(K) ηB1g(Pa-s)")
    #     for T in 2.0:0.5:14.0
    #         ηB1g = get_property("ηB1g", T, n_ε, n_θ, Uee, Vimp)
    #         @show ηB1g
    #         write(file, "\n$(T) $(ηB1g)")
    #     end
    # end

end

main(12, 38)


# open(joinpath(@__DIR__, "..", "data", "rho_ee.dat"), "w") do file
#     write(file, "#T(K) ρ(μΩ-cm)")
#     for T in 2.0:0.5:14.0
#         ρ = real(main(T, n_ε, n_θ, bands))
#         write(file, "\n$(T) $(ρ)")
#     end
# end