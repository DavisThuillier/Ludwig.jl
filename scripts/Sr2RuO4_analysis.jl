using Ludwig
using HDF5
using StaticArrays
using CairoMakie, LaTeXStrings, Colors
using LinearAlgebra
using LsqFit
using DelimitedFiles
using ProgressBars

function symmetrize(L, dV, E, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    D = diagm(dV .* fd .* (1 .- fd))

    return 0.5 * (L + inv(D) * L' * D)
end

function load(file, T)
    h5open(file, "r") do fid
        g = fid["data"]

        L = 2pi * read(g, "L") 
        k = reinterpret(SVector{2,Float64}, vec(transpose(read(g, "momenta"))))
        v = reinterpret(SVector{2,Float64}, vec(transpose(read(g, "velocities"))))
        E = read(g, "energies")
        dV = read(g, "dVs")
        corners = read(g, "corners")
        corner_ids = read(g, "corner_ids")

        L = symmetrize(L, dV, E, kb * T)

        # Enforce particle conservation
        for i in 1:size(L)[1]
            L[i,i] -= sum(L[i, :])
        end

        return L, k, v, E, dV, corners, corner_ids
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

function get_property(prop::String, T, n_ε::Int, n_θ::Int, Uee::Real, Vimp::Real; include_impurity::Bool = true)
    eefile = joinpath(data_dir, "Sr2RuO4_$(Float64(T))_$(n_ε)x$(n_θ).h5")
    
    Γ, k, v, E, dV, _, _= load(eefile, T)
    Γ *= 0.5 * Uee^2
    ℓ = size(Γ)[1]

    if include_impurity
        impfile = joinpath(data_dir, "Sr2RuO4_unitary_imp_$(T)_$(n_ε)x$(n_θ).h5")
        Γimp, _, _, _, _, _, _ = load(impfile, T)
        Γ += Γimp * Vimp^2
    end

    T = kb*T

    if prop == "σ"
        σ = Ludwig.conductivity(Γ, v, E, dV, T)
        return σ[1,1] / c
    elseif prop == "σxx"
        σ = Ludwig.longitudinal_conductivity(Γ, first.(v), E, dV, T)
        return σ[1,1] / c
    elseif prop == "ηB1g"
        Dxx = zeros(Float64, ℓ)
        Dyy = zeros(Float64, ℓ)
        for i in 1:ℓ
            μ = (i-1)÷(ℓ÷3) + 1 # Band index
            Dxx[i] = dii_μ(k[i], 1, μ)
            Dyy[i] = dii_μ(k[i], 2, μ)
        end

        return Ludwig.ηB1g(Γ, E, dV, Dxx, Dyy, T) / (a^2 * c)
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
    # T = 2.0:0.5:14.0
    T = 2.0:0.5:14.0
    ℓ = 12 * (n_ε - 1) * (n_θ - 1) # Dimension of matrix

    info_file = joinpath(data_dir, "spectrum.aux")
    open(info_file, "w") do file
        write(file, "#About spectrum.csv")
        write(file, "\n#Uee = $(Uee) eV")
        write(file, "\n#n_ε = $(n_ε), n_θ = $(n_θ)")
        write(file, "\n#T(K) is row index\n")
        writedlm(file, T', ",")
    end

    # return nothing

    spectrum = Matrix{Float64}(undef, length(T), ℓ)
    for i in ProgressBar(eachindex(T))
        spectrum[i, :] .= real.(get_property("spectrum", T[i], n_ε, n_θ, Uee, Vimp; include_impurity = false))
    end

    file = joinpath(data_dir, "spectrum.csv")
    writedlm(file, spectrum, ",")

end

function τ_eff(n_ε, n_θ, Uee, Vimp)
    T = 2.0:0.5:14.0

    file = joinpath(data_dir, "τ_eff.dat")
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
    Γ, k, v, E, dV, corners, corner_ids = load(file, T)

    T = kb * T


    fd = f0.(E, T) # Fermi dirac on grid points
    D = diagm(sqrt.(dV .* fd .* (1 .- fd)))
    M = D * Γ * inv(D)

    ℓ = size(Γ)[1]
    f = Figure(size = (2000, 2000))
    ax = Axis(f[1,1], aspect = 1.0)
    b = 6e-4
    h = heatmap!(ax, -M, colormap = :lisbon, colorrange = (-b, b))
    # h = heatmap!(ax, abs.(M-M') / norm(M), colormap = :lisbon) #, colorrange = (-b, b))
    Colorbar(f[1,2], h)
    display(f)
    # save(joinpath(@__DIR__,"..", "plots","Sr2RuO4_interband_heatmap.png"), f)

    # λ = eigvals(M)
    # @show λ[1:4]
    # f = Figure()
    # ax = Axis(f[1,1])
    # scatter!(ax, real.(λ))
    # display(f)
end

function modes(T, n_ε, n_θ, Uee)
    file = joinpath(data_dir, "Sr2RuO4_$(T)_$(n_ε)x$(n_θ).h5")
    T *= kb

    Γ, k, v, E, dV, corners, corner_ids = load(file, T)
    # Γ *= Uee^2 
    fd = f0.(E, T) # Fermi dirac on grid points

    D = diagm(sqrt.(dV .* fd .* (1 .- fd)))
    M = D * Γ * inv(D)

    eigenvalues  = eigvals(M)
    @show eigenvalues[1:4]

    quads = Vector{Vector{SVector{2, Float64}}}(undef, 0)
    for i in 1:size(corner_ids)[1]
        push!(quads, map(x -> SVector{2}(corners[x, :]), corner_ids[i, :]))
    end

    N = 10
    
    @time eigenvalues  = eigvals(M)
    @time eigenvectors = eigvecs(Γ)

    eigenvalues *= 1e-12 / hbar
    
    for i in eachindex(eigenvalues)
        if i < N
            println("λ_$(i) = ", eigenvalues[i])
            f = Figure(size = (1000,1000), fontsize = 30)
            ax  = Axis(f[1,1], aspect = 1.0, title = latexstring("\$ \\tau_{$(i -1)} = $(round(real(eigenvalues[i]), digits = 6)) \\text{ fs}\$"), limits = (-0.5, 0.5, -0.5, 0.5))
            p = poly!(ax, quads, color = real.(eigenvectors[:, i]), colormap = :viridis, #colorrange = (-10, 10)
            )
            Colorbar(f[1,2], p)
            display(f)
        end
    end

    outfile = joinpath(plot_dir, "Sr2RuO4_spectrum_αβγ.png")
    # save(outfile, f)
end

function ρ_fit_with_impurities(n_ε, n_θ)
    # Fit to Lupien's data
    lupien_file = joinpath(exp_dir, "rhovT_Lupien_digitized.dat")
    lupien_data = readdlm(lupien_file)
    
    lupien_T = lupien_data[11:60,1]
    lupien_ρ = lupien_data[11:60,2]

    lupien_model(t, p) = p[1] .+ p[2] * t.^p[3]
    lupien_fit = curve_fit(lupien_model, lupien_T, lupien_ρ, [0.12, 0.003, 1.9])
    @show lupien_fit.param 

    temps = 2.0:2.0:14.0
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

    Uee = 0.07517388226660576
    Vimp = 8.647920506354473e-5

    guess = [Uee, Vimp]
    Vfit = curve_fit(Vfit_model, temps, lupien_ρ, guess, lower = [0.0, 0.0])
    @show Vfit.param
end

function plot_ρ(n_ε, n_θ, Uee, Vimp)
    t = 2.0:0.5:12.0

    σ = Vector{Float64}(undef, length(t))
    for i in eachindex(t)
        σ[i] = get_property("σxx", t[i], n_ε, n_θ, Uee, Vimp; include_impurity = true)
        @show 10^8 / σ[i]
    end

    ρ = 10^8  ./ σ # Convert to μΩ-cm
    @show ρ

    ρ = [0.10959056505116326, 0.12151019463132146, 0.13504856338798898, 0.15016845154757302, 0.16672467041152575, 0.1847198405479588, 0.2041851433762483, 0.22516814159799386, 0.24777750498999843, 0.27197760942785343, 0.29784506878946493, 0.32543193395073594, 0.35473522979103506, 0.38579389146608156, 0.4186462443045558, 0.4534385594028692, 0.49009715381486624, 0.5286627463900118, 0.5691567274598581, 0.6116340508555088, 0.6561653206886255]

    model(t, p) = p[1] .+ p[2] * t.^p[3]
    fit = curve_fit(model, t, ρ, [0.0, 0.0001, 2.0], lower = [0.0, 0.0, 0.0])
    @show fit.param

    

    f = Figure(fontsize = 20)
    ax = Axis(f[1,1],
              xlabel = L"T(\mathrm{K})",
              ylabel = L"\rho (\mathrm{\mu\Omega \,cm})")
    # lines!(ax, t, ρ)
    lupien_file = joinpath(exp_dir, "rhovT_Lupien_digitized.dat")
    lupien_data = readdlm(lupien_file)
    scatter!(ax, lupien_data[:, 1], lupien_data[:, 2], color = :black)
    domain = 0.0:0.1:30.0
    scatter!(ax, t, ρ)
    lines!(ax, domain, model(domain, fit.param))
    display(f)

    # outfile = joinpath(plot_dir, "ρ_unitary_scattering.png")
    # save(outfile, f)

end

function get_η(n_ε, n_θ, Uee, Vimp)
    temps = 2.0:0.5:14.0
    data_file = joinpath(data_dir, "ηB1g.dat")

    # open(data_file, "w") do file
        for T in temps
            η = get_property("ηB1g", T, n_ε, n_θ, Uee, Vimp; include_impurity = true)
            # println(file, "$(T), $(η)")
            println("$(T), $(η)")
        end
    # end
end

function plot_η()
    data_file = joinpath(data_dir, "ηB1g.dat")

    T = Float64[]
    ηB1g = Float64[]
    open(data_file, "r") do file
        for line in readlines(file)
            startswith(line, "#") && continue
            t, η1 = split(line, ",")
            push!(T, parse(Float64, t))
            push!(ηB1g, parse(Float64, η1))
        end
    end
    ηB1g *= pi

    # Fit to Brad's data
    visc_file = joinpath(exp_dir,"B1gViscvT_new.dat")
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
    ylims!(ax, 0.0, 24)
    xlims!(ax, 0.0, 14^2)
    
    domain = 0.0:0.1:14.0
    scatter!(ax, data[:, 1].^2, 1 ./ data[:, 2],color = :black)
    lines!(ax, domain.^2, 1 ./ (rmodel(domain, rfit.param) .- rfit.param[1]))
    lines!(ax, domain.^2, 1 ./ (model(domain, fit.param)), color = :red)
    scatter!(ax, T.^2, 1 ./ ηB1g)


    display(f)
    # outfile = joinpath(@__DIR__, "..", "plots", "η_unitary.png")
    # save(outfile, f)

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
    # τ_eff_file = joinpath(data_dir, "τ_eff.dat")
   
    # T = Float64[]
    # τ_eff = Float64[]
    # open(τ_eff_file, "r") do file
    #     for line in readlines(file)
    #         startswith(line, "#") && continue
    #         t, τ = split(line, " ")
    #         push!(T, parse(Float64, t))
    #         push!(τ_eff, parse(Float64, τ))
    #     end
    # end

    model(t,p) = p[1] .+ p[2] * t.^p[3]
    # fit = curve_fit(model, T, τ_eff.^-1, [0.0, 1.0, 2.0], lower = [0.0, 0.0, 1.0])
    # @show fit.param

    f = Figure(fontsize = 20)
    ax = Axis(f[1,1], 
            xlabel = L"T^2(K^2)", 
            ylabel = L"\tau^{-1}_\mathrm{eff}\, (\mathrm{ps}^{-1})")
    # scatter!(ax, T.^2, 1 ./ τ_eff)
    domain = 0.0:0.1:14
    
    
    # Plot spectrum from file
    aux_file = joinpath(data_dir, "spectrum.aux")
    spectrum_T = vec(readdlm(aux_file, ',', Float64, comments = true, comment_char = '#'))

    N = 324
    spectrum_file = joinpath(data_dir, "spectrum.csv")
    spectrum = readdlm(spectrum_file, ',', Float64)[:, 1:N]

    for i in 1:N
        scatter!(ax, spectrum_T.^2, 1e-12 * spectrum[:, i], color = :black)
        # mode_fit = curve_fit(model, spectrum_T, 1e-12 * spectrum[:, i], [0.0, 1e-3, 2.0], lower = [0.0, 0.0, 0.0])
        # @show mode_fit.param
    end

    # lines!(ax, domain.^2, model(domain, fit.param), color = :red)
    display(f)
   

    # outfile = joinpath(plot_dir, "τ_eff.png")
    # save(outfile, f)
end

function main(n_ε, n_θ)    
    # Unitary
    Uee = 0.03295629587089841
    Vimp = 7.560563025815616e-5

    # τ_eff(n_ε, n_θ, Uee, Vimp)
    # plot_spectrum(n_ε, n_θ, Uee, Vimp)

    # modes(12.0, n_ε, n_θ, Uee, bands)

    # plot_η()
    # open(joinpath(data_dir, "viscosity_unitary.dat"), "w") do file
    #     write(file, "#Uee = $(Uee) eV")
    #     write(file, "\n#√n = $(Vimp) m^{-3/2}")
    #     write(file, "\n#T(K) ηB1g(Pa-s)")
        for T in 2.0:0.5:14.0
            ηB1g = get_property("ηB1g", T, n_ε, n_θ, Uee, Vimp)
            @show ηB1g
            # write(file, "\n$(T) $(ηB1g)")
        end
    # end

end

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))
data_dir = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "model_1")
plot_dir = joinpath(@__DIR__, "..", "plots", "Sr2RuO4")
exp_dir  = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "experiment")

n_ε = 12
n_θ = 38
Uee = 0.07517388226660576
Vimp = 8.647920506354473e-5

# display_heatmap(joinpath(data_dir, "Sr2RuO4_12.0_12x38.h5"), 12)
# ρ_fit_with_impurities(12, 38)
plot_ρ(12, 38, Uee, Vimp)
# modes(12.0, 10, 24, 1.0)
# plot_η()
# get_η(n_ε, n_θ, Uee, Vimp)
# @show get_property("ηB1g", 12.0, 12, 38, Uee, Vimp; include_impurity = true)