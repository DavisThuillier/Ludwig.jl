using Ludwig
using HDF5
using StaticArrays
using CairoMakie, LaTeXStrings, Colors

# Set default font to Computer Modern so figures have same font as paper
MT = Makie.MathTeXEngine
mt_fonts_dir = joinpath(dirname(pathof(MT)), "..", "assets", "fonts", "NewComputerModern")

set_theme!(fonts = (
    regular = joinpath(mt_fonts_dir, "NewCM10-Regular.otf"),
    bold = joinpath(mt_fonts_dir, "NewCM10-Bold.otf")
))


using LinearAlgebra
using LsqFit
using DelimitedFiles
import Statistics: mean

include("LudwigIO.jl")

##################
### Fit Models ###
##################

model_0(t, p) = p[1] .+ p[2] * t.^2
model_1(t, p) = p[1] .+ p[2] * t.^p[3]
model_2(t, p) = p[1] .+ 1 ./ (p[2] .+ p[3] * t.^2)
model_3(t, p) = p[1] .+ 1 ./ (p[2] .+ p[3] * t.^p[4])

##################################
### Property Wrapper Functions ###
##################################

function get_σ(L, k, v, E, dV, T; kwargs)
    σ = Ludwig.conductivity(L, v, E, dV, T)
    return σ / c
end

function get_σxx(L, k, v, E, dV, T; kwargs)
    σxx = Ludwig.longitudinal_conductivity(L, first.(v), E, dV, T)
    return σxx / c
end

function get_ρ(L, k, v, E, dV, T; kwargs)
    σxx = Ludwig.longitudinal_conductivity(L, first.(v), E, dV, T)
    return c / σxx
end

function get_ηB1g(L, k, v, E, dV, T; kwargs)
    ℓ = length(k)
    Dxx = zeros(Float64, ℓ)
    Dyy = zeros(Float64, ℓ)
    for i in eachindex(k)
        μ = (i-1)÷(ℓ÷3) + 1 # Band index
        Dxx[i] = dii_μ(k[i], 1, μ)
        Dyy[i] = dii_μ(k[i], 2, μ)
    end

    return Ludwig.ηB1g(L, E, dV, Dxx, Dyy, T) / (a^2 * c)
end

function get_ηB2g(L, k, v, E, dV, T; kwargs)
    ℓ = length(k)
    Dxy = zeros(Float64, ℓ)
    for i in eachindex(k)
        μ = (i-1)÷(ℓ÷3) + 1 # Band index
        Dxy[i] = dxy_μ(k[i], μ)
    end

    return Ludwig.ηB2g(L, E, dV, Dxy, T) / (a^2 * c)
end

function get_τeff_σ(L, k, v, E, dV, T; kwargs)
    return Ludwig.σ_lifetime(L, v, E, dV, T)
end

function get_τeff_η(L, k, v, E, dV, T; kwargs)
    ℓ = length(k)
    Dxx = zeros(Float64, ℓ)
    Dyy = zeros(Float64, ℓ)
    for i in eachindex(k)
        μ = (i-1)÷(ℓ÷3) + 1 # Band index
        Dxx[i] = dii_μ(k[i], 1, μ)
        Dyy[i] = dii_μ(k[i], 2, μ)
    end

    return Ludwig.η_lifetime(L, Dxx, Dyy, E, dV, T)
end

function get_γ_ηB1g(L, k, v, E, dV, T; kwargs)
    Dxx = dii_μ.(k, 1, 3)
    Dyy = dii_μ.(k, 2, 3)

    return Ludwig.ηB1g(L, E, dV, Dxx, Dyy, T) / (a^2 * c)
end

function get_γ_ηB2g(L, k, v, E, dV, T; kwargs)
    Dxy = dxy_μ.(k, 3)

    return Ludwig.ηB2g(L, E, dV, Dxy, T) / (a^2 * c)
end

function get_Rh(L, k, v, E, dV, T; kwargs)
    Ludwig.hall_coefficient(L, k, v, E, dV, T; kwargs)
end

###############################
### Visualization Utilities ###
###############################

function rational_format(x)
    x = Rational(x)
    if x == 0
        return L"0"
    elseif x == 1
        return L"\pi"
    elseif denominator(x) == 1
        return L"%$(numerator(x)) \pi"
    elseif numerator(x) == 1
        return L"\frac{\pi}{%$(denominator(x))}"
    elseif numerator(x) == -1
        return L"-\frac{\pi}{%$(denominator(x))}"
    else 
        return L"\frac{%$(numerator(x))}{%$(denominator(x))} \pi"
    end
end

function parity(v)
    ℓ = length(v) ÷ 3

    p = 0.0
    for μ in 1:3
        for i in 1:(ℓ ÷ 2)
            p += sign(v[ℓ * (μ - 1) + i] * v[ℓ * (μ - 1) + mod(i - 1 + ℓ÷2, ℓ) + 1])
        end
    end

    return p / (3*ℓ) 
end

function single_band_parity(v)
    ℓ = length(v)

    p = 0.0
    for i in 1:(ℓ ÷ 2)
        p += sign(real(v[i] * v[mod(i - 1 + ℓ÷2, ℓ) + 1]))
    end

    return 2 * p / ℓ 
end

function mirror_x(v, n_ε, n_θ)
    M = reshape(v, n_ε - 1, 4 * (n_θ - 1)) # Vector reshaped so that each row corresponds to an energy contour

    return vec(reverse(M, dims = 2))
end

function mirror_y(v, n_ε, n_θ)
    M = copy(reshape(v, n_ε - 1, 4 * (n_θ - 1))) # Vector reshaped so that each row corresponds to an energy contour

    display(M)
    M .= circshift(M, (0, -2))
    display(M)
    reverse!(M, dims = 2)
    display(M)
    M .= circshift(M, (0, 2))
    display(M)

    return vec(M)
end

function mirror_symmetrize(v1, v2, n_ε, n_θ)
    W = Matrix{Float64}(undef, 2, 2)
    W[1,1] = dot(v1, mirror_x(v1, n_ε, n_θ))
    W[1,2] = dot(v1, mirror_x(v2, n_ε, n_θ))
    W[2,1] = dot(v2, mirror_x(v1, n_ε, n_θ))
    W[2,2] = dot(v2, mirror_x(v2, n_ε, n_θ))

    weights = eigvecs(W)
    @show weights

    w1 = weights[1,1] * v1 + weights[2,1] * v2
    w2 = weights[1,2] * v1 + weights[2,2] * v2

    theta1 = mean(mod.(angle.(w1), pi))
    theta2 = mean(mod.(angle.(w2), pi))

    w1 = real.(exp(- im * theta1) * w1)
    w2 = real.(exp(- im * theta2) * w2)

    return w1, w2
end

function mirror_gamma_modes(T, n_ε, n_θ, Uee)
    file = joinpath(data_dir, "Sr2RuO4_γ_$(T)_$(n_ε)x$(n_θ).h5")

    L, k, v, E, dV, corners, corner_ids = load(file, T; symmetrized = true)
    L *= 0.5 * Uee^2

    eigenvectors = eigvecs(L)

    v1 = eigenvectors[:, 3]
    v2 = eigenvectors[:, 4]

    w1, w2 = mirror_symmetrize(v1, v2, n_ε, n_θ)

    quads = Vector{Vector{SVector{2, Float64}}}(undef, 0)
    for i in 1:size(corner_ids)[1]
        push!(quads, map(x -> SVector{2}(corners[x, :]), corner_ids[i, :]))
    end

    f = Figure(size = (1000,1000), fontsize = 30)
    ax  = Axis(f[1,1], aspect = 1.0, 
        # title = latexstring("\$ \\tau_{$(i -1)} = $(round(real(1 / eigenvalues[i]), digits = 6)) \\text{ ps}\$"), 
        limits = (-0.5, 0.5, -0.5, 0.5),
        xlabel = L"k_x",
        xticks = map(x -> (x // 4), -4:1:4),
        xtickformat = values -> rational_format.(values),
        ylabel = L"k_y",
        yticks = map(x -> (x // 4), -4:1:4),
        ytickformat = values -> rational_format.(values),
        xlabelsize = 30,
        ylabelsize = 30,
        # yautolimitmargin = (0.05f0, 0.1f0)
    )
    p = poly!(ax, quads, color = real.(w1) / maximum(abs.(w1)), colormap = :roma100, colorrange = (-1, 1)
    )

    Colorbar(f[1,2], p,) #label = L"\varepsilon - \mu \,(\text{eV})", labelsize = 30)
    display(f)
    outfile = joinpath(plot_dir, "23 August 2024", "Sr2RuO4_γ_12K_mode_3_4_mirror_symmetrized_1.png")
    # save(outfile, f)

    f = Figure(size = (1000,1000), fontsize = 30)
    ax  = Axis(f[1,1], aspect = 1.0, 
        # title = latexstring("\$ \\tau_{$(i -1)} = $(round(real(1 / eigenvalues[i]), digits = 6)) \\text{ ps}\$"), 
        limits = (-0.5, 0.5, -0.5, 0.5),
        xlabel = L"k_x",
        xticks = map(x -> (x // 4), -4:1:4),
        xtickformat = values -> rational_format.(values),
        ylabel = L"k_y",
        yticks = map(x -> (x // 4), -4:1:4),
        ytickformat = values -> rational_format.(values),
        xlabelsize = 30,
        ylabelsize = 30,
        # yautolimitmargin = (0.05f0, 0.1f0)
    )
    p = poly!(ax, quads, color = real.(w2) / maximum(abs.(w2)), colormap = :roma100, colorrange = (-1, 1)
    )

    Colorbar(f[1,2], p,) #label = L"\varepsilon - \mu \,(\text{eV})", labelsize = 30)
    display(f)
    outfile = joinpath(plot_dir, "23 August 2024", "Sr2RuO4_γ_12K_mode_3_4_mirror_symmetrized_2.png")
    # save(outfile, f)
end

#############
### Plots ###
#############

function display_heatmap(file, T)
    Γ, _, _, E, dV, _, _ = load(file, T)

    T = kb * T

    fd = f0.(E, T) # Fermi dirac on grid points
    D = diagm(sqrt.(dV .* fd .* (1 .- fd)))
    M = D * Γ * inv(D)

    f = Figure(size = (2000, 2000))
    ax = Axis(f[1,1], aspect = 1.0, 
    backgroundcolor = :transparent,
        leftspinevisible = false,
        rightspinevisible = false,
        bottomspinevisible = false,
        topspinevisible = false,
        xticklabelsvisible = false, 
        yticklabelsvisible = false,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
        xminorticksvisible = false,
        yminorticksvisible = false,
        xticksvisible = false,
        yticksvisible = false,
        xautolimitmargin = (0.0,0.0),
        yautolimitmargin = (0.0,0.0),
        yreversed = true
    )
    b = 6e-4
    h = heatmap!(ax, -M, colormap = :lisbon, colorrange = (-b, b))
    # h = heatmap!(ax, abs.(M-M') / norm(M), colormap = :lisbon) #, colorrange = (-b, b))
    # Colorbar(f[1,2], h)
    display(f)
    # save(joinpath(plot_dir,"23 August 2024","Sr2RuO4_interband_heatmap.png"), f)
end

function modes(T, n_ε, n_θ, Uee, Vimp; include_impurity = false)
    file = joinpath(data_dir, material*"_$(T)_$(n_ε)x$(n_θ).h5")

    L, k, v, E, dV, corners, corner_ids = load(file, T; symmetrized = true)
    L *= 0.5 * Uee^2

    if include_impurity
        impfile = joinpath(data_dir, material*"_imp_$(T)_$(n_ε)x$(n_θ).h5")
        Limp, _, _, _, _, _, _ = load(impfile, T)
        L += Limp * Vimp^2
    end

    T *= kb

    @time eigenvalues  = eigvals(L)
    eigenvalues *= 1e-9 / hbar

    @show eigenvalues[1:5]

    return nothing

    @time eigenvectors = eigvecs(L)
    parities = Vector{Float64}(undef, length(eigenvalues))
    for i in eachindex(eigenvalues)
        parities[i] = parity(eigenvectors[:, i])
    end

    # f = Figure(size = (700, 500), fontsize = 24)
    # ax = Axis(f[1,1],
    #         xlabel = "Mode Index",
    #         ylabel = L"\lambda \,(\mathrm{ps}^{-1})"
    # )
    # for i in 1:50
    #     if i  < 5
    #         scatter!(ax, i, eigenvalues[i], 
    #         color = :gray, 
    #         )
    #     else
    #         scatter!(ax, i, eigenvalues[i], 
    #         #color = parities[i] > 0 ? :orange : :blue, 
    #         )
    #     end
    # end

    # orange_marker = MarkerElement(color = :orange, marker = :circle, markersize = 15,
    # strokecolor = :orange)
    # blue_marker = MarkerElement(color = :blue, marker = :circle, markersize = 15,
    # strokecolor = :blue)
    # gray_marker = MarkerElement(color = :gray, marker = :circle, markersize = 15,
    # strokecolor = :gray)

    # axislegend(ax, [orange_marker, blue_marker, gray_marker], ["Even", "Odd", L"\tau \to \infty"], position = :rb)
    # display(f)

    # outfile = joinpath(plot_dir, "23 August 2024", "Sr2RuO4_spectrum_1_to_50.png")
    # save(outfile, f)

    return nothing
    
    quads = Vector{Vector{SVector{2, Float64}}}(undef, 0)
    for i in 1:size(corner_ids)[1]
        push!(quads, map(x -> SVector{2}(corners[x, :]), corner_ids[i, :]))
    end

    N = 5
    
    for i in eachindex(eigenvalues)
        if i < N
            println("λ_$(i) = ", eigenvalues[i])
            f = Figure(size = (1000,1000), fontsize = 30)
            ax  = Axis(f[1,1], aspect = 1.0, 
                # title = latexstring("\$ \\tau_{$(i -1)} = $(round(real(1 / eigenvalues[i]), digits = 6)) \\text{ ps}\$"), 
                limits = (-0.5, 0.5, -0.5, 0.5),
                xlabel = L"k_x",
                xticks = map(x -> (x // 4), -4:1:4),
                xtickformat = values -> rational_format.(values),
                ylabel = L"k_y",
                yticks = map(x -> (x // 4), -4:1:4),
                ytickformat = values -> rational_format.(values),
                xlabelsize = 30,
                ylabelsize = 30,
                # yautolimitmargin = (0.05f0, 0.1f0)
            )
            p = poly!(ax, quads, color = real.(eigenvectors[:, i]) / maximum(abs.(eigenvectors[:, i])), colormap = :berlin, colorrange = (-1, 1)
            )

            Colorbar(f[1,2], p,) #label = L"\varepsilon - \mu \,(\text{eV})", labelsize = 30)
            display(f)
            # outfile = joinpath(plot_dir, "23 August 2024", "Sr2RuO4_12K_mode_$(i).png")
            # save(outfile, f)
        end
    end

    outfile = joinpath(plot_dir, "Sr2RuO4_spectrum_αβγ.png")
    # save(outfile, f)
end

###
### γ mode in isolation
#######################

function γ_modes(T, n_ε, n_θ, Uee)
    file = joinpath(data_dir, "Sr2RuO4_γ_$(T)_$(n_ε)x$(n_θ).h5")

    L, k, v, E, dV, corners, corner_ids = load(file, T; symmetrized = true)
    L *= 0.5 * Uee^2

    T *= kb

    @time eigenvalues  = eigvals(L)
    eigenvalues *= 1e-12 / hbar
    @time eigenvectors = eigvecs(L)
    parities = Vector{Float64}(undef, length(eigenvalues))
    for i in eachindex(eigenvalues)
        parities[i] = single_band_parity(eigenvectors[:, i])
    end

    
    f = Figure(size = (700, 500), fontsize = 24)
    ax = Axis(f[1,1],
            xlabel = "Mode Index",
            ylabel = L"\lambda \,(\mathrm{ps}^{-1})"
    )
    for i in 1:50
        if i  == 1
            scatter!(ax, i, real(eigenvalues[i]), 
            color = :gray, 
            )
        elseif i ==2
            nothing
        else
            scatter!(ax, i, real(eigenvalues[i]), 
            color = parities[i] > 0 ? :orange : :blue, 
            )
        end
    end

    orange_marker = MarkerElement(color = :orange, marker = :circle, markersize = 15,
    strokecolor = :orange)
    blue_marker = MarkerElement(color = :blue, marker = :circle, markersize = 15,
    strokecolor = :blue)
    gray_marker = MarkerElement(color = :gray, marker = :circle, markersize = 15,
    strokecolor = :gray)

    axislegend(ax, [orange_marker, blue_marker, gray_marker], ["Even", "Odd", L"\tau \to \infty"], position = :rb)
    display(f)

    outfile = joinpath(plot_dir, "23 August 2024", "Sr2RuO4_γ_only_spectrum_1_to_50_energy_mode_subtracted.png")
    # save(outfile, f)

    # return nothing
    
    @time eigenvectors = eigvecs(L)
    # parities = Vector{Float64}(undef, length(eigenvalues))
    # for i in eachindex(eigenvalues)
    #     parities[i] = parity(eigenvectors[:, i])
    # end
    # @show parities


    eigenvalues *= 1e-12 / hbar


    quads = Vector{Vector{SVector{2, Float64}}}(undef, 0)
    for i in 1:size(corner_ids)[1]
        push!(quads, map(x -> SVector{2}(corners[x, :]), corner_ids[i, :]))
    end

    # N = 20
    
    for i in eachindex(eigenvalues)
        if i < 50
            println("λ_$(i) = ", eigenvalues[i])
            f = Figure(size = (1000,1000), fontsize = 30)
            ax  = Axis(f[1,1], aspect = 1.0, 
                # title = latexstring("\$ \\tau_{$(i -1)} = $(round(real(1 / eigenvalues[i]), digits = 6)) \\text{ ps}\$"), 
                limits = (-0.5, 0.5, -0.5, 0.5),
                xlabel = L"k_x",
                xticks = map(x -> (x // 4), -4:1:4),
                xtickformat = values -> rational_format.(values),
                ylabel = L"k_y",
                yticks = map(x -> (x // 4), -4:1:4),
                ytickformat = values -> rational_format.(values),
                xlabelsize = 30,
                ylabelsize = 30,
                # yautolimitmargin = (0.05f0, 0.1f0)
            )
            p = poly!(ax, quads, color = real.(eigenvectors[:, i]) / maximum(abs.(eigenvectors[:, i])), colormap = :roma100, colorrange = (-1, 1)
            )

            Colorbar(f[1,2], p,) #label = L"\varepsilon - \mu \,(\text{eV})", labelsize = 30)
            display(f)
            outfile = joinpath(plot_dir, "23 August 2024", "Sr2RuO4_γ_12K_mode_$(i).png")
            # save(outfile, f)
        end
    end

    outfile = joinpath(plot_dir, "Sr2RuO4_spectrum_αβγ.png")
    # save(outfile, f)
end

function γ_lifetimes(n_ε, n_θ, include_impurity)
    t = 2.0:0.5:14.0

    σ_τ = Vector{Float64}(undef, length(t))
    η_τ = Vector{Float64}(undef, length(t))
    for i in eachindex(t)
        println("T = $(t[i])")
        file = joinpath(data_dir, "Sr2RuO4_γ_$(t[i])_$(n_ε)x$(n_θ).h5")

        L, k, v, E, dV, corners, corner_ids = load(file, t[i]; symmetrized = true)
        ℓ = size(L)[1]
        L *= 0.5 * Uee^2

        σ_τ[i] = Ludwig.σ_lifetime(L, v, E, dV, kb * t[i])

        Dxx = zeros(Float64, ℓ)
        Dyy = zeros(Float64, ℓ)
        for i in 1:ℓ
            Dxx[i] = dii_μ(k[i], 1, 3, 0.0)
            Dyy[i] = dii_μ(k[i], 2, 3, 0.0)
        end
        η_τ[i] = Ludwig.η_lifetime(L, Dxx, Dyy, E, dV, kb * t[i])
        
    end
    @show σ_τ
    @show η_τ

    @show σ_τ ./ η_τ

    # With impurity 

    f = Figure(fontsize = 20)
    ax = Axis(f[1,1], ylabel = L"\tau_\text{eff}^{-1}\,(\mathrm{ps}^{-1})", xlabel = L"T\, (\mathrm{K})", 
    xticks = [4, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196],
                xtickformat = values -> [L"%$(Int(sqrt(x)))^2" for x in values])
                xlims!(ax, 0, 200)
    scatter!(ax, t.^2, 1e-12 ./ σ_τ, label = L"\tau_\sigma")
    scatter!(ax, t.^2, 1e-12 ./ η_τ, label = L"\tau_\eta")
    axislegend(ax, position = :lt)
    display(f)
    save(joinpath(plot_dir, "23 August 2024", "τ_eff_γ.png"),f)

end

function γ_deformation_overlap(n_ε, n_θ, Uee, Vimp, T; include_impurity = true) 
    eefile = joinpath(data_dir, material*"_$(Float64(T))_$(n_ε)x$(n_θ).h5")
    L, k, v, E, dV, _, _= LudwigIO.load(eefile, symmetrized = true)
    L *= 0.5 * Uee^2

    if Vimp != 0.0 && include_impurity
        impfile = joinpath(data_dir, material*"_imp_$(T)_$(n_ε)x$(n_θ).h5")
        Limp, _, _, _, _, _, _ = LudwigIO.load(impfile)
        L += Limp * Vimp^2
    end

    D = dii_μ.(k, 1, 3) .- dii_μ.(k, 2, 3)
    Dxy = dxy_μ.(k, 3)
    
    @time eigenvalues  = eigvals(L)
    eigenvalues *= 1e-12 / hbar
    @time eigenvectors = eigvecs(L)

    b1g_weight = Vector{Float64}(undef, length(k))
    b2g_weight = Vector{Float64}(undef, length(k))
    for i in eachindex(k)
        b1g_weight[i] = abs2(dot(eigenvectors[:, i], D))
        b2g_weight[i] = abs2(dot(eigenvectors[:, i], Dxy))
    end

    f = Figure()
    ax = Axis(f[1,1],
                xlabel = "Mode Index",
                ylabel = L"|\langle \phi | D \rangle |^2" 
    )
    scatter!(ax, b1g_weight, label = L"D_{xx} - D_{yy}")
    scatter!(ax, b2g_weight, label = L"D_{xy}")
    axislegend(ax)
    display(f)

    outfile = joinpath(plot_dir, "γ_B1g_B2g_overlaps.png")
    # save(outfile, f)
end 

#######################

function visualize_rows(rows, T, n_ε, n_θ)
    file = joinpath(data_dir, "Sr2RuO4_$(T)_$(n_ε)x$(n_θ).h5")

    L, k, v, E, dV, corners, corner_ids = load(file, T; symmetrized = true)

    quads = Vector{Vector{SVector{2, Float64}}}(undef, 0)
    for i in 1:size(corner_ids)[1]
        push!(quads, map(x -> SVector{2}(corners[x, :]), corner_ids[i, :]))
    end

    bound = 6e-4
    for i in rows
        f = Figure(size = (1000,1000), fontsize = 30)
        ax  = Axis(f[1,1], aspect = 1.0, 
            title = "Row $(i)", 
            limits = (-0.5, 0.5, -0.5, 0.5),
            xlabel = L"k_x",
            xticks = map(x -> (x // 4), -4:1:4),
            xtickformat = values -> rational_format.(values),
            ylabel = L"k_y",
            yticks = map(x -> (x // 4), -4:1:4),
            ytickformat = values -> rational_format.(values),
            xlabelsize = 30,
            ylabelsize = 30,
            # yautolimitmargin = (0.05f0, 0.1f0)
        )
        p = poly!(ax, quads, color = -L[i, :], colormap = :berlin, colorrange = (-bound, bound)
        )
        # @show maximum(abs.(L[i, :]))

        Colorbar(f[1,2], p) #label = L"\varepsilon - \mu \,(\text{eV})", labelsize = 30)
        display(f)
        save(joinpath(plot_dir, "23 August 2024", "L_row_$(i).png"), f)
    end 

end

visualize_rows(i::Int, T, n_ε, n_θ) = visualize_rows([i], T, n_ε, n_θ)

function separate_band_conductivities(n_ε, n_θ, Uee, Vimp)
    t = 2.0:0.5:14.0
    # σ_γ = Vector{Float64}(undef, length(t))
    # σ_αβ = Vector{Float64}(undef, length(t))

    # for (i,T) in enumerate(t)

    #     eefile = joinpath(data_dir, "Sr2RuO4_$(Float64(T))_$(n_ε)x$(n_θ).h5")
        
    #     Γ, k, v, E, dV, _, _= load(eefile, T, symmetrized = true)
    #     Γ *= 0.5 * Uee^2
    #     ℓ = size(Γ)[1]

    #     impfile = joinpath(data_dir, "Sr2RuO4_unitary_imp_$(T)_$(n_ε)x$(n_θ).h5")
    #     Γimp, _, _, _, _, _, _ = load(impfile, T)
    #     Γ += Γimp * Vimp^2

    #     T = kb*T

    #     v_αβ = zeros(Float64, ℓ)
    #     v_αβ[1:2*(ℓ÷3)] .= first.(v)[1:2*(ℓ÷3)]

    #     v_γ = zeros(Float64, ℓ)
    #     v_γ[2*(ℓ÷3) + 1: end] .= first.(v)[2*(ℓ÷3)+1:end]

        
    #     σ_αβ[i] = Ludwig.longitudinal_conductivity(Γ, v_αβ, E, dV, T) / c
    #     σ_γ[i] = Ludwig.longitudinal_conductivity(Γ, v_γ, E, dV, T) / c
    # end
    # @show σ_αβ
    # @show σ_γ

    σ_αβ = [3.485251623174754e8, 3.350498393897893e8, 3.202862026705056e8, 3.047088090749613e8, 2.8872385087608236e8, 2.7267799850090873e8, 2.568477279579944e8, 2.4144034241533476e8, 2.2658609712477714e8, 2.1243205120582095e8, 1.9902772929225042e8, 1.8639990204702818e8, 1.7457213568633828e8, 1.6352932768996656e8, 1.5324807371226996e8, 1.4366868979668513e8, 1.3477717890939176e8, 1.2652865107789516e8, 1.1888088147367662e8, 1.1179094727934167e8, 1.052123464105264e8, 9.91025905527583e7, 9.342433584219703e7, 8.817130861808705e7, 8.325071476242156e7]
    σ_γ = [3.3649312165033937e8, 2.9462607517610204e8, 2.56226259998439e8, 2.219849180600946e8, 1.9251216786805204e8, 1.6742496072802386e8, 1.4617108091441426e8, 1.2818244976761436e8, 1.1288689385708356e8, 9.991044474076752e7, 8.883932529838382e7, 7.93530986379384e7, 7.120170460874698e7, 6.416227833834111e7, 5.804692948255603e7, 5.268992422550955e7, 4.7999698813893124e7, 4.387441929634054e7, 4.023242058760622e7, 3.6998668113804646e7, 3.4114136277737245e7, 3.153048169503758e7, 2.9204638084954284e7, 2.7124330302763082e7, 2.5216218224687155e7]

    ρ = [0.10959056505116326, 0.12151019463132146, 0.13504856338798898, 0.15016845154757302, 0.16672467041152575, 0.1847198405479588, 0.2041851433762483, 0.22516814159799386, 0.24777750498999843, 0.27197760942785343, 0.29784506878946493, 0.32543193395073594, 0.35473522979103506, 0.38579389146608156, 0.4186462443045558, 0.4534385594028692, 0.49009715381486624, 0.5286627463900118, 0.5691567274598581, 0.6116340508555088, 0.6561653206886255]

    f = Figure()
    ax = Axis(f[1,1], xlabel = L"T(\mathrm{K})",
    ylabel = L"\sigma (\mathrm{MS \,m^{-1}})")

    scatter!(ax, 2.0:0.5:12.0, 1e-6 * σ_αβ[1:21], color = :green, label = L"\alpha,\beta")
    scatter!(ax, 2.0:0.5:12.0, 1e-6 * σ_γ[1:21], color = :blue, label = L"\gamma")
    scatter!(ax, 2.0:0.5:12.0, 1e2 ./ ρ, color = :black, label = L"\alpha, \beta, \gamma")
    axislegend(ax, position = :rt)

    display(f)

    # save(joinpath(plot_dir, "23 August 2024", "σ_band_contributions_with_total.png"),f)
end 

function fit_ρ(n_ε, n_θ, temps, model_1, Uee, Vimp)
    # Fit to Lupien's data
    lupien_file = joinpath(exp_dir, "rhovT_Lupien_digitized.dat")
    lupien_data = readdlm(lupien_file)
    
    lupien_T = lupien_data[11:40,1]
    lupien_ρ = lupien_data[11:40,2]

    lupien_fit = curve_fit(model_1, lupien_T, lupien_ρ, [0.12, 0.003, 1.9])
    @show lupien_fit.param 

    lupien_ρ = model_1(temps, lupien_fit.param)
    @show lupien_ρ

    Vfit_model(t, p) = begin 
        @show p
        ρ = Vector{Float64}(undef, length(t))
        for i in eachindex(t)
            ρ[i] = LudwigIO.get_property("ρ", data_dir, material, t[i], n_ε, n_θ, p[1], p[2]; imp_stem = impurity_stem) * 1e8
            @show ρ[i]
        end

        χ = sqrt(sum((ρ .- lupien_ρ).^2) / length(t))
        @show χ
        ρ
    end

    guess = [Uee, Vimp]
    Vfit = curve_fit(Vfit_model, temps, lupien_ρ, guess, lower = [0.0, 0.0])
    @show Vfit.param
end

function plot_ρ(n_ε, n_θ)
    t, ρ, _ = LudwigIO.read_property_from_file(joinpath(data_dir, "ρ_$(n_ε)x$(n_θ).dat"))  
    ρ *= 1e8

    lupien_file = joinpath(exp_dir, "rhovT_Lupien_digitized.dat")
    lupien_data = readdlm(lupien_file)

    model = model_1
    fit = curve_fit(model, t, ρ, [0.0, 0.0001, 2.0], lower = [0.0, 0.0, 0.0])
    @show fit.param

    l_model = model_1   
    lfit = curve_fit(l_model, lupien_data[10:40, 1], lupien_data[10:40, 2], [0.0, 0.0001, 2.0], lower = [0.0, 0.0, 0.0])
    @show lfit.param

    f = Figure(fontsize = 20)
    ax = Axis(f[1,1],
              xlabel = L"T(\mathrm{K})",
              ylabel = L"\rho (\mathrm{\mu\Omega \,cm})")

    xlims!(ax, 0.0, 14.0.^2)
    ylims!(ax, 0.0, 1.0)
    domain = 0.0:0.1:14.0

    scatter!(ax, lupien_data[:, 1].^2, lupien_data[:, 2], color = :black)
    scatter!(ax, t.^2, ρ, color = :red)
    lines!(ax, domain.^2, model(domain, fit.param), color = :red)
    # lines!(ax, domain, l_model(domain, lfit.param))
    display(f)

    outfile = joinpath(plot_dir, "ρ_with_fit.png")
    # save(outfile, f)
end

function plot_ρ_strain(n_ε, n_θ)

    

    strains = 0.0:-0.01:-0.03
    scaling = Vector{Float64}(undef, length(strains))
    for (i,ϵ) in enumerate(strains)
        if ϵ == 0.0
            file = "ρ_$(n_ε)x$(n_θ).dat"
        else
            file = "ρ_ϵ_$(ϵ)_$(n_ε)x$(n_θ).dat"
        end 
        t, ρ, _ = LudwigIO.read_property_from_file(joinpath(data_dir, file))  
        ρ *= 1e8

        model = model_1
        fit = curve_fit(model, t, ρ, [0.0, 0.0001, 2.0], lower = [0.0, 0.0, 0.0])
        scaling[i] = fit.param[3]

        # scatter!(ax, t, ρ)
        # lines!(ax, domain, model(domain, fit.param), label = latexstring("\$\\epsilon = $(ϵ)\$"))
    end

    f = Figure(fontsize = 20)
    ax = Axis(f[1,1],
              xlabel = L"\epsilon_{xx}",
              ylabel = L"d\ln(\rho_{xx} - ρ_0) / d\ln T")
    scatter!(ax, strains, scaling)
    display(f)

    outfile = joinpath(plot_dir, "ρ_strain.png")
    # save(outfile, f)
end

function plot_η(n_ε, n_θ)
    # Fit to Brad Ramshaw's viscosity data
    visc_file = joinpath(exp_dir,"B1gViscvT_new.dat")
    data = readdlm(visc_file)
    rfit = curve_fit(model_3, data[1:300, 1], data[1:300, 2], [0.1866, 1.2, 0.1, 2.0])
    data[:, 2] .-= rfit.param[1] # Subtract off systematic offset

    η_model(t, p) = p[1] .+ p[2] * t.^rfit.param[4]
    t1, η1, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "ηB1g_$(n_ε)x$(n_θ).dat"))  
    fit1 = curve_fit(model_1, t1, 1 ./ η1, [1.0, 0.1, 2.0])
    η0 = 1 / fit1.param[1]

    t2, η2, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "ηB2g_$(n_ε)x$(n_θ).dat"))  
    fit2 = curve_fit(model_1, t2, 1 ./ η2, [1.0, 0.1, 2.0])

    # ηB1g vs ηB2g
    f = Figure()
    ax = Axis(f[1,1], 
            aspect = 1.0,
            ylabel = L"\eta / \eta_\text{B1g}^{(0)}",
            xlabel = L"T(\mathrm{K^2})",
            )
    xlims!(ax, 0.0, 13)
    domain = 0.0:0.1:14.0
    lines!(ax, domain, η0^-1 ./ model_1(domain, fit1.param), label = L"\eta_\text{B1g}")
    scatter!(ax, t1, η1 / η0)
    lines!(ax, domain, η0^-1 ./ model_1(domain, fit2.param), label = L"\eta_\text{B2g}")
    scatter!(ax, t2, η2 / η0)
    axislegend(ax)

    display(f)
    
    outfile = joinpath(plot_dir, "ηB1g_vs_ηB2g.png")
    save(outfile, f)

    xticks = [4, 16, 36, 49, 64, 81, 100, 121, 144, 169, 196]
    domain = 0.0:0.1:14.0

    f = Figure(size = (1000, 600), fontsize = 24)
    ax = Axis(f[1,1],
               xlabel = L"T^2\,(\mathrm{K}^2)",
               ylabel = L"\eta_0 / \eta",
               xticks = xticks,
               xtickformat = values -> [L"%$(Int(sqrt(x)))^2" for x in values],
               yticks = vcat([1], 2:2:20),
               aspect = 1.0
    )
    xlims!(ax, 0.0, 14^2)
    ylims!(ax, 0.0, 18.0)
    scatter!(ax, data[:, 1].^2, rfit.param[2] ./ data[:, 2], color = :black)
    lines!(ax, domain.^2, model_1(domain, fit1.param) / fit1.param[1], color = :red)
    scatter!(ax, t1.^2, fit1.param[1]^(-1) ./ η1, color = :red)

    display(f)
    outfile = joinpath(plot_dir, "ηB1g.png")
    save(outfile, f)

end

function plot_ρ_and_η(n_ε, n_θ)
    T = 2.0:0.5:12.0
    
    xticks = [4, 16, 36, 49, 64, 81, 100, 121, 144, 169, 196]
    domain = 0.0:0.1:14.0

    f = Figure(size = (1000, 600), fontsize = 24)
    ax1 = Axis(f[1,1],
               xlabel = L"T^2\,(\mathrm{K}^2)",
               ylabel = L"\rho / \rho_0",
               xticks = xticks,
               xtickformat = values -> [L"%$(Int(sqrt(x)))^2" for x in values],
               yticks = vcat([1], 2:2:8),
               aspect = 0.9
    )
    xlims!(ax1, 0.0, 14^2)
    ylims!(ax1, 0.0, 8.0)

    lupien_file = joinpath(exp_dir, "rhovT_Lupien_digitized.dat")
    lupien_data = readdlm(lupien_file)
    l_model(t, p) = p[1] .+ p[2] * t.^2
    lfit = curve_fit(l_model, lupien_data[10:60, 1], lupien_data[10:60, 2], [0.0, 0.0001, 2.0], lower = [0.0, 0.0, 0.0])
    @show lfit.param[2] / lfit.param[1]
    
    ρ_fit = curve_fit(model_1, T, ρ, [1e-2, 0.1, 2.0])
    @show ρ_fit.param[2] / ρ_fit.param[1]

    scatter!(ax1, lupien_data[:, 1].^2, lupien_data[:, 2] / lfit.param[1], color = :black)

    # scatter!(ax1, t.^2, ρ)
    lines!(ax1, domain.^2, ρ_model(domain, ρ_fit.param) / ρ_fit.param[1], color = :red)
    scatter!(ax1, T.^2, ρ / ρ_fit.param[1], color = :red)

    ax2 = Axis(f[1,2],
               xlabel = L"T^2\,(\mathrm{K}^2)",
               ylabel = L"(\eta / \eta_0)^{-1}",
               xticks = xticks,
               xtickformat = values -> [L"%$(Int(sqrt(x)))^2" for x in values],
               aspect = 0.9,
               yticks = vcat([1], 5:5:20)
    )
    xlims!(ax2, 0.0, 14^2)
    ylims!(ax2, 0.0, 20.0)
    visc_file = joinpath(exp_dir,"B1gViscvT_new.dat")
    data = readdlm(visc_file)
    rmodel(t, p) = p[1] .+ 1 ./ (p[2] .+ p[3] * t.^p[4])
    bound = 300 # ~ 8 K
    rfit = curve_fit(rmodel, data[1:bound, 1], data[1:bound, 2], [0.1866, 1.2, 0.1, 2.0], lower = [0.0, 0.0, 0.0, 1.5])
    data[:, 2] .-= rfit.param[1]
    @show rfit.param[3] / rfit.param[2]
    @show rfit.param[4]

    η_model(t, p) = p[1] .+ p[2] * t.^rfit.param[4]
    η_fit = curve_fit(η_model, T, 1 ./ η, [1.0, 0.1])
    @show η_fit.param[2] / η_fit.param[1]
    
    scatter!(ax2, data[:, 1].^2, rfit.param[2] ./ data[:, 2], color = :black)
    # scatter!(ax2, data[bound, 1]^2, rfit.param[2] / data[bound, 2], color = :green)
    
    lines!(ax2, domain.^2, η_model(domain, η_fit.param) / η_fit.param[1], color = :red)
    scatter!(ax2, T.^2, η_fit.param[1]^(-1) ./ η, color = :red)
    @show η_fit.param[1]

    display(f)

    save(joinpath(plot_dir, "ρ_and_η_fit.png"), f)
end

function plot_lifetime_matthiesen(n_ε, n_θ, prop)
    t, τeff, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "τeff_$(prop)_$(n_ε)x$(n_θ).dat")) 
    _, τeff_ee, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "τeff_$(prop)_ee_only_$(n_ε)x$(n_θ).dat")) 
    _, τeff_imp, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "τeff_$(prop)_impurity_only_$(n_ε)x$(n_θ).dat")) 

    τ_matthiesen = 1 ./ (1 ./ τeff_ee .+ 1 ./ τeff_imp) 
   
    f = Figure(fontsize = 20)
    ax = Axis(f[1,1], ylabel = L"\tau_\text{eff}^{-1}\,(\mathrm{ps}^{-1})", xlabel = L"T\, (\mathrm{K})", 
    xticks = [4, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196],
                xtickformat = values -> [L"%$(Int(sqrt(x)))^2" for x in values])
                xlims!(ax, 0, 200)
    scatter!(ax, t.^2, 1e-12 ./ τeff, label = L"\tau")
    scatter!(ax, t.^2, 1e-12 ./ τ_matthiesen, label = L"\tau_\text{Matthiesen}", color = :grey)
    axislegend(ax, position = :lt)
    display(f)

    outfile = joinpath(plot_dir, "τeff_$(prop)_matthiesen_comparison.png")
    save(outfile, f)
end

function plot_matthiesen_violation(n_ε, n_θ, prop)
    t, τeff, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "τeff_$(prop)_$(n_ε)x$(n_θ).dat")) 
    _, τeff_ee, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "τeff_$(prop)_ee_only_$(n_ε)x$(n_θ).dat")) 
    _, τeff_imp, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "τeff_$(prop)_impurity_only_$(n_ε)x$(n_θ).dat")) 

    τeff_matthiesen = 1 ./ (1 ./ τeff_ee .+ 1 ./ τeff_imp)
   
    f = Figure(fontsize = 20)
    ax = Axis(f[1,1],
        ylabel = "% Error", 
        xlabel = L"T\, (\mathrm{K})", 
        xticks = 0:2:14
        )
    xlims!(ax, 0, 14.5)
    scatter!(ax, t, abs.(τeff .- τeff_matthiesen) ./ τeff , label = L"\tau_\text{Matthiesen}", color = :grey)
    display(f)

    outfile = joinpath(plot_dir, "τeff_$(prop)_matthiesen_violation.png")
    save(outfile, f)
end

function plot_lifetimes(n_ε, n_θ; addendum = "")
    if addendum == ""
        t_σ, τeff_σ, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "τeff_σ_$(n_ε)x$(n_θ).dat"))  
        t_η, τeff_η, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "τeff_η_$(n_ε)x$(n_θ).dat"))  
    else
        t_σ, τeff_σ, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "τeff_σ_"*addendum*"_$(n_ε)x$(n_θ).dat"))  
        t_η, τeff_η, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "τeff_η_"*addendum*"_$(n_ε)x$(n_θ).dat"))  
    end

    f = Figure(fontsize = 20)
    ax = Axis(f[1,1], ylabel = L"\tau_\text{eff}^{-1}\,(\mathrm{ps}^{-1})", xlabel = L"T\, (\mathrm{K})", 
    xticks = [4, 16, 25, 36, 49, 64, 81, 100, 144, 169, 196],
                xtickformat = values -> [L"%$(Int(sqrt(x)))^2" for x in values])
                xlims!(ax, 0, 200)
    scatter!(ax, t_σ.^2, 1e-12 ./ τeff_σ, label = L"\tau_\sigma")
    scatter!(ax, t_η.^2, 1e-12 ./ τeff_η, label = L"\tau_\eta")
    axislegend(ax, position = :lt)
    display(f)

    if addendum == ""
        save(joinpath(plot_dir, "τ_eff.png"), f)
    else
        save(joinpath(plot_dir, "τ_eff_"*addendum*".png"), f)
    end

end

function scaled_impurity_model(n_ε, n_θ; addendum = "")
    addendum != "" && (addendum = addendum*"_")

    t, ρ, _ = LudwigIO.read_property_from_file(joinpath(data_dir, "ρ_impurity_only_$(n_ε)x$(n_θ).dat"))
    ρ *= 1e8

    lupien_file = joinpath(exp_dir, "rhovT_Lupien_digitized.dat")
    lupien_data = readdlm(lupien_file)
    lfit = curve_fit(model_1, lupien_data[11:40, 1], lupien_data[11:40, 2], [0.0, 0.0001, 2.0], lower = [0.0, 0.0, 0.0])

    @show lfit.param
    λ = model_1(t, lfit.param) ./ ρ

    domain = 0.0:0.1:14.0
    f = Figure(size = (700, 600), fontsize = 24)
    ax = Axis(f[1,1],
               xlabel = L"T^2\,(\mathrm{K}^2)",
               ylabel = L"\rho / \rho_0",
               xticks = [4, 16, 36, 49, 64, 81, 100, 121, 144, 169, 196],
               xtickformat = values -> [L"%$(Int(sqrt(x)))^2" for x in values],
               yticks = vcat([1], 2:2:24),
               aspect = 1.0
    )
    xlims!(ax, 0.0, 14.0^2)
    ylims!(ax, 0.0, 10.0)

    scatter!(ax, lupien_data[:, 1].^2, lupien_data[:, 2] / lfit.param[1], color = :black, label = "Experiment")
    lines!(ax, domain.^2, model_1(domain, lfit.param) / lfit.param[1], color = :grey, label = "RTA")
    scatter!(ax, t.^2, ρ .* λ / lfit.param[1], color = :grey)

    # Result from Ludwig.jl
    t, ρ, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "ρ_"*addendum*"$(n_ε)x$(n_θ).dat"))
    ρ *= 1e8
    
    fit = curve_fit(model_1, t, ρ, [1.0, 0.1, 2.0])  
    @show fit.param

    scatter!(ax, t.^2, ρ / lfit.param[1], color = :red)
    lines!(ax, domain.^2, model_1(domain, fit.param) / lfit.param[1], color = :red, label = "Ludwig.jl")

    axislegend(ax, position = :lt)
    display(f)

    outfile = joinpath(plot_dir, addendum*"rta_ρ.png")
    save(outfile, f)

    visc_file = joinpath(exp_dir,"B1gViscvT_new.dat")
    data = readdlm(visc_file)
    rfit = curve_fit(model_3, data[1:300, 1], data[1:300, 2], [0.1866, 1.2, 0.1, 2.0])
    data[:, 2] .-= rfit.param[1] # Subtract off systematic offset

    t, η, _ = LudwigIO.read_property_from_file(joinpath(data_dir, "ηB1g_impurity_only_$(n_ε)x$(n_θ).dat"))
    η ./= λ
    fit = curve_fit(model_0, t, 1 ./ η, [1.0, 1.0], lower = [0.0, 0.0])
    @show fit.param

    η_model(t, p) = p[1] .+ p[2] * t.^rfit.param[4]
    t1, η1, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "ηB1g_"*addendum*"$(n_ε)x$(n_θ).dat"))  
    fit1 = curve_fit(model_1, t1, 1 ./ η1, [1.0, 0.1, 2.0])

    f = Figure(size = (700, 600), fontsize = 24)
    ax = Axis(f[1,1],
               xlabel = L"T^2\,(\mathrm{K}^2)",
               ylabel = L"\eta_0 / \eta",
               xticks = [4, 16, 36, 49, 64, 81, 100, 121, 144, 169, 196],
               xtickformat = values -> [L"%$(Int(sqrt(x)))^2" for x in values],
               yticks = vcat([1], 2:2:24),
               aspect = 1.0
    )
    xlims!(ax, 0.0, 14^2)
    ylims!(ax, 0.0, 18.0)
    domain = 0.0:0.1:14.0
    scatter!(ax, data[:, 1].^2, rfit.param[2] ./ data[:, 2], color = :black, label = "Experiment")

    lines!(ax, domain.^2, model_0(domain, fit.param) / fit.param[1], color = :grey, label = "RTA")
    scatter!(ax, t.^2, 1 ./ (fit.param[1] * η), color = :grey)

    lines!(ax, domain.^2, model_1(domain, fit1.param) / fit1.param[1], color = :red, label = "Ludwig.jl")
    scatter!(ax, t1.^2, fit1.param[1]^(-1) ./ η1, color = :red)

    axislegend(ax, position = :lt)

    display(f)
    outfile = joinpath(plot_dir, addendum*"rta_η.png")
    save(outfile, f)
end

function deformation_overlap(n_ε, n_θ, Uee, Vimp, T; include_impurity = true) 
    eefile = joinpath(data_dir, material*"_$(Float64(T))_$(n_ε)x$(n_θ).h5")
    L, k, v, E, dV, _, _= LudwigIO.load(eefile, symmetrized = true)
    L *= 0.5 * Uee^2

    if Vimp != 0.0 && include_impurity
        impfile = joinpath(data_dir, material*"_imp_$(T)_$(n_ε)x$(n_θ).h5")
        Limp, _, _, _, _, _, _ = LudwigIO.load(impfile)
        L += Limp * Vimp^2
    end

    ℓ = length(k)
    D = zeros(Float64, ℓ)
    Dxy = zeros(Float64, ℓ)
    for i in eachindex(k)
        μ = (i-1)÷(ℓ÷3) + 1 # Band index
        D[i] = dii_μ(k[i], 1, μ) - dii_μ(k[i], 2, μ)
        Dxy[i] = dxy_μ(k[i], μ)
    end
    
    @time eigenvalues  = eigvals(L)
    eigenvalues *= 1e-12 / hbar
    @time eigenvectors = eigvecs(L)

    b1g_weight = Vector{Float64}(undef, length(k))
    b2g_weight = Vector{Float64}(undef, length(k))
    for i in eachindex(k)
        b1g_weight[i] = abs2(dot(eigenvectors[:, i], D))
        b2g_weight[i] = abs2(dot(eigenvectors[:, i], Dxy))
    end

    f = Figure()
    ax = Axis(f[1,1],
                xlabel = "Mode Index",
                ylabel = L"|\langle \phi | D \rangle |^2" 
    )
    scatter!(ax, b1g_weight, label = L"D_{xx} - D_{yy}")
    scatter!(ax, b2g_weight, label = L"D_{xy}")
    axislegend(ax)
    display(f)

    outfile = joinpath(plot_dir, "B1g_B2g_overlaps.png")
    save(outfile, f)
end 

function optical_conductivity(n_ε, n_θ, Uee, Vimp, T; include_impurity = true)
    eefile = joinpath(data_dir, material*"_$(Float64(T))_$(n_ε)x$(n_θ).h5")
    L, k, v, E, dV, _, _= LudwigIO.load(eefile, symmetrized = true)
    L *= 0.5 * Uee^2

    if Vimp != 0.0 && include_impurity
        impfile = joinpath(data_dir, material*"_imp_$(T)_$(n_ε)x$(n_θ).h5")
        Limp, _, _, _, _, _, _ = LudwigIO.load(impfile)
        L += Limp * Vimp^2
    end

    freq = LinRange(0.0, 2.0, 200) # In THz
    ω = map(x -> 2pi * x * 1e12 * hbar, freq)

    file = joinpath(data_dir, "optical_conductivity_$(T)_$(n_ε)x$(n_θ).dat")
    open(file, "w") do f
        println(f, "---")
        println(f, "quantity: σ(ω)")
        println(f, "n_ε: $(n_ε)")
        println(f, "n_θ: $(n_θ)")
        println(f, "Uee: $(Uee)")
        println(f, "Vimp: $(Vimp)")
        println(f, "---")
        println(f, "# ω/2π, σ")
        for i in eachindex(ω)
            q = Ludwig.longitudinal_conductivity(L - im*ω[i]*I, first.(v), E, dV, kb * T) / c
            println("ω/2π = $(freq[i]) THz, σ(ω) = $(q)")
            println(f, "$(freq[i]),$(q)")
        end
    end 
end

function plot_optical_conductivity(n_ε, n_θ, T)
    freq, σ, _ = LudwigIO.read_property_from_file(joinpath(data_dir, "optical_conductivity_$(T)_$(n_ε)x$(n_θ).dat"))

    f = Figure()
    ax = Axis(f[1,1], 
                xlabel = L"\omega / 2\pi \, (\mathrm{THz})", 
                ylabel = L"σ_{xx}\,\, (\mathrm{\mu\Omega^{-1} \, cm^{-1}})"
    )
    lines!(ax, freq, real.(σ) * 1e-8, color = :red, label = L"\Re{[\sigma(\omega)]}")
    lines!(ax, freq, imag.(σ) * 1e-8, color = :blue, label = L"\Im{[\sigma(\omega)]}")
    axislegend(ax)
    display(f)

    outfile = joinpath(plot_dir, "optical_conductivity_$(T)_$(n_ε)x$(n_θ).png")
    # save(outfile, f)
end

function impurity_spectrum(n_ε, n_θ, Vimp, T)
    impfile = joinpath(data_dir, material*"_imp_$(T)_$(n_ε)x$(n_θ).h5")
    L, _, _, _, _, _, _ = LudwigIO.load(impfile)
    L *= Vimp^2 

    # λ = real.(eigvals(L))
    λ = [-2.0124963234788157e-19, -1.8685519420784373e-19, -3.1874363074662196e-20, -7.991223927839145e-21, 5.229853587611364e-20, 8.084737846273705e-20, 1.0282985589423074e-19, 1.7490230009331856e-19, 2.1781761450780775e-19, 2.6259245253981717e-19, 5.020711648723129e-19, 3.233628513892864e-6, 3.233628513892877e-6, 3.233628513892988e-6, 3.2336285138929983e-6, 3.2336285138931397e-6, 3.2336285138931745e-6, 3.233628513893303e-6, 3.299824755521965e-6, 3.2998247555220377e-6, 3.2998247555221554e-6, 3.299824755522229e-6, 3.299824755522297e-6, 3.2998247555223375e-6, 3.2998247555224993e-6, 3.3621889752541346e-6, 3.47810578961476e-6, 3.7711072415247484e-6, 3.771107241524876e-6, 3.771107241524901e-6, 3.77110724152498e-6, 3.7711072415250144e-6, 3.7711072415250707e-6, 3.771107241525194e-6, 3.7748901583246683e-6, 3.7914770352846296e-6, 3.7914770352846296e-6, 3.791477035284722e-6, 3.7914770352847605e-6, 3.791477035284799e-6, 3.7914770352848054e-6, 3.7914770352848426e-6, 3.792737143035832e-6, 3.798109876711259e-6, 3.7981098767114708e-6, 3.79810987671157e-6, 3.7981098767116876e-6, 3.7981098767117045e-6, 3.7981098767117206e-6, 3.7981098767117456e-6, 3.7986093978166376e-6, 3.80069844657831e-6, 3.8006984465783613e-6, 3.80069844657842e-6, 3.8006984465784955e-6, 3.8006984465785463e-6, 3.8006984465785997e-6, 3.8006984465787585e-6, 3.800911312472395e-6, 3.8020273754342065e-6, 3.802027375434337e-6, 3.8020273754343886e-6, 3.802027375434495e-6, 3.8020273754346546e-6, 3.8020273754347e-6, 3.802027375434768e-6, 3.8020463690447243e-6, 3.802072328930279e-6, 3.8020723289304555e-6, 3.8020723289305987e-6, 3.8020729094010014e-6, 3.802074091984414e-6, 3.8020740919844858e-6, 3.802074091984563e-6, 3.802074091984688e-6, 3.802074091984778e-6, 3.802074091984852e-6, 3.802074091984897e-6, 3.802113505820317e-6, 3.8021479892415584e-6, 3.802147989241596e-6, 3.8021479892416503e-6, 3.80214798924174e-6, 3.8021479892417562e-6, 3.8021479892417626e-6, 3.802147989241801e-6, 3.8021961036517395e-6, 3.8022802849475263e-6, 3.8022802849475695e-6, 3.80228028494764e-6, 3.802280284947704e-6, 3.8022802849479083e-6, 3.802280284948002e-6, 3.8022802849480307e-6, 3.802302283823224e-6, 3.80236646982336e-6, 3.8023664698234283e-6, 3.8023664698234986e-6, 3.8023664698235007e-6, 3.8023664698235473e-6, 3.8023664698236138e-6, 3.802366469823666e-6, 3.8023669122751756e-6, 3.8023673471561552e-6, 3.802367347156174e-6, 3.8023673471561772e-6, 3.802367347156215e-6, 3.8023673471562823e-6, 3.802367347156301e-6, 3.802367347156345e-6, 3.802380580512299e-6, 3.802389882328117e-6, 3.802389882328181e-6, 3.8023898823282055e-6, 3.8023898823283207e-6, 3.8023898823284194e-6, 3.802389882328429e-6, 3.802389882328631e-6, 3.8024628521919063e-6, 3.8025010581279204e-6, 3.802501058127932e-6, 3.802501058128001e-6, 3.8025010581280335e-6, 3.8025010581280886e-6, 3.802501058128151e-6, 3.8025010581282254e-6, 3.8025305434623203e-6, 3.8025538512450795e-6, 3.8025538512451434e-6, 3.8025538512451608e-6, 3.802553851245188e-6, 3.8025538512452163e-6, 3.8025538512452527e-6, 3.802553851245542e-6, 3.8025619090761106e-6, 3.80256815344667e-6, 3.802568153446696e-6, 3.8025681534467377e-6, 3.8025681534467754e-6, 3.8025681534467843e-6, 3.8025681534469245e-6, 3.802568153446969e-6, 3.805101161485324e-6, 3.8057444067661178e-6, 3.8057444067661178e-6, 3.8057444067661673e-6, 3.805744406766228e-6, 3.805744406766238e-6, 3.8057444067662584e-6, 3.80574440676629e-6, 3.80683491820649e-6, 3.8072923423653007e-6, 3.8072923423653066e-6, 3.8072923423654718e-6, 3.807292342365475e-6, 3.8072923423654955e-6, 3.8072923423655522e-6, 3.8072923423656204e-6, 4.201447643067698e-6, 4.275131595287556e-6, 4.275131595287609e-6, 4.275131595287652e-6, 4.2751315952877145e-6, 4.2751315952877535e-6, 4.275131595287756e-6, 4.275131595287826e-6, 4.474765711614038e-6, 4.474765711614129e-6, 4.47476571161414e-6, 4.474765711614278e-6, 4.47476571161435e-6, 4.4747657116144976e-6, 4.47476571161465e-6, 4.498241396623363e-6, 4.565913190931904e-6, 4.565913190932086e-6, 4.5659131909320975e-6, 4.565913190932225e-6, 4.565913190932286e-6, 4.565913190932307e-6, 4.565913190932463e-6, 4.573056752695611e-6, 4.593380545410913e-6, 4.593380545411075e-6, 4.593380545411091e-6, 4.593380545411107e-6, 4.5933805454111294e-6, 4.593380545411218e-6, 4.593380545411315e-6, 4.5961779032133965e-6, 4.604374338093319e-6, 4.604374338093326e-6, 4.604374338093393e-6, 4.604374338093492e-6, 4.6043743380935e-6, 4.604374338093525e-6, 4.604374338093603e-6, 4.605576644307294e-6, 4.6093223381752515e-6, 4.609322338175342e-6, 4.609322338175389e-6, 4.6093223381753904e-6, 4.6093223381755065e-6, 4.609322338175637e-6, 4.60932233817566e-6, 4.609841351079003e-6, 4.611561219225026e-6, 4.611561219225132e-6, 4.611561219225218e-6, 4.611561219225326e-6, 4.611561219225335e-6, 4.61156121922539e-6, 4.611561219225449e-6, 4.611786634207747e-6, 4.612712043224371e-6, 4.612712043224423e-6, 4.6127120432244236e-6, 4.612712043224449e-6, 4.61271204322446e-6, 4.612712043224476e-6, 4.612712043224491e-6, 4.6127684403599915e-6, 4.612910608927658e-6, 4.6129106089277085e-6, 4.6129106089277245e-6, 4.612910608927761e-6, 4.6129106089278525e-6, 4.6129106089279575e-6, 4.612910608928049e-6, 4.612938161644934e-6, 4.613049094314218e-6, 4.613049094314499e-6, 4.613049094314516e-6, 4.613051859081363e-6, 4.613062964651079e-6, 4.613062964651105e-6, 4.6130629646512225e-6, 4.613062964651256e-6, 4.613062964651294e-6, 4.613062964651445e-6, 4.613062964651531e-6, 4.613069508895718e-6, 4.613081956111564e-6, 4.6130819561118415e-6, 4.6130819561118915e-6, 4.613081956111898e-6, 4.613081956111919e-6, 4.613081956111951e-6, 4.613081956112008e-6, 4.613086749002754e-6, 4.613089692431955e-6, 4.613089692432135e-6, 4.613089692432151e-6, 4.613089692432155e-6, 4.613089692432261e-6, 4.613089692432353e-6, 4.613089692432388e-6, 4.613150960462192e-6, 4.613191904607748e-6, 4.613191904607823e-6, 4.613191904607902e-6, 4.6131919046079355e-6, 4.613191904607946e-6, 4.613191904608163e-6, 4.613191904608209e-6, 4.61319951759345e-6, 4.6132076941748084e-6, 4.613207694174875e-6, 4.613207694174917e-6, 4.613207694174992e-6, 4.613207694175017e-6, 4.613207694175062e-6, 4.613207694175101e-6, 4.613311362964153e-6, 4.613375655182155e-6, 4.613375655182326e-6, 4.613375655182383e-6, 4.613375655182406e-6, 4.613375655182483e-6, 4.613375655182578e-6, 4.613375655182706e-6, 4.6133782856519524e-6, 4.613381060235305e-6, 4.613381060235387e-6, 4.6133810602354464e-6, 4.613381060235488e-6, 4.613381060235589e-6, 4.613381060235603e-6, 4.61338106023561e-6, 4.613403547759584e-6, 4.6134136943579964e-6, 4.613413694358012e-6, 4.613413694358032e-6, 4.61341369435811e-6, 4.613413694358177e-6, 4.613413694358208e-6, 4.61341369435823e-6, 5.466147916142918e-6, 5.816058623946691e-6, 6.0248471593445535e-6, 6.024847159344581e-6, 6.02484715934462e-6, 6.024847159344702e-6, 6.024847159344815e-6, 6.0248471593448356e-6, 6.024847159344886e-6, 7.96565287639037e-6, 8.611058494731958e-6, 8.611058494732078e-6, 8.611058494732313e-6, 8.611058494732398e-6, 8.61105849473243e-6, 8.611058494732464e-6, 8.611058494732659e-6, 8.615441083293172e-6, 8.634021762965844e-6, 8.6340217629659e-6, 8.634021762966067e-6, 8.634021762966272e-6, 8.634021762966286e-6, 8.634021762966347e-6, 8.634021762966469e-6, 8.637558339234202e-6, 8.671525843463369e-6, 8.671525843463415e-6, 8.671525843463503e-6, 8.671525843463691e-6, 8.671525843464026e-6, 8.671525843464028e-6, 8.671525843464103e-6, 8.671620482578407e-6, 8.672148928672167e-6, 8.672148928672174e-6, 8.672148928672249e-6, 8.672148928672337e-6, 8.67214892867242e-6, 8.67214892867256e-6, 8.672148928672823e-6, 8.672169837892469e-6, 8.672200193903715e-6, 8.672200193903734e-6, 8.67220019390375e-6, 8.672200193903993e-6, 8.67220019390417e-6, 8.672200193904237e-6, 8.672200193904258e-6, 8.6722669967697e-6, 8.672374890212885e-6, 8.672374890212946e-6, 8.672374890213019e-6, 8.67237489021302e-6, 8.67237489021305e-6, 8.672374890213199e-6, 8.67237489021326e-6, 8.672418177698342e-6, 8.672482325572375e-6, 8.672482325572527e-6, 8.672482325572627e-6, 8.672482325572646e-6, 8.672482325572646e-6, 8.67248232557267e-6, 8.672482325572708e-6, 8.672537315081303e-6, 8.672628958607197e-6, 8.672628958607377e-6, 8.672628958607421e-6, 8.67262895860745e-6, 8.6726289586076e-6, 8.672628958607844e-6, 8.672628958607995e-6, 8.672654577961361e-6, 8.672705001767217e-6, 8.672705001767261e-6, 8.672705001767332e-6, 8.67270500176756e-6, 8.672705001767613e-6, 8.67270500176763e-6, 8.672705001767662e-6, 8.672711194915999e-6, 8.672718034701293e-6, 8.672718034701388e-6, 8.672718034701437e-6, 8.672718034701505e-6, 8.67271803470152e-6, 8.672718034701605e-6, 8.672718034701846e-6, 8.672735916094659e-6, 8.672747148695342e-6, 8.672747148695488e-6, 8.672747148695688e-6, 8.67274714869571e-6, 8.672747148695898e-6, 8.672747148695932e-6, 8.672747148695962e-6, 8.67277500977284e-6, 8.672786797470155e-6, 8.672786797470293e-6, 8.672786797470328e-6, 8.672786797470342e-6, 8.672786797470535e-6, 8.67278679747066e-6, 8.672786797470694e-6, 8.679702055280714e-6, 8.68337185025979e-6, 8.683371850260075e-6, 8.68337185026013e-6, 8.683436341879202e-6, 8.683640129963522e-6, 8.683640129963638e-6, 8.683640129963877e-6, 8.683640129963908e-6, 8.683640129964035e-6, 8.683640129964226e-6, 8.683640129964262e-6, 8.68384030842261e-6, 8.684079101957882e-6, 8.684079101957923e-6, 8.684079101957976e-6, 8.684079101958065e-6, 8.684079101958179e-6, 8.684079101958214e-6, 8.684079101958231e-6, 8.684491295877648e-6, 8.684927776608849e-6, 8.684927776608891e-6, 8.684927776608911e-6, 8.684927776608962e-6, 8.684927776608962e-6, 8.684927776609125e-6, 8.684927776609155e-6, 8.685158780299825e-6, 8.685364294915644e-6, 8.685364294915667e-6, 8.685364294915706e-6, 8.685364294915749e-6, 8.685364294915905e-6, 8.685364294915932e-6, 8.685364294916e-6, 8.68591645167628e-6, 8.68620989471852e-6, 8.686209894718733e-6, 8.686209894718772e-6, 8.686209894718815e-6, 8.686209894718898e-6, 8.686209894718948e-6, 8.686209894719031e-6, 8.68710226125198e-6, 8.68744277751747e-6, 8.687442777517511e-6, 8.687442777517569e-6, 8.687442777517698e-6, 8.687442777517727e-6, 8.687442777517867e-6, 8.68744277751788e-6, 1.0181740051152635e-5, 1.0181740051152688e-5, 1.018174005115285e-5, 1.0181740051152927e-5, 1.018174005115303e-5, 1.0181740051153164e-5, 1.0181740051153179e-5, 1.0211480127590988e-5, 1.0521308643781314e-5, 1.0521308643781461e-5, 1.0521308643781697e-5, 1.0521308643782442e-5, 1.052130864378273e-5, 1.0521308643783049e-5, 1.0521308643783145e-5, 1.0521397971611633e-5, 1.0521673694631226e-5, 1.052167369463139e-5, 1.0521673694631414e-5, 1.0521673694631505e-5, 1.052167369463153e-5, 1.0521673694631612e-5, 1.0521673694631656e-5, 1.0521725857434139e-5, 1.052179390253856e-5, 1.0521793902538588e-5, 1.052179390253859e-5, 1.0521793902538609e-5, 1.052179390253865e-5, 1.0521793902538732e-5, 1.0521793902538839e-5, 1.0521980034663006e-5, 1.0522339651191833e-5, 1.0522339651191905e-5, 1.0522339651192038e-5, 1.0522339651192083e-5, 1.0522339651192546e-5, 1.0522339651192629e-5, 1.0522339651192688e-5, 1.0522398164414972e-5, 1.0522508733982929e-5, 1.052250873398303e-5, 1.0522508733983065e-5, 1.0522508733983087e-5, 1.0522508733983109e-5, 1.0522508733983212e-5, 1.0522508733983303e-5, 1.0522565125045096e-5, 1.052266686554586e-5, 1.0522666865545928e-5, 1.0522666865545937e-5, 1.0522666865545972e-5, 1.0522666865545986e-5, 1.052266686554605e-5, 1.0522666865546147e-5, 1.0522688605737358e-5, 1.0522722531373379e-5, 1.0522722531373441e-5, 1.0522722531373485e-5, 1.052272253137349e-5, 1.0522722531373599e-5, 1.0522722531373831e-5, 1.0522722531373872e-5, 1.0522728561385092e-5, 1.0522733988774535e-5, 1.0522733988774631e-5, 1.0522733988774721e-5, 1.0522733988774755e-5, 1.052273398877496e-5, 1.0522733988774975e-5, 1.0522733988775173e-5, 1.0522844529996315e-5, 1.052287818809519e-5, 1.0522878188095291e-5, 1.0522878188095304e-5, 1.0522878188095357e-5, 1.0522878188095525e-5, 1.0522878188095643e-5, 1.0522878188095664e-5, 1.053038589047894e-5, 1.053608614399e-5, 1.0536086143990096e-5, 1.0536086143990191e-5, 1.0536086143990354e-5, 1.0536086143990489e-5, 1.0536086143990603e-5, 1.0536086143990692e-5, 1.0536201228251446e-5, 1.0536319917943757e-5, 1.0536319917943847e-5, 1.0536319917943952e-5, 1.053643467151688e-5, 1.0536679128957234e-5, 1.0536679128957245e-5, 1.0536679128957245e-5, 1.0536679128957269e-5, 1.05366791289574e-5, 1.0536679128957401e-5, 1.0536679128957452e-5, 1.0536904111365737e-5, 1.0537153016251543e-5, 1.0537153016251587e-5, 1.0537153016251704e-5, 1.0537153016251733e-5, 1.0537153016251838e-5, 1.0537153016251926e-5, 1.053715301625195e-5, 1.0537434550168489e-5, 1.053765873473226e-5, 1.0537658734732388e-5, 1.0537658734732463e-5, 1.0537658734732625e-5, 1.0537658734732697e-5, 1.0537658734732756e-5, 1.0537658734732764e-5, 1.0538276473529922e-5, 1.053861899019229e-5, 1.0538618990192329e-5, 1.0538618990192527e-5, 1.0538618990192629e-5, 1.0538618990192693e-5, 1.0538618990192793e-5, 1.0538618990192844e-5, 1.0539693229387551e-5, 1.0540192574841736e-5, 1.0540192574841802e-5, 1.0540192574841933e-5, 1.0540192574841943e-5, 1.0540192574842084e-5, 1.0540192574842151e-5, 1.0540192574842219e-5, 1.0541332782794857e-5, 1.0541805325920379e-5, 1.0541805325920531e-5, 1.0541805325920573e-5, 1.0541805325920575e-5, 1.0541805325920641e-5, 1.0541805325920648e-5, 1.054180532592083e-5, 1.0733177109364438e-5, 1.0748056713548799e-5, 1.0748056713548852e-5, 1.0748056713549033e-5, 1.074805671354945e-5, 1.0748056713549761e-5, 1.0748056713549937e-5, 1.0748056713549968e-5, 1.1518351030664918e-5, 1.1518351030665067e-5, 1.15183510306651e-5, 1.151835103066512e-5, 1.1518351030665175e-5, 1.151835103066523e-5, 1.1518351030665238e-5, 1.2141143782274775e-5, 1.2559891980800888e-5, 1.2670885419887147e-5, 1.2670885419887166e-5, 1.2670885419887447e-5, 1.26708854198875e-5, 1.2670885419887927e-5, 1.2670885419888023e-5, 1.2670885419888174e-5, 1.30802283352826e-5, 1.3080228335282682e-5, 1.3080228335282766e-5, 1.3080228335282963e-5, 1.308022833528309e-5, 1.3080228335283197e-5, 1.3080228335283308e-5, 1.3254165312603965e-5, 1.325416531260401e-5, 1.3254165312604111e-5, 1.3254165312604135e-5, 1.325416531260415e-5, 1.3254165312604348e-5, 1.3254165312604379e-5, 1.35507857871104e-5, 1.3563886920214679e-5, 1.3638149060749113e-5, 1.441650366997559e-5, 1.4416503669975734e-5, 1.4416503669975904e-5, 1.4416503669975919e-5, 1.4416503669976064e-5, 1.4416503669976139e-5, 1.4416503669976373e-5, 1.4545632325903219e-5, 1.4712883171306194e-5, 1.47128831713062e-5, 1.4712883171306243e-5, 1.4712883171306279e-5, 1.471288317130651e-5, 1.4712883171306543e-5, 1.4712883171306635e-5, 1.486223010539257e-5, 1.5189035289588351e-5, 1.5189035289588515e-5, 1.5189035289588575e-5, 1.5189035289588612e-5, 1.5189035289588647e-5, 1.5189035289588734e-5, 1.5189035289588895e-5, 1.5245225168393988e-5, 1.5258207935041903e-5, 1.525820793504195e-5, 1.5258207935042042e-5, 1.5258207935042078e-5, 1.5258207935042306e-5, 1.5258207935042369e-5, 1.5258207935042438e-5, 1.527212964062845e-5, 1.533282033758377e-5, 1.5332820337583858e-5, 1.5332820337583973e-5, 1.5332820337584013e-5, 1.5332820337584115e-5, 1.5332820337584142e-5, 1.5332820337584176e-5, 1.5337672011383293e-5, 1.5351446765929183e-5, 1.5351446765929355e-5, 1.5351446765929582e-5, 1.535144676592959e-5, 1.5351446765929643e-5, 1.5351446765929725e-5, 1.5351446765929867e-5, 1.535832504485901e-5, 1.535832504485918e-5, 1.535832504485924e-5, 1.5358325044859255e-5, 1.535832504485938e-5, 1.5358325044859387e-5, 1.5358325044859438e-5, 1.53602815135088e-5, 1.536889798239307e-5, 1.536889798239319e-5, 1.5368897982393227e-5, 1.5368897982393274e-5, 1.53688979823933e-5, 1.5368897982393308e-5, 1.5368897982393498e-5, 1.5369637523783996e-5, 1.537321440658357e-5, 1.5373214406583572e-5, 1.5373214406583576e-5, 1.537321440658375e-5, 1.537321440658399e-5, 1.5373214406584067e-5, 1.5373214406584118e-5, 1.5373399587295442e-5, 1.537412565601517e-5, 1.5374125656015448e-5, 1.5374125656015536e-5, 1.5374138070444355e-5, 1.5374239260833343e-5, 1.5374239260833434e-5, 1.5374239260833434e-5, 1.537423926083346e-5, 1.5374239260833563e-5, 1.5374239260833614e-5, 1.5374239260833638e-5, 1.537424154847372e-5, 1.5374244571807158e-5, 1.537424457180724e-5, 1.537424457180725e-5, 1.5374244571807328e-5, 1.5374244571807433e-5, 1.537424457180755e-5, 1.537424457180763e-5, 1.5374249110253925e-5, 1.5374252524925794e-5, 1.5374252524925808e-5, 1.537425252492582e-5, 1.537425252492605e-5, 1.5374252524926123e-5, 1.537425252492628e-5, 1.537425252492635e-5, 1.5374294414511846e-5, 1.5374312821599976e-5, 1.5374312821599993e-5, 1.5374312821600247e-5, 1.537431282160042e-5, 1.5374312821600572e-5, 1.5374312821600596e-5, 1.5374312821600775e-5, 1.537451182527832e-5, 1.5374620130053386e-5, 1.537462013005342e-5, 1.5374620130053478e-5, 1.537462013005352e-5, 1.537462013005355e-5, 1.53746201300536e-5, 1.5374620130053627e-5, 1.5374739823753145e-5, 1.537489600722956e-5, 1.537489600722963e-5, 1.537489600722969e-5, 1.537489600722989e-5, 1.5374896007230036e-5, 1.5374896007230094e-5, 1.5374896007230148e-5, 1.53748964401302e-5, 1.537489678140107e-5, 1.537489678140117e-5, 1.5374896781401226e-5, 1.5374896781401385e-5, 1.5374896781401473e-5, 1.5374896781401775e-5, 1.5374896781401897e-5, 1.5375355828370933e-5, 1.5375534683240545e-5, 1.5375534683240596e-5, 1.5375534683240606e-5, 1.5375534683240694e-5, 1.5375534683240856e-5, 1.537553468324088e-5, 1.5375534683240975e-5, 1.5375572305184015e-5, 1.537560394951791e-5, 1.5375603949517914e-5, 1.5375603949518002e-5, 1.5375603949518002e-5, 1.537560394951805e-5, 1.537560394951806e-5, 1.537560394951811e-5, 1.538730979271629e-5, 1.5388827659966593e-5, 1.5388827659966695e-5, 1.5388827659966854e-5, 1.5388830789061954e-5, 1.5388837233045044e-5, 1.538883723304517e-5, 1.5388837233045214e-5, 1.538883723304524e-5, 1.5388837233045515e-5, 1.5388837233045586e-5, 1.5388837233045664e-5, 1.5388947656443852e-5, 1.5389039236930258e-5, 1.538903923693028e-5, 1.5389039236930434e-5, 1.5389039236930516e-5, 1.5389039236930607e-5, 1.5389039236930634e-5, 1.5389039236930638e-5, 1.538974334678816e-5, 1.539034828807377e-5, 1.5390348288073803e-5, 1.5390348288073858e-5, 1.5390348288073885e-5, 1.5390348288073912e-5, 1.539034828807396e-5, 1.539034828807398e-5, 1.539063932391012e-5, 1.5390639323910273e-5, 1.5390639323910384e-5, 1.5390639323910784e-5, 1.539063932391109e-5, 1.5390639323911174e-5, 1.5390639323911343e-5, 1.5390969500244676e-5, 1.5391353221576932e-5, 1.5391353221576973e-5, 1.539135322157701e-5, 1.5391353221577088e-5, 1.539135322157722e-5, 1.539135322157728e-5, 1.539135322157734e-5, 1.5392298270302094e-5, 1.5393590986593772e-5, 1.5393590986593938e-5, 1.539359098659397e-5, 1.5393590986594036e-5, 1.53935909865941e-5, 1.5393590986594104e-5, 1.539359098659424e-5, 1.539425953818088e-5, 1.5395352695946383e-5, 1.5396044359142363e-5, 1.539604435914252e-5, 1.539604435914262e-5, 1.539604435914264e-5, 1.5396044359142644e-5, 1.5396044359142708e-5, 1.539604435914279e-5, 1.5396324157727244e-5, 1.5396658830633284e-5, 1.5396658830633335e-5, 1.5396658830633342e-5, 1.5396658830633376e-5, 1.5396658830633396e-5, 1.539665883063342e-5, 1.5396658830633464e-5, 1.539688080722943e-5, 1.5396880807229436e-5, 1.5396880807229507e-5, 1.5396880807229534e-5, 1.5396880807229653e-5, 1.5396880807229683e-5, 1.5396880807229812e-5, 1.5397234258192436e-5, 1.5397491030978043e-5, 1.5397491030978104e-5, 1.5397491030978155e-5, 1.5397491030978155e-5, 1.5397491030978216e-5, 1.539749103097822e-5, 1.53974910309784e-5, 1.539793651874841e-5, 1.539793651874848e-5, 1.539793651874855e-5, 1.5397936518748722e-5, 1.5397936518748736e-5, 1.5397936518748763e-5, 1.539793651874881e-5, 1.5398221951683036e-5, 1.5398976237253614e-5, 1.539897623725379e-5, 1.5398976237253865e-5, 1.5398976237254454e-5, 1.5398976237254857e-5, 1.539897623725489e-5, 1.5398976237254952e-5, 1.5399112340364004e-5, 1.5399489956109236e-5, 1.539948995610927e-5, 1.5399489956109307e-5, 1.5399489956109344e-5, 1.5399489956109412e-5, 1.5399489956109422e-5, 1.5399489956109446e-5, 1.5399500843382724e-5, 1.5399511879047263e-5, 1.539951187904744e-5, 1.5399511879047473e-5, 1.53995118790476e-5, 1.5399511879047656e-5, 1.5399511879047663e-5, 1.5399511879047707e-5, 1.539978816931502e-5, 1.5400242930789395e-5, 1.5400242930789486e-5, 1.540024293078949e-5, 1.5400242930789503e-5, 1.5400242930789534e-5, 1.540024293078965e-5, 1.5400242930789717e-5, 1.5400272615482026e-5, 1.5400322718866968e-5, 1.5400322718866978e-5, 1.5400322718867012e-5, 1.5400322718867073e-5, 1.5400322718867107e-5, 1.540032271886714e-5, 1.5400322718867195e-5, 1.5400367339234184e-5, 1.5400431092744125e-5, 1.5400431092744254e-5, 1.540043109274431e-5, 1.540043109274471e-5, 1.5400431092745128e-5, 1.540043109274515e-5, 1.5400431092745206e-5, 1.5400465365539095e-5, 1.540051931659903e-5, 1.5400519316599047e-5, 1.5400519316599067e-5, 1.54005193165991e-5, 1.5400519316599165e-5, 1.5400519316599192e-5, 1.5400519316599274e-5, 1.5400527397601954e-5, 1.5400535414510838e-5, 1.5400535414510906e-5, 1.5400535414510974e-5, 1.5400535414510984e-5, 1.5400535414510984e-5, 1.5400535414510987e-5, 1.5400535414511082e-5, 1.540057033715515e-5, 1.540058829726383e-5, 1.5400588297263943e-5, 1.5400588297264075e-5, 1.5400588297264092e-5, 1.540058829726418e-5, 1.5400588297264204e-5, 1.5400588297264434e-5, 1.5400635517138868e-5, 1.5400653768674317e-5, 1.5400653768674378e-5, 1.5400653768674385e-5, 1.5400653768674405e-5, 1.5400653768674442e-5, 1.540065376867445e-5, 1.5400653768674513e-5, 1.5452986747318122e-5, 1.5467220822770117e-5, 1.5467220822770205e-5, 1.5467220822770347e-5, 1.5467220822770378e-5, 1.546722082277045e-5, 1.5467220822770523e-5, 1.5467220822770713e-5, 1.5481787862554673e-5, 1.553770900257858e-5, 1.5537709002578633e-5, 1.5537709002578714e-5, 1.553770900257879e-5, 1.5537709002579063e-5, 1.553770900257908e-5, 1.5537709002579127e-5, 1.5542866858362447e-5, 1.5562835295325317e-5, 1.5562835295325493e-5, 1.556283529532556e-5, 1.556283529532567e-5, 1.5562835295325683e-5, 1.5562835295325855e-5, 1.5562835295325866e-5, 1.5564814922654133e-5, 1.55725016373328e-5, 1.5572501637332817e-5, 1.557250163733283e-5, 1.5572501637332875e-5, 1.557250163733291e-5, 1.557250163733291e-5, 1.557250163733333e-5, 1.5573282776182012e-5, 1.557635836216222e-5, 1.5576358362162337e-5, 1.5576358362162405e-5, 1.5576358362162446e-5, 1.5576358362162452e-5, 1.557635836216248e-5, 1.5576358362162517e-5, 1.5576644375601422e-5, 1.55776014245516e-5, 1.5577601424551664e-5, 1.5577601424552e-5, 1.5577626944733908e-5, 1.5577731381336445e-5, 1.557773138133657e-5, 1.557773138133665e-5, 1.557773138133673e-5, 1.5577731381336784e-5, 1.557773138133688e-5, 1.5577731381336906e-5, 1.5577807271082725e-5, 1.5577994048831266e-5, 1.5577994048831303e-5, 1.557799404883132e-5, 1.557799404883151e-5, 1.557799404883154e-5, 1.5577994048831645e-5, 1.5577994048831767e-5, 1.5578043461516225e-5, 1.557812025999539e-5, 1.5578120259995466e-5, 1.5578120259995524e-5, 1.557812025999557e-5, 1.5578120259995652e-5, 1.557812025999566e-5, 1.5578120259995666e-5, 1.5578198367237718e-5, 1.5578294232656223e-5, 1.5578294232656264e-5, 1.5578294232656308e-5, 1.557829423265651e-5, 1.557829423265657e-5, 1.557829423265663e-5, 1.557829423265665e-5, 1.5578383109791648e-5, 1.557848441403375e-5, 1.557848441403383e-5, 1.5578484414033844e-5, 1.5578484414033858e-5, 1.5578484414033933e-5, 1.5578484414033977e-5, 1.557848441403415e-5, 1.557856463501522e-5, 1.5578652823293954e-5, 1.5578652823294018e-5, 1.557865282329404e-5, 1.5578652823294123e-5, 1.5578652823294157e-5, 1.55786528232942e-5, 1.557865282329432e-5, 1.5578770990611044e-5, 1.557883329991613e-5, 1.557883329991624e-5, 1.5578833299916268e-5, 1.557883329991637e-5, 1.5578833299916417e-5, 1.557883329991658e-5, 1.5578833299916593e-5, 1.5579078912273527e-5, 1.5579194661813786e-5, 1.557919466181388e-5, 1.5579194661814118e-5, 1.5579194661814125e-5, 1.557919466181424e-5, 1.557919466181427e-5, 1.5579194661814372e-5, 1.5579300503964056e-5, 1.5579360318326355e-5, 1.5579360318326368e-5, 1.5579360318326385e-5, 1.557936031832677e-5, 1.557936031832707e-5, 1.5579360318327148e-5, 1.5579360318327158e-5, 1.5580142778579436e-5, 1.560121332625582e-5, 1.56023453604253e-5, 1.56023453604253e-5, 1.5602345360425378e-5, 1.5602345360425415e-5, 1.5602345360425578e-5, 1.56023453604256e-5, 1.560234536042564e-5, 1.560543547111329e-5, 1.560543547111354e-5, 1.560543547111355e-5, 1.5605435471113556e-5, 1.5605435471113597e-5, 1.560543547111372e-5, 1.560543547111373e-5, 1.5640170028251694e-5, 1.5640170028251816e-5, 1.5640170028251935e-5, 1.564017002825199e-5, 1.564017002825199e-5, 1.5640170028252084e-5, 1.564017002825217e-5, 1.5648948650050292e-5, 1.5654411283499837e-5, 1.5704978748882123e-5, 1.5704978748882137e-5, 1.5704978748882235e-5, 1.570497874888227e-5, 1.5704978748882397e-5, 1.5704978748882445e-5, 1.5704978748882502e-5, 1.5710062509090918e-5, 1.5728370543914643e-5, 1.5728370543914666e-5, 1.5728370543914738e-5, 1.5728370543914917e-5, 1.5728370543915073e-5, 1.572837054391512e-5, 1.5728370543915253e-5, 1.5730457286547594e-5, 1.5738282107281465e-5, 1.5738282107281512e-5, 1.573828210728158e-5, 1.5738282107281614e-5, 1.5738282107281827e-5, 1.573828210728194e-5, 1.573828210728196e-5, 1.5739116063417374e-5, 1.5742722190376582e-5, 1.574272219037677e-5, 1.5742722190377032e-5, 1.574272219037712e-5, 1.5742722190377256e-5, 1.574272219037728e-5, 1.5742722190377375e-5, 1.5742939884274995e-5, 1.5743457829730292e-5, 1.5743457829730353e-5, 1.5743457829730522e-5, 1.57435030077347e-5, 1.5743851319071124e-5, 1.574385131907118e-5, 1.5743851319071287e-5, 1.574385131907135e-5, 1.5743851319071368e-5, 1.5743851319071524e-5, 1.5743851319071564e-5, 1.574386156288421e-5, 1.5743873489307748e-5, 1.5743873489307857e-5, 1.5743873489307914e-5, 1.5743873489308202e-5, 1.574387348930826e-5, 1.5743873489308277e-5, 1.5743873489308483e-5, 1.5743926063795047e-5, 1.574396494703529e-5, 1.5743964947035336e-5, 1.574396494703534e-5, 1.574396494703544e-5, 1.5743964947035444e-5, 1.5743964947035566e-5, 1.574396494703557e-5, 1.574418254470967e-5, 1.574451737555342e-5, 1.5744517375553465e-5, 1.5744517375553478e-5, 1.5744517375553512e-5, 1.5744517375553563e-5, 1.57445173755536e-5, 1.574451737555367e-5, 1.5744549480872526e-5, 1.574460902350586e-5, 1.5744609023505872e-5, 1.574460902350591e-5, 1.5744609023505916e-5, 1.574460902350602e-5, 1.574460902350604e-5, 1.5744609023506153e-5, 1.5744677362293344e-5, 1.5744731262914687e-5, 1.5744731262914744e-5, 1.5744731262914754e-5, 1.5744731262914897e-5, 1.574473126291492e-5, 1.5744731262914934e-5, 1.5744731262915083e-5, 1.5744802055392883e-5, 1.5744856627529528e-5, 1.5744856627529602e-5, 1.5744856627529602e-5, 1.5744856627529657e-5, 1.574485662752969e-5, 1.5744856627529785e-5, 1.5744856627529972e-5, 1.574497266179832e-5, 1.574504869447205e-5, 1.5745048694472163e-5, 1.574504869447218e-5, 1.574504869447245e-5, 1.574504869447247e-5, 1.57450486944725e-5, 1.5745048694472532e-5, 1.5745089247592067e-5, 1.5745112382233852e-5, 1.574511238223388e-5, 1.57451123822339e-5, 1.5745112382234035e-5, 1.5745112382234343e-5, 1.574511238223451e-5, 1.574511238223456e-5, 1.5755114615957036e-5, 1.5771190832821003e-5, 1.5776208716049014e-5, 1.5776208716049048e-5, 1.5776208716049204e-5, 1.5776208716049302e-5, 1.5776208716049366e-5, 1.577620871604942e-5, 1.577620871604973e-5, 1.58159699948885e-5, 1.581596999488865e-5, 1.5815969994888736e-5, 1.5815969994888895e-5, 1.5815969994888905e-5, 1.5815969994888953e-5, 1.5815969994888966e-5, 1.582903761128953e-5, 1.583210920565267e-5, 1.5832109205652764e-5, 1.5832109205652832e-5, 1.5832109205653032e-5, 1.5832109205653076e-5, 1.5832109205653164e-5, 1.583210920565327e-5, 1.5872592722657742e-5, 1.587612527793434e-5, 1.5876125277934407e-5, 1.58761252779345e-5, 1.5876125277934773e-5, 1.587612527793495e-5, 1.587612527793496e-5, 1.587612527793517e-5, 1.5880987444064992e-5, 1.5898575745265942e-5, 1.589857574526602e-5, 1.589857574526606e-5, 1.5898575745266342e-5, 1.5898575745266413e-5, 1.589857574526649e-5, 1.5898575745266525e-5, 1.5900603868373895e-5, 1.590840724639339e-5, 1.5908407246393723e-5, 1.590840724639374e-5, 1.590840724639383e-5, 1.5908407246394255e-5, 1.5908407246394347e-5, 1.5908407246394547e-5, 1.5909180494728e-5, 1.591275647551856e-5, 1.5912756475518575e-5, 1.5912756475518826e-5, 1.5912756475518832e-5, 1.5912756475518873e-5, 1.591275647551892e-5, 1.5912756475518988e-5, 1.591291468645982e-5, 1.5913259204644244e-5, 1.5913259204644275e-5, 1.5913259204644417e-5, 1.591329264930777e-5, 1.5913440301839996e-5, 1.5913440301840183e-5, 1.5913440301840315e-5, 1.5913440301840413e-5, 1.5913440301840535e-5, 1.591344030184054e-5, 1.5913440301840576e-5, 1.5913513787728615e-5, 1.5913739705898434e-5, 1.5913739705898447e-5, 1.5913739705898508e-5, 1.5913739705898572e-5, 1.5913739705898664e-5, 1.5913739705898667e-5, 1.5913739705898874e-5, 1.5913752262931365e-5, 1.5913766198439305e-5, 1.5913766198439366e-5, 1.5913766198439403e-5, 1.591376619843941e-5, 1.5913766198439566e-5, 1.5913766198439583e-5, 1.591376619843962e-5, 1.5913855422765237e-5, 1.5913957354756097e-5, 1.591395735475629e-5, 1.59139573547564e-5, 1.5913957354756632e-5, 1.5913957354756764e-5, 1.591395735475687e-5, 1.5913957354756876e-5, 1.591404705866363e-5, 1.5914107135459426e-5, 1.5914107135459518e-5, 1.5914107135459596e-5, 1.5914107135459626e-5, 1.591410713545965e-5, 1.5914107135459667e-5, 1.591410713545971e-5, 1.5914253965520957e-5, 1.5914367286454014e-5, 1.591436728645417e-5, 1.5914367286454174e-5, 1.5914367286454282e-5, 1.5914367286454343e-5, 1.5914367286454397e-5, 1.5914367286454404e-5, 1.5914451750475885e-5, 1.59145423778019e-5, 1.591454237780191e-5, 1.5914542377801943e-5, 1.5914542377801973e-5, 1.5914542377801993e-5, 1.5914542377802014e-5, 1.591454237780208e-5, 1.591458109081052e-5, 1.591460650021751e-5, 1.5914606500217653e-5, 1.5914606500217714e-5, 1.5914606500217734e-5, 1.591460650021778e-5, 1.591460650021789e-5, 1.591460650021791e-5, 1.5914859324364088e-5, 1.5914927491827404e-5, 1.5914927491827448e-5, 1.591492749182747e-5, 1.5914927491827546e-5, 1.5914927491827624e-5, 1.5914927491827725e-5, 1.591492749182779e-5, 1.5944676942913968e-5, 1.5950695484415347e-5, 1.5950695484415438e-5, 1.5950695484415557e-5, 1.595069548441558e-5, 1.5950695484415604e-5, 1.5950695484415746e-5, 1.5950695484415746e-5, 1.602008041726636e-5, 1.6020080417266368e-5, 1.602008041726638e-5, 1.6020080417266402e-5, 1.6020080417266493e-5, 1.6020080417266507e-5, 1.6020080417266598e-5, 1.6032419617571103e-5, 1.607693409535585e-5, 1.607693409535597e-5, 1.607693409535603e-5, 1.6076934095356233e-5, 1.6076934095356304e-5, 1.6076934095356413e-5, 1.6076934095356436e-5, 1.608167364587367e-5, 1.6083211876396224e-5, 1.609894998994585e-5, 1.6098949989945884e-5, 1.6098949989945918e-5, 1.6098949989946064e-5, 1.6098949989946115e-5, 1.609894998994616e-5, 1.609894998994625e-5, 1.6099707995164617e-5, 1.6099707995164753e-5, 1.6099707995164773e-5, 1.6099707995164824e-5, 1.6099707995164942e-5, 1.6099707995164942e-5, 1.6099707995164966e-5, 1.610089792143359e-5, 1.6108104723449248e-5, 1.610810472344931e-5, 1.6108104723449316e-5, 1.6108104723449607e-5, 1.610810472344962e-5, 1.610810472344963e-5, 1.6108104723449705e-5, 1.6108911269969777e-5, 1.6112189991400783e-5, 1.6112189991400824e-5, 1.6112189991400854e-5, 1.6112189991400878e-5, 1.6112189991401007e-5, 1.6112189991401048e-5, 1.611218999140119e-5, 1.6112441983961905e-5, 1.6113285201204287e-5, 1.6113285201204355e-5, 1.6113285201204385e-5, 1.6113285201204422e-5, 1.6113285201204585e-5, 1.611328520120463e-5, 1.6113285201204694e-5, 1.6113315542125303e-5, 1.611336724691691e-5, 1.611336724691707e-5, 1.6113367246917253e-5, 1.6113367246917345e-5, 1.611336724691743e-5, 1.6113367246917518e-5, 1.6113367246917602e-5, 1.6113379488149034e-5, 1.6113385249280685e-5, 1.611338524928079e-5, 1.611338524928085e-5, 1.6113590004702395e-5, 1.6113915172945543e-5, 1.6113915172945625e-5, 1.6113915172945652e-5, 1.6113915172945716e-5, 1.6113915172945723e-5, 1.6113915172945848e-5, 1.6113915172946058e-5, 1.6113984350592624e-5, 1.6114140024388062e-5, 1.611414002438807e-5, 1.611414002438817e-5, 1.6114140024388174e-5, 1.6114140024388265e-5, 1.611414002438832e-5, 1.611414002438842e-5, 1.6114224483878367e-5, 1.6114315521393906e-5, 1.6114315521393916e-5, 1.6114315521394007e-5, 1.6114315521394204e-5, 1.6114315521394248e-5, 1.6114315521394353e-5, 1.6114315521394536e-5, 1.61143718823453e-5, 1.611442964797454e-5, 1.6114429647974593e-5, 1.6114429647974664e-5, 1.6114429647974826e-5, 1.6114429647974986e-5, 1.6114429647974986e-5, 1.6114429647975033e-5, 1.6114528248757123e-5, 1.6114597049620835e-5, 1.6114597049620937e-5, 1.6114597049621153e-5, 1.6114597049621153e-5, 1.6114597049621228e-5, 1.6114597049621238e-5, 1.6114597049621455e-5, 1.6114697798482786e-5, 1.611479429526011e-5, 1.6114794295260263e-5, 1.6114794295260494e-5, 1.6114794295260527e-5, 1.611479429526055e-5, 1.611479429526059e-5, 1.611479429526073e-5, 1.6114801131103126e-5, 1.6114807876053485e-5, 1.6114807876053637e-5, 1.611480787605373e-5, 1.611480787605379e-5, 1.6114807876053803e-5, 1.6114807876053847e-5, 1.611480787605385e-5, 1.6135307017754245e-5, 1.6150339482123246e-5, 1.615787905477623e-5, 1.6157879054776495e-5, 1.6157879054776553e-5, 1.615787905477659e-5, 1.6157879054776593e-5, 1.6157879054776688e-5, 1.615787905477669e-5, 1.6207027382602234e-5, 1.6207027382602407e-5, 1.6207027382602512e-5, 1.6207027382602525e-5, 1.620702738260269e-5, 1.6207027382602695e-5, 1.6207027382602854e-5, 1.625912936007199e-5, 1.6267241499739035e-5, 1.6267241499739455e-5, 1.6267241499739526e-5, 1.6267241499739547e-5, 1.626724149973957e-5, 1.62672414997397e-5, 1.626724149973981e-5, 1.6278629712706698e-5, 1.6319745722628597e-5, 1.6319745722628716e-5, 1.631974572262877e-5, 1.6319745722628834e-5, 1.631974572262896e-5, 1.6319745722628983e-5, 1.631974572262911e-5, 1.6324220127694313e-5, 1.6339879876479074e-5, 1.6339879876479138e-5, 1.6339879876479246e-5, 1.6339879876479307e-5, 1.633987987647931e-5, 1.6339879876479345e-5, 1.6339879876479396e-5, 1.6341880532634025e-5, 1.6349238722167738e-5, 1.634923872216787e-5, 1.6349238722167883e-5, 1.634923872216804e-5, 1.6349238722168043e-5, 1.6349238722168117e-5, 1.6349238722168144e-5, 1.6350049642821688e-5, 1.6353370969738886e-5, 1.6353370969739e-5, 1.6353370969739022e-5, 1.6353370969739117e-5, 1.6353370969739334e-5, 1.635337096973938e-5, 1.6353370969739557e-5, 1.635362298434669e-5, 1.635457268336161e-5, 1.63545726833617e-5, 1.6354572683361704e-5, 1.63545726833618e-5, 1.635457268336195e-5, 1.635457268336202e-5, 1.6354572683362086e-5, 1.6354600053566768e-5, 1.6354642759962245e-5, 1.6354642759962262e-5, 1.6354642759962296e-5, 1.6354642759962516e-5, 1.6354642759962682e-5, 1.635464275996269e-5, 1.6354642759962852e-5, 1.635471397462186e-5, 1.635477787386409e-5, 1.6354777873864124e-5, 1.6354777873864293e-5, 1.6354823264905086e-5, 1.6354966568925544e-5, 1.6354966568925575e-5, 1.635496656892578e-5, 1.6354966568925887e-5, 1.635496656892601e-5, 1.6354966568926083e-5, 1.6354966568926253e-5, 1.635497918627716e-5, 1.6354991817667244e-5, 1.6354991817667305e-5, 1.6354991817667342e-5, 1.635499181766741e-5, 1.635499181766765e-5, 1.6354991817667718e-5, 1.6354991817667776e-5, 1.6355115225427566e-5, 1.6355198044718024e-5, 1.635519804471803e-5, 1.6355198044718078e-5, 1.635519804471813e-5, 1.635519804471814e-5, 1.6355198044718234e-5, 1.6355198044718244e-5, 1.635532793449828e-5, 1.6355438267591335e-5, 1.6355438267591362e-5, 1.6355438267591386e-5, 1.6355438267591487e-5, 1.6355438267591772e-5, 1.6355438267591864e-5, 1.6355438267592013e-5, 1.6355613001706963e-5, 1.6355701122097895e-5, 1.6355701122098064e-5, 1.6355701122098084e-5, 1.6355701122098125e-5, 1.6355701122098362e-5, 1.6355701122098508e-5, 1.6355701122098664e-5, 1.635594589473618e-5, 1.6356084767068928e-5, 1.635608476706894e-5, 1.635608476706899e-5, 1.635608476706907e-5, 1.6356084767069182e-5, 1.6356084767069226e-5, 1.6356084767069294e-5, 1.63561491122933e-5, 1.635619429894163e-5, 1.6356194298941774e-5, 1.635619429894187e-5, 1.6356194298941913e-5, 1.635619429894219e-5, 1.6356194298942448e-5, 1.6356194298942536e-5, 1.6399159470953668e-5, 1.6405633979764167e-5, 1.6405633979764214e-5, 1.6405633979764238e-5, 1.6405633979764295e-5, 1.6405633979764367e-5, 1.6405633979764444e-5, 1.6405633979764763e-5, 1.6408896511955103e-5, 1.6408896511955147e-5, 1.6408896511955228e-5, 1.640889651195529e-5, 1.6408896511955414e-5, 1.6408896511955503e-5, 1.6408896511955516e-5, 1.6428911856081015e-5, 1.642891185608117e-5, 1.6428911856081188e-5, 1.642891185608135e-5, 1.642891185608135e-5, 1.6428911856081544e-5, 1.6428911856081564e-5, 1.650834822028105e-5, 1.6538549850887698e-5, 1.6538549850887728e-5, 1.6538549850887935e-5, 1.653854985088799e-5, 1.653854985088807e-5, 1.6538549850888077e-5, 1.6538549850888104e-5, 1.6570771172664283e-5, 1.6687277881308856e-5, 1.6687277881308924e-5, 1.6687277881309053e-5, 1.6687277881309144e-5, 1.6687277881309148e-5, 1.6687277881309415e-5, 1.6687277881309547e-5, 1.669839639816716e-5, 1.6737781609450013e-5, 1.6737781609450094e-5, 1.6737781609450172e-5, 1.673778160945033e-5, 1.6737781609450386e-5, 1.6737781609450545e-5, 1.6737781609450708e-5, 1.674233695363805e-5, 1.6758755253865877e-5, 1.6758755253866013e-5, 1.6758755253866037e-5, 1.675875525386609e-5, 1.6758755253866284e-5, 1.6758755253866426e-5, 1.6758755253866437e-5, 1.6760651433461717e-5, 1.6767569937322963e-5, 1.6767569937323018e-5, 1.676756993732309e-5, 1.676756993732313e-5, 1.6767569937323217e-5, 1.676756993732322e-5, 1.676756993732325e-5, 1.676837067960395e-5, 1.6771618028001252e-5, 1.6771618028001283e-5, 1.6771618028001425e-5, 1.6771618028001445e-5, 1.6771618028001615e-5, 1.677161802800163e-5, 1.6771618028001852e-5, 1.6771872386192134e-5, 1.677283996758838e-5, 1.67728399675886e-5, 1.6772839967588646e-5, 1.6772839967588673e-5, 1.6772839967588734e-5, 1.677283996758875e-5, 1.6772839967588775e-5, 1.6772851178567514e-5, 1.6772863756951233e-5, 1.6772863756951324e-5, 1.6772863756951663e-5, 1.677286375695168e-5, 1.6772863756951795e-5, 1.6772863756951893e-5, 1.677286375695192e-5, 1.6772967122547897e-5, 1.6773090624274212e-5, 1.677309062427426e-5, 1.677309062427438e-5, 1.677309062427443e-5, 1.6773090624274456e-5, 1.67730906242745e-5, 1.6773090624274642e-5, 1.6773146602513248e-5, 1.67731829826136e-5, 1.677318298261367e-5, 1.6773182982613782e-5, 1.6773269399034703e-5, 1.6773476968025186e-5, 1.677347696802528e-5, 1.6773476968025338e-5, 1.6773476968025433e-5, 1.6773476968025457e-5, 1.6773476968025524e-5, 1.677347696802556e-5, 1.6773484431618582e-5, 1.6773494655752513e-5, 1.6773494655752604e-5, 1.6773494655752648e-5, 1.6773494655753014e-5, 1.6773494655753407e-5, 1.6773494655753468e-5, 1.6773494655753533e-5, 1.6773738495467747e-5, 1.677393936512993e-5, 1.6773939365129946e-5, 1.6773939365130075e-5, 1.6773939365130112e-5, 1.6773939365130295e-5, 1.6773939365130366e-5, 1.6773939365130407e-5, 1.6773975528233657e-5, 1.6774039605573926e-5, 1.677403960557396e-5, 1.6774039605573967e-5, 1.6774039605574e-5, 1.677403960557418e-5, 1.6774039605574238e-5, 1.677403960557431e-5, 1.677404747561558e-5, 1.6774054222289685e-5, 1.6774054222289777e-5, 1.6774054222289845e-5, 1.677405422228985e-5, 1.6774054222290014e-5, 1.677405422229002e-5, 1.6774054222290167e-5, 1.677422442965287e-5, 1.6774269649278975e-5, 1.6774269649279e-5, 1.6774269649279077e-5, 1.677426964927908e-5, 1.6774269649279114e-5, 1.6774269649279182e-5, 1.6774269649279192e-5, 1.6826077155436636e-5, 1.6838737520303086e-5, 1.6838737520303174e-5, 1.6838737520303208e-5, 1.6838737520303215e-5, 1.6838737520303337e-5, 1.6838737520303462e-5, 1.6838737520303527e-5, 1.6938020691042717e-5, 1.6938020691042856e-5, 1.6938020691042974e-5, 1.693802069104301e-5, 1.6938020691043103e-5, 1.6938020691043337e-5, 1.6938020691043405e-5, 1.6967834798814303e-5, 1.6967834798814336e-5, 1.6967834798814435e-5, 1.6967834798814526e-5, 1.6967834798814753e-5, 1.69678347988148e-5, 1.6967834798815048e-5, 1.6971964837260363e-5, 1.6982988832893406e-5, 1.6982988832893484e-5, 1.69829888328936e-5, 1.698298883289366e-5, 1.698298883289373e-5, 1.698298883289385e-5, 1.6982988832894003e-5, 1.7051661482764248e-5, 1.7055502381294465e-5, 1.710195122049013e-5, 1.7101951220490197e-5, 1.7101951220490312e-5, 1.7101951220490373e-5, 1.7101951220490407e-5, 1.710195122049055e-5, 1.71019512204906e-5, 1.7116860743024355e-5, 1.7149114454454677e-5, 1.7149114454454718e-5, 1.714911445445489e-5, 1.7149114454454907e-5, 1.7149114454454928e-5, 1.7149114454455023e-5, 1.714911445445506e-5, 1.7154352251917948e-5, 1.7169025141845512e-5, 1.7169025141845624e-5, 1.7169025141845712e-5, 1.716902514184587e-5, 1.7169025141846027e-5, 1.7169025141846068e-5, 1.716902514184611e-5, 1.7171124315848745e-5, 1.717795882951864e-5, 1.717795882951866e-5, 1.7177958829518836e-5, 1.7177958829518873e-5, 1.717795882951904e-5, 1.7177958829519256e-5, 1.7177958829519307e-5, 1.7178780994629778e-5, 1.7181873146498158e-5, 1.7181873146498568e-5, 1.7181873146498575e-5, 1.71818731464986e-5, 1.7181873146498643e-5, 1.718187314649871e-5, 1.71818731464988e-5, 1.7182143558306982e-5, 1.718286523416523e-5, 1.7182995270961487e-5, 1.718299527096165e-5, 1.7182995270961677e-5, 1.7182995270961728e-5, 1.7182995270961728e-5, 1.718299527096185e-5, 1.7182995270962094e-5, 1.7183018532657313e-5, 1.7183032664839568e-5, 1.7183032664839574e-5, 1.7183032664839676e-5, 1.7183162358081015e-5, 1.7183750334781593e-5, 1.718375033478167e-5, 1.718375033478167e-5, 1.718375033478173e-5, 1.7183750334781796e-5, 1.7183750334781878e-5, 1.718375033478212e-5, 1.718375752496079e-5, 1.7183772272572637e-5, 1.718377227257294e-5, 1.7183772272572973e-5, 1.718377227257304e-5, 1.7183772272573058e-5, 1.7183772272573135e-5, 1.718377227257319e-5, 1.7183788678182123e-5, 1.718379999165894e-5, 1.718379999165911e-5, 1.718379999165918e-5, 1.7183799991659195e-5, 1.7183799991659358e-5, 1.718379999165943e-5, 1.7183799991659456e-5, 1.7183847229373027e-5, 1.718387379194968e-5, 1.718387379194978e-5, 1.718387379194987e-5, 1.718387379194991e-5, 1.718387379195001e-5, 1.7183873791950072e-5, 1.7183873791950167e-5, 1.7183973780940304e-5, 1.7184024368138316e-5, 1.7184024368138814e-5, 1.7184024368138997e-5, 1.7184024368139085e-5, 1.718402436813912e-5, 1.718402436813923e-5, 1.718402436813923e-5, 1.7184131460020384e-5, 1.718420916836813e-5, 1.7184209168368187e-5, 1.718420916836822e-5, 1.7184209168368593e-5, 1.7184209168368654e-5, 1.7184209168368776e-5, 1.7184209168368864e-5, 1.7184275155189923e-5, 1.7184321584642152e-5, 1.7184321584642206e-5, 1.7184321584642328e-5, 1.718432158464233e-5, 1.7184321584642528e-5, 1.7184321584642586e-5, 1.718432158464262e-5, 1.7184419028916268e-5, 1.718445537603968e-5, 1.718445537603982e-5, 1.718445537603988e-5, 1.7184455376039953e-5, 1.7184455376040214e-5, 1.7184455376040268e-5, 1.7184455376040292e-5, 1.733707599914928e-5, 1.7337075999149323e-5, 1.733707599914938e-5, 1.733707599914953e-5, 1.733707599914963e-5, 1.7337075999149696e-5, 1.733707599914987e-5, 1.7385237459913725e-5, 1.7385237459913803e-5, 1.7385237459913844e-5, 1.7385237459913946e-5, 1.738523745991411e-5, 1.7385237459914417e-5, 1.73852374599145e-5, 1.741635629641979e-5, 1.7506106374208515e-5, 1.7506106374208715e-5, 1.7506106374208753e-5, 1.7506106374208922e-5, 1.7506106374209224e-5, 1.7506106374209345e-5, 1.750610637420935e-5, 1.7517337263673107e-5, 1.7550691631104987e-5, 1.7550691631105008e-5, 1.7550691631105106e-5, 1.7550691631105157e-5, 1.7550691631105245e-5, 1.75506916311053e-5, 1.7550691631105387e-5, 1.7555357697827193e-5, 1.7569634458283796e-5, 1.7569634458283824e-5, 1.7569634458283908e-5, 1.7569634458283946e-5, 1.7569634458284044e-5, 1.7569634458284057e-5, 1.7569634458284305e-5, 1.757174149241354e-5, 1.7578952841150382e-5, 1.7578952841150406e-5, 1.7578952841150514e-5, 1.7578952841150725e-5, 1.7578952841150904e-5, 1.7578952841150958e-5, 1.7578952841151382e-5, 1.757978504239565e-5, 1.758321083343552e-5, 1.7583210833435536e-5, 1.758321083343563e-5, 1.7583210833435777e-5, 1.7583210833436014e-5, 1.7583210833436275e-5, 1.758321083343635e-5, 1.758344434812009e-5, 1.7584479592811512e-5, 1.7584479592811634e-5, 1.7584479592811777e-5, 1.7584479592811837e-5, 1.7584479592811868e-5, 1.7584479592811953e-5, 1.758447959281197e-5, 1.7584494818361232e-5, 1.7584540330718458e-5, 1.7584540330718485e-5, 1.7584540330718556e-5, 1.7584540330718648e-5, 1.7584540330718753e-5, 1.75845403307188e-5, 1.7584540330718922e-5, 1.7584542225423686e-5, 1.758454428572728e-5, 1.758454428572731e-5, 1.7584544285727376e-5, 1.7584544285727572e-5, 1.758454428572759e-5, 1.758454428572762e-5, 1.7584544285727674e-5, 1.758455550723e-5, 1.7584558775089458e-5, 1.758455877508948e-5, 1.75845587750895e-5, 1.7584631852769294e-5, 1.758469804010672e-5, 1.7584698040106884e-5, 1.7584698040107e-5, 1.758469804010703e-5, 1.7584698040107046e-5, 1.758469804010716e-5, 1.758469804010742e-5, 1.758470918600568e-5, 1.7584722333222312e-5, 1.7584722333222465e-5, 1.7584722333222478e-5, 1.7584722333222546e-5, 1.7584722333222583e-5, 1.7584722333222766e-5, 1.7584722333222997e-5, 1.7585102192665634e-5, 1.7585356453375854e-5, 1.75853564533759e-5, 1.7585356453375915e-5, 1.7585356453375997e-5, 1.758535645337612e-5, 1.7585356453376126e-5, 1.7585356453376305e-5, 1.7585359139660473e-5, 1.75853617048819e-5, 1.7585361704882038e-5, 1.7585361704882177e-5, 1.758536170488218e-5, 1.758536170488221e-5, 1.758536170488222e-5, 1.7585361704882315e-5, 1.7585485853492334e-5, 1.7585560392860044e-5, 1.7585560392860122e-5, 1.7585560392860176e-5, 1.7585560392860373e-5, 1.758556039286038e-5, 1.758556039286058e-5, 1.758556039286068e-5, 1.7585607771753537e-5, 1.758563553458548e-5, 1.7585635534585677e-5, 1.758563553458568e-5, 1.7585635534585724e-5, 1.75856355345858e-5, 1.7585635534585846e-5, 1.758563553458587e-5, 1.8669586005729493e-5, 1.866958600572975e-5, 1.866958600572979e-5, 1.866958600572988e-5, 1.866958600573016e-5, 1.8669586005730168e-5, 1.8669586005730273e-5, 1.8669960300519388e-5, 1.8671044665047266e-5, 1.8671044665047358e-5, 1.8671044665047385e-5, 1.8671044665048286e-5, 1.8671044665048876e-5, 1.8671044665048926e-5, 1.8671044665049093e-5, 1.8671387012705044e-5, 1.8671877797480514e-5, 1.8671877797480605e-5, 1.8671877797480687e-5, 1.867187779748079e-5, 1.8671877797480832e-5, 1.8671877797480907e-5, 1.8671877797480988e-5, 1.8672699091414092e-5, 1.867352634231087e-5, 1.867352634231103e-5, 1.867352634231143e-5, 1.86740459566478e-5, 1.86758535672799e-5, 1.867585356727998e-5, 1.8675853567280028e-5, 1.8675853567280028e-5, 1.8675853567280164e-5, 1.8675853567280194e-5, 1.8675853567280597e-5, 1.8676333759867012e-5, 1.86771486491287e-5, 1.8677148649128757e-5, 1.867714864912885e-5, 1.8677148649129302e-5, 1.8677148649129438e-5, 1.867714864912944e-5, 1.8677148649129468e-5, 1.867789185241189e-5, 1.8679307468708045e-5, 1.867930746870816e-5, 1.8679307468708184e-5, 1.8679307468708197e-5, 1.8679307468708238e-5, 1.867930746870839e-5, 1.8679307468708597e-5, 1.8679857060321647e-5, 1.8681172586079735e-5, 1.8681172586079786e-5, 1.8681172586079874e-5, 1.8681172586079904e-5, 1.8681172586079908e-5, 1.8681172586079925e-5, 1.8681172586079996e-5, 1.8681541476410276e-5, 1.868245184735942e-5, 1.8682451847359518e-5, 1.8682451847359592e-5, 1.8682451847359603e-5, 1.868245184735961e-5, 1.868245184735966e-5, 1.8682451847359687e-5, 1.8682720438915957e-5, 1.8683521137010272e-5, 1.8683521137010397e-5, 1.868352113701072e-5, 1.868352113701081e-5, 1.8683521137010814e-5, 1.8683521137010844e-5, 1.8683521137010993e-5, 1.8683539338378022e-5, 1.868355809704912e-5, 1.8683558097049288e-5, 1.8683558097049298e-5, 1.8683558097049854e-5, 1.8683558097050017e-5, 1.8683558097050233e-5, 1.8683558097050616e-5, 1.8683911149333466e-5, 1.8684294281468005e-5, 1.8684294281468188e-5, 1.8684294281468337e-5, 1.8684294281468395e-5, 1.868429428146856e-5, 1.8684294281468652e-5, 1.868429428146882e-5, 1.8684465852357096e-5, 1.8684738161562107e-5, 1.8684738161562276e-5, 1.8684738161562283e-5, 1.86847381615623e-5, 1.868473816156246e-5, 1.8684738161562673e-5, 1.8684738161562788e-5, 1.8684869542746566e-5, 1.868508973890327e-5, 1.8685089738903342e-5, 1.8685089738903498e-5, 1.8685089738903843e-5, 1.8685089738903908e-5, 1.8685089738904087e-5, 1.86850897389042e-5, 1.8685158533074948e-5, 1.8685252773036424e-5, 1.868525277303652e-5, 1.8685252773036736e-5, 1.868525277303679e-5, 1.8685252773036977e-5, 1.8685252773037092e-5, 1.8685252773037126e-5, 1.868531314696119e-5, 1.8685377794501766e-5, 1.86853777945019e-5, 1.8685377794501986e-5, 1.8685377794502288e-5, 1.8685377794502443e-5, 1.868537779450246e-5, 1.8685377794502498e-5, 1.8685421246337486e-5, 1.868545318496711e-5, 1.868545318496752e-5, 1.868545318496762e-5, 1.8685453184967632e-5, 1.868545318496769e-5, 1.868545318496778e-5, 1.8685453184968357e-5, 1.8685644493690226e-5, 1.8685711069365147e-5, 1.868571106936539e-5, 1.868571106936544e-5, 1.8685711069365462e-5, 1.868571106936548e-5, 1.8685711069365662e-5, 1.868571106936571e-5, 1.8685951221563988e-5, 1.8686021186131717e-5, 1.8686021186131832e-5, 1.8686021186131883e-5, 1.8686021186131903e-5, 1.868602118613204e-5, 1.8686021186132184e-5, 1.868602118613224e-5, 1.8863042042170193e-5, 1.8874322792330326e-5, 1.9561710560790956e-5, 1.9561710560790984e-5, 1.956171056079114e-5, 1.9561710560791214e-5, 1.9561710560791285e-5, 1.9561710560791393e-5, 1.9561710560791587e-5, 1.959160516161019e-5, 1.959160516161028e-5, 1.9591605161610557e-5, 1.9591605161610638e-5, 1.959160516161074e-5, 1.959160516161085e-5, 1.959160516161106e-5, 1.9924712488335848e-5, 2.0773577185841645e-5, 2.077357718584194e-5, 2.0773577185842184e-5, 2.0773577185842286e-5, 2.077357718584251e-5, 2.0773577185842648e-5, 2.077357718584279e-5, 2.1304275251936998e-5, 2.1334315514962135e-5, 2.1702622628984926e-5, 2.2274926971762408e-5, 2.2274926971762432e-5, 2.2274926971762493e-5, 2.2274926971762547e-5, 2.2274926971762652e-5, 2.2274926971762676e-5, 2.227492697176277e-5, 2.301395250991005e-5, 2.34718827399398e-5, 2.347188273993987e-5, 2.347188273993988e-5, 2.3471882739939928e-5, 2.3471882739940328e-5, 2.3471882739940395e-5, 2.347188273994053e-5, 2.363083196714301e-5, 2.403236576868883e-5, 2.403236576868895e-5, 2.4032365768688994e-5, 2.4032365768689035e-5, 2.40323657686891e-5, 2.4032365768689333e-5, 2.4032365768689645e-5, 2.4223726848772127e-5, 2.4970891471199567e-5, 2.591580292683436e-5, 2.591580292683449e-5, 2.591580292683461e-5, 2.5915802926834685e-5, 2.5915802926834753e-5, 2.5915802926835068e-5, 2.5915802926835132e-5, 2.6047017179354913e-5, 2.650144250503052e-5, 2.6617056179505146e-5, 2.715776441755261e-5, 2.8423586739345168e-5, 3.042434402336967e-5, 3.313100048738068e-5, 3.3131000487380744e-5, 3.313100048738078e-5, 3.313100048738088e-5, 3.313100048738107e-5, 3.313100048738131e-5, 3.313100048738166e-5, 3.326250124658579e-5, 3.3262501246586034e-5, 3.3262501246586116e-5, 3.32625012465863e-5, 3.3262501246586305e-5, 3.326250124658663e-5, 3.326250124658672e-5, 3.33219240289205e-5, 3.332192402892068e-5, 3.3321924028920906e-5, 3.332192402892111e-5, 3.3321924028921116e-5, 3.332192402892115e-5, 3.332192402892132e-5, 3.34427448580397e-5, 3.3565364876769655e-5, 3.360932521184464e-5, 3.360932521184479e-5, 3.3609325211845005e-5, 3.360932521184504e-5, 3.360932521184512e-5, 3.360932521184542e-5, 3.360932521184547e-5, 3.362098350475038e-5, 3.378787589112423e-5, 3.3940150334522924e-5, 3.394015033452309e-5, 3.394015033452314e-5, 3.394015033452338e-5, 3.394015033452363e-5, 3.3940150334523866e-5, 3.394015033452418e-5, 3.4058883046246264e-5, 3.4183244333996464e-5, 3.418324433399649e-5, 3.418324433399663e-5, 3.418324433399775e-5, 3.418324433399775e-5, 3.4183244333998104e-5, 3.418324433399829e-5, 3.424714905602696e-5, 3.4480026110239156e-5, 3.448002611023976e-5, 3.448002611024011e-5, 3.448002611024026e-5, 3.44800261102404e-5, 3.448002611024049e-5, 3.448002611024107e-5, 3.4821057968075595e-5, 3.506412046020437e-5, 3.5064120460204645e-5, 3.5064120460205e-5, 3.506412046020517e-5, 3.5064120460205384e-5, 3.506412046020587e-5, 3.506412046020602e-5, 3.5064351671962406e-5, 3.5065369126784196e-5, 3.506536912678433e-5, 3.506536912678456e-5, 3.506536912678469e-5, 3.506536912678515e-5, 3.506536912678534e-5, 3.5065369126785464e-5, 3.5065432140291776e-5, 3.506551788575403e-5, 3.506551788575418e-5, 3.506551788575421e-5, 3.5065517885755145e-5, 3.506551788575618e-5, 3.506551788575633e-5, 3.5065517885756364e-5, 3.506577464987981e-5, 3.5066192181777815e-5, 3.506619218177811e-5, 3.506619218177817e-5, 3.5066192181778425e-5, 3.5066192181778534e-5, 3.506619218177858e-5, 3.506619218177879e-5, 3.5066309555884296e-5, 3.50664939619632e-5, 3.506649396196341e-5, 3.506649396196363e-5, 3.5066493961963766e-5, 3.506649396196382e-5, 3.506649396196429e-5, 3.50664939619643e-5, 3.50666147429361e-5, 3.5066767218363194e-5, 3.506676721836326e-5, 3.506676721836337e-5, 3.506676721836343e-5, 3.5066767218364034e-5, 3.5066767218364346e-5, 3.506676721836436e-5, 3.506686677427817e-5, 3.5066967870533034e-5, 3.5066967870533386e-5, 3.506696787053392e-5, 3.5066967870533976e-5, 3.5066967870534416e-5, 3.506696787053499e-5, 3.5066967870535385e-5, 3.506706218366738e-5, 3.506713440067521e-5, 3.506713440067542e-5, 3.5067134400675486e-5, 3.506713440067554e-5, 3.506713440067554e-5, 3.506713440067565e-5, 3.506713440067585e-5, 3.506722876960895e-5, 3.506727817669407e-5, 3.506727817669457e-5, 3.506727817669472e-5, 3.506727817669493e-5, 3.506727817669527e-5, 3.506727817669564e-5, 3.506727817669586e-5, 3.50929756174066e-5, 3.511204342130663e-5, 3.5112043421306723e-5, 3.511204342130687e-5, 3.511234831194741e-5, 3.511363862406319e-5, 3.511363862406334e-5, 3.511363862406341e-5, 3.5113638624063585e-5, 3.5113638624064e-5, 3.511363862406427e-5, 3.5113638624064696e-5, 3.5114211274203295e-5, 3.511500866843262e-5, 3.511500866843308e-5, 3.511500866843367e-5, 3.5115008668433936e-5, 3.5115008668433936e-5, 3.511500866843404e-5, 3.5115008668434194e-5, 3.511621121802807e-5, 3.5117543370629784e-5, 3.5117543370630326e-5, 3.511754337063039e-5, 3.5117543370630394e-5, 3.51175433706309e-5, 3.5117543370631085e-5, 3.511754337063121e-5, 3.511876267157888e-5, 3.511993043499941e-5, 3.511993043499943e-5, 3.51199304349995e-5, 3.511993043499968e-5, 3.511993043499974e-5, 3.5119930434999866e-5, 3.5119930435000524e-5, 3.51214716517647e-5, 3.512258567024824e-5, 3.512258567024827e-5, 3.512258567024834e-5, 3.5122585670248444e-5, 3.512258567024892e-5, 3.512258567024924e-5, 3.512258567024947e-5, 3.512535781943816e-5, 3.512681787306314e-5, 3.512681787306315e-5, 3.512681787306321e-5, 3.512681787306345e-5, 3.5126817873064455e-5, 3.51268178730647e-5, 3.512681787306482e-5, 3.5130509796245946e-5, 3.513203537165987e-5, 3.513203537166001e-5, 3.513203537166025e-5, 3.513203537166032e-5, 3.513203537166043e-5, 3.513203537166087e-5, 3.513203537166101e-5, 3.55287209633577e-5, 3.552872096335798e-5, 3.552872096335841e-5, 3.552872096335853e-5, 3.552872096335854e-5, 3.552872096335865e-5, 3.5528720963359064e-5, 3.552896319881981e-5, 3.5529618982604035e-5, 3.552961898260404e-5, 3.552961898260431e-5, 3.552961898260489e-5, 3.5529618982605e-5, 3.552961898260523e-5, 3.5529618982606136e-5, 3.55299179542897e-5, 3.5530936027485766e-5, 3.55309360274858e-5, 3.553093602748592e-5, 3.553093602748717e-5, 3.553093602748775e-5, 3.553093602748859e-5, 3.5530936027488686e-5, 3.5531045867913406e-5, 3.553125994848679e-5, 3.553125994848704e-5, 3.553125994848726e-5, 3.5531259948487414e-5, 3.553125994848798e-5, 3.553125994848821e-5, 3.553125994848872e-5, 3.553140179064652e-5, 3.553166920302003e-5, 3.553166920302039e-5, 3.5531669203020414e-5, 3.5531669203020475e-5, 3.553166920302087e-5, 3.553166920302111e-5, 3.553166920302125e-5, 3.553174023656729e-5, 3.553183856032935e-5, 3.553183856032965e-5, 3.5531838560329745e-5, 3.553183856032999e-5, 3.553183856033001e-5, 3.553183856033021e-5, 3.5531838560330565e-5, 3.55319194779472e-5, 3.5531999108074033e-5, 3.553199910807432e-5, 3.5531999108074325e-5, 3.553199910807458e-5, 3.553199910807486e-5, 3.5531999108074874e-5, 3.553199910807493e-5, 3.553209631755946e-5, 3.553217870295951e-5, 3.553217870295959e-5, 3.5532178702959865e-5, 3.553217870296008e-5, 3.55321787029602e-5, 3.553217870296024e-5, 3.5532178702960746e-5, 3.5532211298800435e-5, 3.553223489644981e-5, 3.5532234896450027e-5, 3.553223489645005e-5, 3.553223489645017e-5, 3.553223489645022e-5, 3.553223489645044e-5, 3.55322348964512e-5, 3.55586188056585e-5, 3.558000061581363e-5, 3.558000061581393e-5, 3.558000061581398e-5, 3.558003511153497e-5, 3.5580112316784354e-5, 3.5580112316784686e-5, 3.558011231678481e-5, 3.558011231678521e-5, 3.558011231678607e-5, 3.5580112316786576e-5, 3.5580112316786705e-5, 3.558079795213373e-5, 3.5581825747112916e-5, 3.558182574711297e-5, 3.558182574711309e-5, 3.5581825747113296e-5, 3.558182574711374e-5, 3.558182574711389e-5, 3.558182574711414e-5, 3.558222703376891e-5, 3.558265800618457e-5, 3.558265800618468e-5, 3.558265800618513e-5, 3.558265800618544e-5, 3.558265800618598e-5, 3.558265800618609e-5, 3.558265800618612e-5, 3.5583777425535685e-5, 3.55845323818185e-5, 3.558453238181869e-5, 3.5584532381818856e-5, 3.5584532381819405e-5, 3.558453238182084e-5, 3.558453238182122e-5, 3.5584532381821675e-5, 3.558685433567561e-5, 3.5588134232260316e-5, 3.558813423226072e-5, 3.5588134232260905e-5, 3.55881342322623e-5, 3.558813423226316e-5, 3.558813423226324e-5, 3.558813423226375e-5, 3.5591191620744796e-5, 3.5592736490982046e-5, 3.559273649098214e-5, 3.559273649098238e-5, 3.559273649098324e-5, 3.559273649098385e-5, 3.559273649098389e-5, 3.5592736490983916e-5, 3.5596816136765946e-5, 3.559844832692395e-5, 3.5598448326923986e-5, 3.5598448326924115e-5, 3.559844832692414e-5, 3.559844832692442e-5, 3.559844832692454e-5, 3.5598448326924786e-5, 3.5630634406423956e-5, 3.566160750261558e-5, 3.566160750261603e-5, 3.566160750261629e-5, 3.5661607502617096e-5, 3.566160750261857e-5, 3.566160750261888e-5, 3.566160750261918e-5, 3.590875046060417e-5, 3.590875046060448e-5, 3.5908750460604523e-5, 3.590875046060467e-5, 3.590875046060476e-5, 3.590875046060497e-5, 3.5908750460604984e-5, 3.590890683849662e-5, 3.5909406012746553e-5, 3.5909406012746635e-5, 3.590940601274681e-5, 3.590940601274742e-5, 3.590940601274809e-5, 3.590940601274818e-5, 3.590940601274819e-5, 3.5909533178305764e-5, 3.59098366720343e-5, 3.590983667203485e-5, 3.5909836672034956e-5, 3.5909836672035884e-5, 3.5909836672036453e-5, 3.59098366720366e-5, 3.5909836672036745e-5, 3.590991284348423e-5, 3.5910003180760966e-5, 3.591000318076122e-5, 3.5910003180761386e-5, 3.591000318076155e-5, 3.591000318076221e-5, 3.591000318076269e-5, 3.591000318076291e-5, 3.5910246853573254e-5, 3.591056395525391e-5, 3.591056395525424e-5, 3.5910563955254275e-5, 3.591056395525441e-5, 3.5910563955254505e-5, 3.5910563955254566e-5, 3.591056395525486e-5, 3.591067317131868e-5, 3.591087539725879e-5, 3.591087539725925e-5, 3.591087539725936e-5, 3.591087539725966e-5, 3.591087539725973e-5, 3.591087539725993e-5, 3.591087539726004e-5, 3.591090868290429e-5, 3.591095007102645e-5, 3.5910950071026774e-5, 3.591095007102682e-5, 3.591095007102697e-5, 3.591095007102698e-5, 3.591095007102753e-5, 3.591095007102758e-5, 3.591098988234854e-5, 3.591101834101353e-5, 3.591101834101372e-5, 3.591101834101384e-5, 3.591101834101393e-5, 3.591101834101395e-5, 3.591101834101423e-5, 3.591101834101461e-5, 3.591117713283772e-5, 3.591123226157729e-5, 3.5911232261577364e-5, 3.591123226157756e-5, 3.591123226157772e-5, 3.591123226157788e-5, 3.5911232261577886e-5, 3.591123226157843e-5, 3.593736005960015e-5, 3.5956640198207814e-5, 3.5956640198208214e-5, 3.595664019820836e-5, 3.5956640198208464e-5, 3.5956640198209135e-5, 3.595664019820916e-5, 3.595664019820996e-5, 3.595714720247414e-5, 3.595809408896531e-5, 3.595809408896533e-5, 3.595809408896534e-5, 3.595809408896537e-5, 3.595809408896545e-5, 3.5958094088965545e-5, 3.5958094088965755e-5, 3.595841013984059e-5, 3.595857183240665e-5, 3.5958571832406734e-5, 3.595857183240679e-5, 3.595972541455752e-5, 3.596082789711139e-5, 3.596082789711166e-5, 3.5960827897111666e-5, 3.5960827897112175e-5, 3.596082789711225e-5, 3.596082789711238e-5, 3.5960827897112466e-5, 3.596283734800869e-5, 3.596446644382876e-5, 3.5964466443829325e-5, 3.596446644382955e-5, 3.596446644383077e-5, 3.5964466443831595e-5, 3.5964466443831784e-5, 3.5964466443832306e-5, 3.596634161702655e-5, 3.596771169766288e-5, 3.5967711697663045e-5, 3.5967711697663195e-5, 3.596771169766381e-5, 3.596771169766534e-5, 3.5967711697665505e-5, 3.5967711697665634e-5, 3.597083518501116e-5, 3.597251724238662e-5, 3.5972517242386685e-5, 3.597251724238725e-5, 3.597251724238756e-5, 3.597251724238868e-5, 3.5972517242388935e-5, 3.597251724238918e-5, 3.597641905659947e-5, 3.5978072860254485e-5, 3.597807286025455e-5, 3.5978072860254566e-5, 3.597807286025461e-5, 3.5978072860254756e-5, 3.5978072860254905e-5, 3.5978072860255196e-5, 3.6180430597778594e-5, 3.618043059777886e-5, 3.6180430597778987e-5, 3.618043059777944e-5, 3.6180430597779745e-5, 3.618043059777998e-5, 3.618043059778015e-5, 3.629434125993366e-5, 3.629434125993374e-5, 3.629434125993374e-5, 3.6294341259933976e-5, 3.629434125993417e-5, 3.629434125993424e-5, 3.629434125993434e-5, 3.629458688236753e-5, 3.629521624574537e-5, 3.62952162457454e-5, 3.629521624574591e-5, 3.629521624574609e-5, 3.629521624574617e-5, 3.629521624574622e-5, 3.629521624574662e-5, 3.6295544687345504e-5, 3.629673379322557e-5, 3.629673379322569e-5, 3.629673379322584e-5, 3.629673379322591e-5, 3.6296733793226105e-5, 3.629673379322616e-5, 3.629673379322641e-5, 3.629679049714194e-5, 3.6296866394001334e-5, 3.629686639400162e-5, 3.629686639400206e-5, 3.62968663940021e-5, 3.629686639400218e-5, 3.629686639400224e-5, 3.6296866394002303e-5, 3.629709371797576e-5, 3.6297495615436666e-5, 3.629749561543716e-5, 3.629749561543723e-5, 3.6297495615437276e-5, 3.629749561543729e-5, 3.629749561543733e-5, 3.629749561543735e-5, 3.629755169869144e-5, 3.6297649559529586e-5, 3.6297649559530026e-5, 3.6297649559530216e-5, 3.629764955953029e-5, 3.629764955953058e-5, 3.629764955953066e-5, 3.629764955953066e-5, 3.629767494416143e-5, 3.629769887384082e-5, 3.629769887384082e-5, 3.629769887384111e-5, 3.629769887384117e-5, 3.629769887384158e-5, 3.629769887384168e-5, 3.629769887384174e-5, 3.6297876672055074e-5, 3.6298005778895534e-5, 3.629800577889576e-5, 3.629800577889577e-5, 3.6298005778895785e-5, 3.629800577889594e-5, 3.6298005778895995e-5, 3.629800577889625e-5, 3.629800817766706e-5, 3.6298010522607685e-5, 3.629801052260799e-5, 3.629801052260816e-5, 3.629801052260831e-5, 3.6298010522608444e-5, 3.6298010522608606e-5, 3.629801052260886e-5, 3.6324481271101184e-5, 3.634408011406589e-5, 3.634408011406608e-5, 3.634408011406628e-5, 3.6344264462987604e-5, 3.634492910079944e-5, 3.6344929100799514e-5, 3.6344929100799595e-5, 3.63449291008001e-5, 3.634492910080026e-5, 3.6344929100800334e-5, 3.634492910080044e-5, 3.6345347065496e-5, 3.634584194258373e-5, 3.6345841942584165e-5, 3.634584194258442e-5, 3.634584194258472e-5, 3.634584194258473e-5, 3.634584194258501e-5, 3.634584194258554e-5, 3.634700431205132e-5, 3.634796204454855e-5, 3.634796204454885e-5, 3.634796204454912e-5, 3.634796204454912e-5, 3.634796204454923e-5, 3.6347962044549306e-5, 3.634796204454973e-5, 3.634995507513159e-5, 3.6351394312624434e-5, 3.635139431262463e-5, 3.6351394312624834e-5, 3.635139431262521e-5, 3.635139431262524e-5, 3.63513943126255e-5, 3.635139431262591e-5, 3.635409819487364e-5, 3.6355986342988025e-5, 3.635598634298809e-5, 3.635598634298826e-5, 3.635598634298831e-5, 3.6355986342988534e-5, 3.635598634298885e-5, 3.635598634298899e-5, 3.6358233548725816e-5, 3.635968065610364e-5, 3.635968065610375e-5, 3.635968065610443e-5, 3.6359680656104866e-5, 3.6359680656105035e-5, 3.635968065610513e-5, 3.6359680656105353e-5, 3.636355683228913e-5, 3.636516656588692e-5, 3.63651665658871e-5, 3.63651665658876e-5, 3.636516656588782e-5, 3.6365166565887895e-5, 3.636516656588824e-5, 3.636516656588831e-5, 3.644965848685084e-5, 3.675220056043971e-5, 3.6752200560440105e-5, 3.6752200560440254e-5, 3.675220056044033e-5, 3.67522005604404e-5, 3.6752200560440404e-5, 3.67522005604406e-5, 3.675233877780531e-5, 3.675269229568109e-5, 3.6752692295681295e-5, 3.67526922956814e-5, 3.675269229568196e-5, 3.675269229568227e-5, 3.6752692295682664e-5, 3.6752692295682894e-5, 3.675279738145035e-5, 3.6752932666484574e-5, 3.6752932666485346e-5, 3.675293266648558e-5, 3.6752932666485915e-5, 3.675293266648647e-5, 3.6752932666486945e-5, 3.67529326664874e-5, 3.675327791283269e-5, 3.675389202973652e-5, 3.6753892029736615e-5, 3.675389202973679e-5, 3.675389202973691e-5, 3.6753892029737116e-5, 3.675389202973726e-5, 3.67538920297374e-5, 3.67539363899911e-5, 3.675398483136909e-5, 3.6753984831369204e-5, 3.675398483136932e-5, 3.6753984831369374e-5, 3.675398483136948e-5, 3.6753984831369957e-5, 3.675398483137001e-5, 3.675428014463852e-5, 3.675455706146357e-5, 3.6754557061464056e-5, 3.675455706146412e-5, 3.675455706146416e-5, 3.675455706146537e-5, 3.675455706146594e-5, 3.6754557061465994e-5, 3.6754703744658706e-5, 3.6754931516318434e-5, 3.6754931516319085e-5, 3.67549315163192e-5, 3.6754931516319234e-5, 3.675493151631927e-5, 3.675493151631946e-5, 3.6754931516319606e-5, 3.67549360356258e-5, 3.675494055662695e-5, 3.67549405566274e-5, 3.6754940556627544e-5, 3.6754940556627795e-5, 3.675494055662817e-5, 3.675494055662826e-5, 3.675494055662846e-5, 3.675508442681545e-5, 3.675513900418946e-5, 3.67551390041898e-5, 3.675513900418983e-5, 3.675513900419006e-5, 3.675513900419011e-5, 3.6755139004190634e-5, 3.675513900419071e-5, 3.6781945750881884e-5, 3.67980971508191e-5, 3.679809715081921e-5, 3.679809715081966e-5, 3.679886454014774e-5, 3.6804339851693325e-5, 3.6804339851693556e-5, 3.680433985169364e-5, 3.680433985169381e-5, 3.680433985169387e-5, 3.680433985169412e-5, 3.68043398516943e-5, 3.680437853630215e-5, 3.6804417311420766e-5, 3.680441731142099e-5, 3.6804417311421166e-5, 3.680441731142127e-5, 3.680441731142164e-5, 3.680441731142212e-5, 3.680441731142221e-5, 3.680567110556336e-5, 3.680666385160534e-5, 3.6806663851605556e-5, 3.68066638516059e-5, 3.6806663851606004e-5, 3.680666385160602e-5, 3.6806663851606505e-5, 3.680666385160688e-5, 3.680798011573768e-5, 3.680904015435728e-5, 3.680904015435748e-5, 3.68090401543575e-5, 3.680904015435798e-5, 3.680904015435814e-5, 3.6809040154358204e-5, 3.680904015435828e-5, 3.681123930155494e-5, 3.6812640206210215e-5, 3.681264020621051e-5, 3.681264020621058e-5, 3.68126402062108e-5, 3.681264020621095e-5, 3.68126402062111e-5, 3.6812640206211624e-5, 3.6815568302287214e-5, 3.681714318312964e-5, 3.681714318312988e-5, 3.6817143183130056e-5, 3.681714318313037e-5, 3.681714318313051e-5, 3.6817143183130666e-5, 3.681714318313082e-5, 3.68211908916034e-5, 3.682284077153311e-5, 3.682284077153339e-5, 3.682284077153387e-5, 3.6822840771534046e-5, 3.682284077153447e-5, 3.6822840771534656e-5, 3.682284077153466e-5, 3.71343374268011e-5, 3.722698044558583e-5, 3.7226980445585924e-5, 3.722698044558621e-5, 3.722698044558664e-5, 3.722698044558775e-5, 3.722698044558793e-5, 3.7226980445588e-5, 3.730081237483575e-5, 3.7300812374836e-5, 3.730081237483625e-5, 3.7300812374836284e-5, 3.73008123748363e-5, 3.730081237483634e-5, 3.7300812374836365e-5, 3.730113378292045e-5, 3.730235763958213e-5, 3.7302357639582476e-5, 3.730235763958253e-5, 3.730235763958257e-5, 3.730235763958268e-5, 3.730235763958288e-5, 3.7302357639583465e-5, 3.730261668489242e-5, 3.730395321941288e-5, 3.7303953219413116e-5, 3.7303953219413225e-5, 3.7303953219413536e-5, 3.730395321941366e-5, 3.73039532194139e-5, 3.730395321941423e-5, 3.730398428764507e-5, 3.730404792169502e-5, 3.730404792169529e-5, 3.7304047921695563e-5, 3.7304047921695746e-5, 3.7304047921695746e-5, 3.730404792169607e-5, 3.730404792169623e-5, 3.730409238694834e-5, 3.730417614764082e-5, 3.730417614764099e-5, 3.73041761476413e-5, 3.730417614764178e-5, 3.7304176147641795e-5, 3.73041761476418e-5, 3.73041761476428e-5, 3.730419340702719e-5, 3.7304212190596e-5, 3.730421219059605e-5, 3.730421219059624e-5, 3.730421219059628e-5, 3.730421219059634e-5, 3.730421219059656e-5, 3.730421219059729e-5, 3.730427673040404e-5, 3.730434175088696e-5, 3.730434175088698e-5, 3.730434175088714e-5, 3.730434175088716e-5, 3.7304341750887356e-5, 3.7304341750887755e-5, 3.730434175088777e-5, 3.730435698344586e-5, 3.730437398681033e-5, 3.7304373986810556e-5, 3.730437398681062e-5, 3.7304373986810806e-5, 3.730437398681098e-5, 3.730437398681123e-5, 3.730437398681129e-5, 3.7304386864631354e-5, 3.730439548682512e-5, 3.7304395486825235e-5, 3.730439548682554e-5, 3.73043954868256e-5, 3.730439548682613e-5, 3.730439548682621e-5, 3.7304395486826726e-5, 3.733211064100324e-5, 3.7348790731392706e-5, 3.7348790731392835e-5, 3.734879073139288e-5, 3.734934051847516e-5, 3.73513585238883e-5, 3.735135852388846e-5, 3.735135852388906e-5, 3.735135852388989e-5, 3.735135852389051e-5, 3.735135852389058e-5, 3.735135852389089e-5, 3.735284212775913e-5, 3.7355615204182326e-5, 3.735561520418241e-5, 3.735561520418261e-5, 3.7355615204182726e-5, 3.7355615204182875e-5, 3.735561520418295e-5, 3.7355615204183045e-5, 3.735686621755481e-5, 3.735905279932737e-5, 3.735905279932753e-5, 3.735905279932784e-5, 3.7359052799328216e-5, 3.7359052799328284e-5, 3.735905279932853e-5, 3.735905279932877e-5, 3.735967958547514e-5, 3.7360378309943866e-5, 3.7360378309943927e-5, 3.736037830994409e-5, 3.736037830994438e-5, 3.7360378309944855e-5, 3.736037830994502e-5, 3.7360378309945404e-5, 3.73618754915323e-5, 3.736286584664636e-5, 3.7362865846646536e-5, 3.736286584664659e-5, 3.736286584664676e-5, 3.7362865846647105e-5, 3.736286584664737e-5, 3.7362865846647654e-5, 3.736630817773387e-5, 3.736792084171684e-5, 3.7367920841717185e-5, 3.736792084171732e-5, 3.736792084171784e-5, 3.736792084171876e-5, 3.736792084171888e-5, 3.7367920841719225e-5, 3.737245091424705e-5, 3.737421244501606e-5, 3.737421244501614e-5, 3.737421244501616e-5, 3.737421244501688e-5, 3.73742124450174e-5, 3.7374212445017665e-5, 3.7374212445018e-5, 3.791346490490217e-5, 3.7913464904902784e-5, 3.7913464904902804e-5, 3.79134649049029e-5, 3.791346490490335e-5, 3.791346490490369e-5, 3.791346490490413e-5, 3.8108034518249934e-5, 3.8256762955338644e-5, 3.8256762955338793e-5, 3.825676295533894e-5, 3.8256762955339295e-5, 3.825676295533944e-5, 3.825676295533966e-5, 3.8256762955339986e-5, 3.825683137624397e-5, 3.8256928570770115e-5, 3.825692857077049e-5, 3.8256928570770806e-5, 3.825692857077097e-5, 3.825692857077119e-5, 3.825692857077156e-5, 3.8256928570771945e-5, 3.825713873981244e-5, 3.825738612618313e-5, 3.825738612618351e-5, 3.825738612618354e-5, 3.825738612618355e-5, 3.825738612618387e-5, 3.8257386126184035e-5, 3.825738612618404e-5, 3.825774522997699e-5, 3.8258774917967893e-5, 3.825877491796793e-5, 3.8258774917968185e-5, 3.8258774917968415e-5, 3.825877491796857e-5, 3.825877491796889e-5, 3.8258774917969025e-5, 3.825879196119773e-5, 3.8258823443263044e-5, 3.825882344326325e-5, 3.825882344326357e-5, 3.825882344326401e-5, 3.825882344326407e-5, 3.825882344326436e-5, 3.82588234432654e-5, 3.8258842542489734e-5, 3.825887076181691e-5, 3.825887076181709e-5, 3.8258870761817454e-5, 3.8258870761817555e-5, 3.825887076181773e-5, 3.8258870761817847e-5, 3.8258870761817914e-5, 3.825887489661174e-5, 3.825887866510858e-5, 3.825887866510896e-5, 3.8258878665109156e-5, 3.8258878665109495e-5, 3.825887866510951e-5, 3.82588786651097e-5, 3.825887866510988e-5, 3.82590508490527e-5, 3.8259118299712136e-5, 3.825911829971238e-5, 3.8259118299712705e-5, 3.825911829971272e-5, 3.825911829971297e-5, 3.825911829971325e-5, 3.825911829971338e-5, 3.825918470113762e-5, 3.825921990556535e-5, 3.825921990556546e-5, 3.82592199055657e-5, 3.8259219905565736e-5, 3.8259219905565756e-5, 3.825921990556635e-5, 3.825921990556643e-5, 3.828782947779092e-5, 3.830905674919527e-5, 3.830905674919533e-5, 3.830905674919547e-5, 3.8309056749195826e-5, 3.830905674919609e-5, 3.830905674919628e-5, 3.8309056749196375e-5, 3.8309488973465656e-5, 3.831025675577851e-5, 3.83102567557787e-5, 3.831025675577881e-5, 3.831025675577894e-5, 3.831025675577895e-5, 3.831025675577905e-5, 3.831025675577931e-5, 3.8310743079731154e-5, 3.8311033698223756e-5, 3.831103369822376e-5, 3.8311033698223804e-5, 3.8311679140495604e-5, 3.8312436204481615e-5, 3.831243620448167e-5, 3.8312436204481805e-5, 3.831243620448199e-5, 3.831243620448208e-5, 3.831243620448218e-5, 3.831243620448224e-5, 3.831439505708289e-5, 3.8315764165590555e-5, 3.8315764165590894e-5, 3.8315764165590975e-5, 3.8315764165590996e-5, 3.831576416559124e-5, 3.831576416559131e-5, 3.831576416559138e-5, 3.831825181127599e-5, 3.83199007029627e-5, 3.831990070296346e-5, 3.8319900702963585e-5, 3.831990070296364e-5, 3.83199007029638e-5, 3.831990070296382e-5, 3.83199007029644e-5, 3.832274129418054e-5, 3.832438157549903e-5, 3.832438157549965e-5, 3.8324381575499734e-5, 3.8324381575500195e-5, 3.8324381575500385e-5, 3.832438157550076e-5, 3.832438157550095e-5, 3.8328213901256655e-5, 3.8329847303369426e-5, 3.832984730337029e-5, 3.832984730337069e-5, 3.832984730337127e-5, 3.832984730337149e-5, 3.8329847303371764e-5, 3.832984730337232e-5, 3.8564480770669293e-5, 3.8723117466918965e-5, 3.872311746691921e-5, 3.872311746691929e-5, 3.87231174669194e-5, 3.872311746691946e-5, 3.872311746691977e-5, 3.872311746692035e-5, 3.9191133465528666e-5, 3.9191133465528686e-5, 3.9191133465528876e-5, 3.919113346553004e-5, 3.919113346553052e-5, 3.919113346553095e-5, 3.9191133465531085e-5, 3.919146001904022e-5, 3.919272918179797e-5, 3.9192729181798195e-5, 3.919272918179827e-5, 3.9192729181798296e-5, 3.91927291817983e-5, 3.91927291817984e-5, 3.919272918179846e-5, 3.91929719289051e-5, 3.919379322291837e-5, 3.919379322291876e-5, 3.9193793222918965e-5, 3.919379322291922e-5, 3.9193793222919426e-5, 3.919379322291976e-5, 3.9193793222919886e-5, 3.919392426283341e-5, 3.919425407529126e-5, 3.919425407529152e-5, 3.9194254075291545e-5, 3.919425407529186e-5, 3.919425407529278e-5, 3.9194254075292975e-5, 3.919425407529321e-5, 3.9194354792989954e-5, 3.9194562148354584e-5, 3.91945621483547e-5, 3.919456214835489e-5, 3.919456214835495e-5, 3.9194562148354956e-5, 3.9194562148355024e-5, 3.9194562148355505e-5, 3.919462214663934e-5, 3.9194705642053555e-5, 3.9194705642053826e-5, 3.919470564205395e-5, 3.9194705642053955e-5, 3.919470564205413e-5, 3.919470564205426e-5, 3.919470564205513e-5, 3.919475521416304e-5, 3.919479756711505e-5, 3.919479756711531e-5, 3.9194797567115377e-5, 3.919479756711578e-5, 3.9194797567115844e-5, 3.919479756711643e-5, 3.919479756711713e-5, 3.919496700796804e-5, 3.919507557632093e-5, 3.9195075576321284e-5, 3.9195075576321575e-5, 3.919507557632158e-5, 3.9195075576321656e-5, 3.919507557632171e-5, 3.91950755763218e-5, 3.9195108210870806e-5, 3.9195132749181046e-5, 3.9195132749181154e-5, 3.919513274918116e-5, 3.919513274918149e-5, 3.919513274918161e-5, 3.9195132749181615e-5, 3.919513274918231e-5, 3.922493233011558e-5, 3.9247665227966495e-5, 3.924766522796672e-5, 3.9247665227967e-5, 3.924776950876805e-5, 3.924806799103426e-5, 3.924806799103453e-5, 3.924806799103518e-5, 3.9248067991035885e-5, 3.924806799103598e-5, 3.924806799103602e-5, 3.9248067991037064e-5, 3.9248565271304246e-5, 3.924915437404635e-5, 3.9249154374046474e-5, 3.9249154374046495e-5, 3.924915437404666e-5, 3.924915437404694e-5, 3.9249154374046955e-5, 3.924915437404728e-5, 3.9250148979412e-5, 3.92510769977766e-5, 3.92510769977767e-5, 3.925107699777676e-5, 3.9251076997776774e-5, 3.9251076997777004e-5, 3.925107699777733e-5, 3.925107699777752e-5, 3.925245609448756e-5, 3.925352423206442e-5, 3.9253524232064475e-5, 3.925352423206449e-5, 3.9253524232064874e-5, 3.925352423206496e-5, 3.925352423206541e-5, 3.9253524232065464e-5, 3.92557367594279e-5, 3.925711729593246e-5, 3.925711729593264e-5, 3.9257117295932685e-5, 3.9257117295932767e-5, 3.925711729593311e-5, 3.925711729593321e-5, 3.9257117295933336e-5, 3.92601783218935e-5, 3.9261769746527636e-5, 3.926176974652768e-5, 3.926176974652781e-5, 3.926176974652787e-5, 3.9261769746528076e-5, 3.9261769746528354e-5, 3.9261769746528774e-5, 3.9266240666200916e-5, 3.9267990354045464e-5, 3.926799035404602e-5, 3.9267990354046135e-5, 3.926799035404657e-5, 3.92679903540475e-5, 3.926799035404759e-5, 3.9267990354047734e-5, 4.001895123419436e-5, 4.010735172832828e-5, 4.0107351728328545e-5, 4.010735172832861e-5, 4.0107351728328694e-5, 4.0107351728328714e-5, 4.010735172832967e-5, 4.010735172832972e-5, 4.010750347794455e-5, 4.0107968228985615e-5, 4.010796822898593e-5, 4.010796822898599e-5, 4.010796822898627e-5, 4.010796822898637e-5, 4.010796822898638e-5, 4.0107968228986875e-5, 4.0107995361258945e-5, 4.0108023625194726e-5, 4.0108023625195106e-5, 4.0108023625195275e-5, 4.010802362519543e-5, 4.010802362519546e-5, 4.010802362519555e-5, 4.010802362519589e-5, 4.0108433789752955e-5, 4.010880363524591e-5, 4.010880363524631e-5, 4.010880363524636e-5, 4.010880363524645e-5, 4.0108803635246575e-5, 4.010880363524683e-5, 4.0108803635247334e-5, 4.01091513105236e-5, 4.010966911182374e-5, 4.0109669111823957e-5, 4.0109669111824316e-5, 4.0109669111824424e-5, 4.010966911182446e-5, 4.010966911182515e-5, 4.010966911182524e-5, 4.010982605984298e-5, 4.011003679699449e-5, 4.0110036796994556e-5, 4.0110036796994875e-5, 4.011003679699495e-5, 4.0110036796995044e-5, 4.011003679699518e-5, 4.011003679699522e-5, 4.011022032146908e-5, 4.011043306046977e-5, 4.0110433060470134e-5, 4.011043306047051e-5, 4.0110433060471313e-5, 4.0110433060471435e-5, 4.011043306047158e-5, 4.011043306047159e-5, 4.0110521194128824e-5, 4.011060739187018e-5, 4.011060739187059e-5, 4.011060739187074e-5, 4.011060739187082e-5, 4.0110607391870876e-5, 4.0110607391871087e-5, 4.011060739187113e-5, 4.011068627101918e-5, 4.011073466951269e-5, 4.011073466951271e-5, 4.011073466951322e-5, 4.0110734669513295e-5, 4.0110734669513464e-5, 4.0110734669513634e-5, 4.0110734669513715e-5, 4.0140069405244625e-5, 4.016017705616717e-5, 4.0160177056167194e-5, 4.016017705616746e-5, 4.016017705616757e-5, 4.01601770561678e-5, 4.016017705616795e-5, 4.016017705616819e-5, 4.016088889567636e-5, 4.0161806193005614e-5, 4.016180619300637e-5, 4.0161806193006386e-5, 4.016218900197716e-5, 4.016296583566212e-5, 4.0162965835662423e-5, 4.016296583566252e-5, 4.0162965835662884e-5, 4.016296583566299e-5, 4.016296583566306e-5, 4.016296583566325e-5, 4.0164878890060546e-5, 4.016703349068557e-5, 4.016703349068572e-5, 4.016703349068601e-5, 4.0167033490686065e-5, 4.016703349068621e-5, 4.016703349068662e-5, 4.016703349068671e-5, 4.016818966917053e-5, 4.0169304571984415e-5, 4.016930457198445e-5, 4.016930457198495e-5, 4.016930457198497e-5, 4.016930457198499e-5, 4.016930457198506e-5, 4.01693045719854e-5, 4.0171767134646645e-5, 4.017336042622524e-5, 4.0173360426225896e-5, 4.01733604262259e-5, 4.017336042622646e-5, 4.017336042622691e-5, 4.017336042622692e-5, 4.01733604262279e-5, 4.017666235415311e-5, 4.0178427500403296e-5, 4.017842750040338e-5, 4.017842750040373e-5, 4.017842750040395e-5, 4.0178427500404326e-5, 4.017842750040464e-5, 4.017842750040472e-5, 4.0183246973490275e-5, 4.018515233139396e-5, 4.018515233139402e-5, 4.018515233139422e-5, 4.018515233139431e-5, 4.0185152331394646e-5, 4.0185152331394795e-5, 4.01851523313949e-5, 4.024852107599408e-5, 4.024852107599452e-5, 4.024852107599458e-5, 4.024852107599484e-5, 4.0248521075994885e-5, 4.024852107599515e-5, 4.0248521075995305e-5, 4.15676733672185e-5, 4.187648971210991e-5, 4.1876489712110656e-5, 4.187648971211098e-5, 4.1876489712112926e-5, 4.187648971211331e-5, 4.187648971211356e-5, 4.187648971211377e-5, 4.2088311687251286e-5, 4.2223740196828966e-5, 4.222374019682955e-5, 4.222374019682965e-5, 4.22237401968305e-5, 4.222374019683083e-5, 4.2223740196831514e-5, 4.222374019683158e-5, 4.241771904603557e-5, 4.261926938193054e-5, 4.263295326243286e-5, 4.2632953262433516e-5, 4.263295326243379e-5, 4.263295326243387e-5, 4.263295326243395e-5, 4.263295326243463e-5, 4.2632953262435115e-5, 4.282719424663694e-5, 4.297190980441474e-5, 4.297190980441528e-5, 4.2971909804415344e-5, 4.297190980441538e-5, 4.297190980441572e-5, 4.29719098044158e-5, 4.297190980441588e-5, 4.3129658383119595e-5, 4.312965838311986e-5, 4.312965838312065e-5, 4.312965838312069e-5, 4.3129658383120876e-5, 4.312965838312154e-5, 4.312965838312186e-5, 5.090570688757586e-5, 5.172983095103977e-5, 5.242212364049895e-5, 5.314803466055683e-5, 5.395948278468228e-5, 5.489277345132191e-5, 5.651166857267848e-5, 5.809942653662518e-5, 5.968182397081826e-5, 6.221965512376223e-5, 6.221965512376224e-5, 6.221965512376235e-5, 6.221965512376273e-5, 6.221965512376365e-5, 6.221965512376446e-5, 6.221965512376475e-5, 6.22198707739205e-5, 6.221999431952821e-5, 6.221999431952989e-5, 6.221999431953011e-5, 6.222188282475324e-5, 6.222535554516454e-5, 6.222535554516489e-5, 6.222535554516493e-5, 6.222535554516572e-5, 6.222535554516583e-5, 6.222535554516583e-5, 6.222535554516588e-5, 6.22275405321661e-5, 6.223147158657358e-5, 6.223147158657385e-5, 6.223147158657408e-5, 6.223147158657465e-5, 6.223147158657497e-5, 6.223147158657503e-5, 6.223147158657531e-5, 6.223424162512438e-5, 6.224042568118883e-5, 6.224042568118912e-5, 6.224042568118954e-5, 6.224042568118972e-5, 6.224042568119003e-5, 6.224042568119041e-5, 6.224042568119105e-5, 6.224247358247017e-5, 6.224725952932933e-5, 6.224725952933056e-5, 6.224725952933131e-5, 6.224725952933146e-5, 6.224725952933149e-5, 6.224725952933261e-5, 6.224725952933269e-5, 6.224895785032139e-5, 6.225249434854029e-5, 6.225249434854055e-5, 6.225249434854088e-5, 6.225249434854107e-5, 6.225249434854116e-5, 6.225249434854132e-5, 6.225249434854176e-5, 6.225409975918739e-5, 6.225739147363627e-5, 6.22573914736365e-5, 6.225739147363684e-5, 6.225739147363779e-5, 6.225739147363817e-5, 6.225739147363827e-5, 6.225739147363889e-5, 6.225865750777578e-5, 6.226092394213573e-5, 6.226092394213586e-5, 6.226092394213637e-5, 6.226092394213694e-5, 6.226092394213698e-5, 6.22609239421377e-5, 6.22609239421383e-5, 6.226231839190879e-5, 6.226564495584822e-5, 6.226564495584854e-5, 6.226564495584913e-5, 6.22656449558513e-5, 6.226564495585165e-5, 6.226564495585175e-5, 6.226564495585272e-5, 6.226635644979515e-5, 6.226873194415018e-5, 6.226873194415105e-5, 6.226873194415148e-5, 6.226873194415227e-5, 6.226873194415239e-5, 6.226873194415246e-5, 6.226873194415253e-5, 6.226895781765668e-5, 6.226935306591095e-5, 6.22693530659118e-5, 6.226935306591205e-5, 6.226935306591247e-5, 6.226935306591262e-5, 6.22693530659128e-5, 6.226935306591339e-5, 6.226971737867775e-5, 6.227032127845582e-5, 6.227032127845608e-5, 6.22703212784561e-5, 6.227032127845635e-5, 6.227032127845645e-5, 6.227032127845666e-5, 6.227032127845719e-5, 6.227059709343712e-5, 6.227114029912013e-5, 6.22711402991204e-5, 6.227114029912057e-5, 6.227114029912094e-5, 6.227114029912177e-5, 6.227114029912213e-5, 6.227114029912268e-5, 6.227125423996151e-5, 6.227140485015851e-5, 6.227140485015932e-5, 6.227140485016072e-5, 6.227140485016077e-5, 6.227140485016082e-5, 6.22714048501613e-5, 6.227140485016172e-5, 6.227161406352931e-5, 6.22719680235442e-5, 6.227196802354437e-5, 6.227196802354441e-5, 6.227196802354473e-5, 6.227196802354513e-5, 6.227196802354529e-5, 6.227196802354529e-5, 6.227198889216524e-5, 6.2272029007754e-5, 6.227202900775472e-5, 6.227202900775482e-5, 6.227202900775497e-5, 6.2272029007755e-5, 6.227202900775533e-5, 6.227202900775556e-5, 6.227202982404738e-5, 6.227203062628984e-5, 6.227203062628984e-5, 6.227203062629019e-5, 6.227203062629024e-5, 6.227203062629058e-5, 6.227203062629075e-5, 6.227203062629103e-5, 6.227217234531786e-5, 6.22722153503032e-5, 6.227221535030344e-5, 6.227221535030416e-5, 6.227221535030469e-5, 6.227221535030512e-5, 6.22722153503057e-5, 6.227221535030632e-5, 6.303655887391378e-5, 6.303655887391389e-5, 6.30365588739142e-5, 6.303664543559688e-5, 6.303683189032338e-5, 6.303683189032434e-5, 6.303683189032484e-5, 6.303683189032527e-5, 6.303683189032615e-5, 6.303683189032728e-5, 6.303683189032732e-5, 6.303958117495158e-5, 6.304503001272306e-5, 6.30450300127231e-5, 6.304503001272324e-5, 6.304503001272504e-5, 6.304503001272507e-5, 6.304503001272549e-5, 6.304503001272586e-5, 6.304768363048699e-5, 6.305424363908189e-5, 6.305424363908265e-5, 6.305424363908299e-5, 6.305424363908342e-5, 6.305424363908378e-5, 6.30542436390839e-5, 6.305424363908396e-5, 6.305646699507431e-5, 6.306108932819801e-5, 6.30610893281982e-5, 6.306108932819877e-5, 6.306108932819985e-5, 6.306108932820001e-5, 6.306108932820007e-5, 6.306108932820034e-5, 6.306340403891672e-5, 6.306742929187907e-5, 6.306742929187964e-5, 6.306742929187979e-5, 6.306742929188024e-5, 6.306742929188043e-5, 6.306742929188063e-5, 6.306742929188065e-5, 6.30702672301557e-5, 6.307673146666224e-5, 6.307673146666262e-5, 6.307673146666301e-5, 6.307673146666316e-5, 6.307673146666318e-5, 6.307673146666356e-5, 6.307673146666369e-5, 6.307844455467439e-5, 6.308265678400308e-5, 6.308265678400333e-5, 6.30826567840036e-5, 6.308265678400422e-5, 6.30826567840046e-5, 6.308265678400529e-5, 6.308265678400621e-5, 6.308398097864327e-5, 6.308699232004972e-5, 6.308699232005e-5, 6.308699232005031e-5, 6.308699232005043e-5, 6.308699232005052e-5, 6.308699232005086e-5, 6.308699232005158e-5, 6.308814378929565e-5, 6.309147715482963e-5, 6.30914771548306e-5, 6.309147715483066e-5, 6.309147715483066e-5, 6.309147715483208e-5, 6.309147715483253e-5, 6.309147715483387e-5, 6.309197977608073e-5, 6.309301434710831e-5, 6.309301434710834e-5, 6.309301434710857e-5, 6.309301434710918e-5, 6.309301434710921e-5, 6.309301434710997e-5, 6.309301434711e-5, 6.309359686898346e-5, 6.309491883304909e-5, 6.309491883304988e-5, 6.309491883305001e-5, 6.309491883305046e-5, 6.309491883305069e-5, 6.309491883305069e-5, 6.309491883305123e-5, 6.309522526609905e-5, 6.30958366876297e-5, 6.309583668762972e-5, 6.309583668762987e-5, 6.309583668763e-5, 6.309583668763022e-5, 6.309583668763055e-5, 6.309583668763082e-5, 6.309609849961861e-5, 6.309666471079553e-5, 6.309666471079557e-5, 6.309666471079565e-5, 6.30966647107958e-5, 6.309666471079604e-5, 6.309666471079606e-5, 6.309666471079615e-5, 6.309671665055726e-5, 6.30967740946074e-5, 6.309677409460742e-5, 6.309677409460773e-5, 6.309677409460806e-5, 6.309677409460843e-5, 6.309677409460858e-5, 6.309677409460921e-5, 6.309699550532652e-5, 6.309714265773036e-5, 6.309714265773042e-5, 6.30971426577309e-5, 6.309714265773099e-5, 6.30971426577311e-5, 6.309714265773164e-5, 6.309714265773177e-5, 6.309752110678447e-5, 6.309791068158506e-5, 6.309791068158533e-5, 6.309791068158615e-5, 6.30979106815871e-5, 6.30979106815872e-5, 6.309791068158728e-5, 6.309791068158735e-5, 6.309792760301497e-5, 6.309795147596362e-5, 6.30979514759638e-5, 6.309795147596394e-5, 6.309795147596466e-5, 6.309795147596485e-5, 6.309795147596488e-5, 6.309795147596556e-5, 6.309796408876119e-5, 6.309797389283951e-5, 6.309797389284068e-5, 6.309797389284068e-5, 6.3097973892841e-5, 6.30979738928411e-5, 6.309797389284149e-5, 6.309797389284152e-5, 6.371793422004381e-5, 6.371793422004464e-5, 6.371793422004498e-5, 6.371851520527216e-5, 6.372301542709366e-5, 6.372301542709421e-5, 6.372301542709435e-5, 6.372301542709452e-5, 6.372301542709477e-5, 6.372301542709493e-5, 6.372301542709551e-5, 6.372317049292331e-5, 6.37233416688783e-5, 6.372334166887878e-5, 6.372334166887954e-5, 6.372334166887963e-5, 6.372334166887968e-5, 6.372334166888009e-5, 6.372334166888019e-5, 6.372532280523968e-5, 6.372697084500527e-5, 6.372697084500558e-5, 6.372697084500573e-5, 6.372697084500594e-5, 6.372697084500627e-5, 6.372697084500635e-5, 6.372697084500762e-5, 6.373069482911646e-5, 6.373607484626783e-5, 6.373607484626872e-5, 6.37360748462688e-5, 6.373607484626904e-5, 6.373607484626912e-5, 6.373607484626935e-5, 6.373607484626952e-5, 6.373870207336181e-5, 6.374436372151531e-5, 6.374436372151575e-5, 6.374436372151579e-5, 6.374436372151632e-5, 6.374436372151652e-5, 6.37443637215167e-5, 6.374436372151682e-5, 6.374638564784681e-5, 6.375137060422303e-5, 6.375137060422363e-5, 6.375137060422371e-5, 6.375137060422388e-5, 6.3751370604224e-5, 6.375137060422497e-5, 6.375137060422542e-5, 6.37527411257593e-5, 6.375555049346323e-5, 6.375555049346363e-5, 6.375555049346391e-5, 6.375555049346402e-5, 6.375555049346433e-5, 6.375555049346451e-5, 6.37555504934649e-5, 6.375698549777649e-5, 6.376011426581197e-5, 6.376011426581277e-5, 6.376011426581335e-5, 6.376011426581353e-5, 6.37601142658137e-5, 6.37601142658141e-5, 6.376011426581496e-5, 6.376114513874831e-5, 6.37636472793382e-5, 6.376364727933834e-5, 6.376364727933897e-5, 6.376364727933906e-5, 6.376364727933914e-5, 6.376364727933945e-5, 6.376364727934021e-5, 6.376432886426974e-5, 6.376600565641518e-5, 6.376600565641525e-5, 6.37660056564159e-5, 6.37660056564162e-5, 6.376600565641629e-5, 6.376600565641689e-5, 6.376600565641705e-5, 6.376643597919906e-5, 6.37672697980722e-5, 6.376726979807323e-5, 6.376726979807356e-5, 6.376726979807364e-5, 6.376726979807384e-5, 6.376726979807405e-5, 6.376726979807459e-5, 6.37677100216067e-5, 6.376870718045164e-5, 6.376870718045221e-5, 6.376870718045234e-5, 6.37687071804524e-5, 6.37687071804529e-5, 6.376870718045291e-5, 6.376870718045343e-5, 6.376890902565351e-5, 6.376946776274346e-5, 6.376946776274395e-5, 6.376946776274398e-5, 6.376946776274403e-5, 6.376946776274459e-5, 6.376946776274488e-5, 6.376946776274516e-5, 6.376947299570508e-5, 6.376947849624445e-5, 6.37694784962445e-5, 6.376947849624455e-5, 6.376947849624497e-5, 6.376947849624517e-5, 6.376947849624534e-5, 6.376947849624617e-5, 6.376965478358382e-5, 6.376976615566848e-5, 6.37697661556687e-5, 6.376976615566932e-5, 6.376976615566978e-5, 6.376976615567e-5, 6.376976615567096e-5, 6.376976615567161e-5, 6.376994867230023e-5, 6.377006812160095e-5, 6.377006812160114e-5, 6.377006812160136e-5, 6.37700681216014e-5, 6.377006812160141e-5, 6.377006812160237e-5, 6.377006812160331e-5, 6.377033074202559e-5, 6.377051436671938e-5, 6.377051436671984e-5, 6.377051436672038e-5, 6.377051436672041e-5, 6.37705143667206e-5, 6.377051436672098e-5, 6.377051436672126e-5, 6.377054015745561e-5, 6.377056247617858e-5, 6.37705624761791e-5, 6.377056247617932e-5, 6.377056247617936e-5, 6.377056247617993e-5, 6.377056247618009e-5, 6.377056247618016e-5, 6.440127297830384e-5, 6.440127297830387e-5, 6.440127297830434e-5, 6.440127297830475e-5, 6.440127297830585e-5, 6.440127297830631e-5, 6.44012729783065e-5, 6.440181255765825e-5, 6.440257776953191e-5, 6.440257776953203e-5, 6.44025777695321e-5, 6.440257776953252e-5, 6.440257776953265e-5, 6.440257776953274e-5, 6.440257776953443e-5, 6.440444221866939e-5, 6.440574925041261e-5, 6.440574925041328e-5, 6.440574925041334e-5, 6.440755854477184e-5, 6.441093382346926e-5, 6.44109338234693e-5, 6.44109338234704e-5, 6.441093382347115e-5, 6.441093382347223e-5, 6.441093382347234e-5, 6.441093382347272e-5, 6.441465235516825e-5, 6.44223141857833e-5, 6.442231418578481e-5, 6.442231418578565e-5, 6.442231418578583e-5, 6.442231418578636e-5, 6.442231418578717e-5, 6.442231418578725e-5, 6.442471803807288e-5, 6.443033421541968e-5, 6.44303342154203e-5, 6.443033421542033e-5, 6.443033421542165e-5, 6.443033421542223e-5, 6.443033421542243e-5, 6.443033421542247e-5, 6.443243800813936e-5, 6.443756438977928e-5, 6.443756438977957e-5, 6.44375643897796e-5, 6.44375643897798e-5, 6.443756438978014e-5, 6.443756438978029e-5, 6.443756438978082e-5, 6.443908896815877e-5, 6.44422357068792e-5, 6.44422357068804e-5, 6.44422357068804e-5, 6.444223570688056e-5, 6.444223570688102e-5, 6.444223570688105e-5, 6.44422357068811e-5, 6.444384505860624e-5, 6.444801217792252e-5, 6.444801217792279e-5, 6.444801217792293e-5, 6.444801217792397e-5, 6.444801217792465e-5, 6.444801217792477e-5, 6.444801217792478e-5, 6.44488450801152e-5, 6.445067998660448e-5, 6.445067998660464e-5, 6.445067998660489e-5, 6.445067998660502e-5, 6.445067998660605e-5, 6.445067998660621e-5, 6.445067998660791e-5, 6.445152003463505e-5, 6.445392872257354e-5, 6.445392872257369e-5, 6.445392872257401e-5, 6.445392872257401e-5, 6.445392872257423e-5, 6.445392872257464e-5, 6.445392872257553e-5, 6.445426134934397e-5, 6.445503768445836e-5, 6.445503768445856e-5, 6.44550376844589e-5, 6.445503768445905e-5, 6.445503768445924e-5, 6.445503768445977e-5, 6.445503768445982e-5, 6.445531230807735e-5, 6.44558863621684e-5, 6.4455886362169e-5, 6.445588636216917e-5, 6.445588636216974e-5, 6.445588636217007e-5, 6.44558863621702e-5, 6.445588636217236e-5, 6.44559908087496e-5, 6.445610525556999e-5, 6.44561052555706e-5, 6.445610525557099e-5, 6.445610525557104e-5, 6.445610525557121e-5, 6.445610525557157e-5, 6.445610525557172e-5, 6.445655694769678e-5, 6.445720829890954e-5, 6.445720829890998e-5, 6.445720829891029e-5, 6.445720829891049e-5, 6.445720829891133e-5, 6.445720829891168e-5, 6.445720829891362e-5, 6.445721885570396e-5, 6.445723018569701e-5, 6.445723018569716e-5, 6.445723018569742e-5, 6.445723018569767e-5, 6.445723018569786e-5, 6.445723018569816e-5, 6.445723018569998e-5, 6.445732240504549e-5, 6.445737500189114e-5, 6.445737500189151e-5, 6.445737500189267e-5, 6.44573750018929e-5, 6.445737500189303e-5, 6.44573750018936e-5, 6.445737500189551e-5, 6.445762972452846e-5, 6.44577244056563e-5, 6.44577244056564e-5, 6.445772440565651e-5, 6.445772440565721e-5, 6.445772440565751e-5, 6.445772440565847e-5, 6.44577244056593e-5, 6.44585481624758e-5, 6.445873272646316e-5, 6.445873272646334e-5, 6.445873272646339e-5, 6.4458732726464e-5, 6.445873272646423e-5, 6.445873272646433e-5, 6.445873272646438e-5, 6.520550404837889e-5, 6.520550404837912e-5, 6.520550404837926e-5, 6.520641520434074e-5, 6.521211941775612e-5, 6.521211941775621e-5, 6.521211941775646e-5, 6.521211941775657e-5, 6.521211941775707e-5, 6.521211941775797e-5, 6.521211941775806e-5, 6.521413275937628e-5, 6.522011623498754e-5, 6.522011623498854e-5, 6.522011623498862e-5, 6.522011623498903e-5, 6.522011623499e-5, 6.522011623499117e-5, 6.522011623499142e-5, 6.52220277169498e-5, 6.522599063042564e-5, 6.52259906304262e-5, 6.52259906304263e-5, 6.522599063042754e-5, 6.522599063042896e-5, 6.52259906304293e-5, 6.522599063042998e-5, 6.522843587664453e-5, 6.523311760549255e-5, 6.523311760549329e-5, 6.52331176054934e-5, 6.523311760549357e-5, 6.523311760549503e-5, 6.523311760549536e-5, 6.523311760549551e-5, 6.523570314556886e-5, 6.524135877568418e-5, 6.524135877568467e-5, 6.52413587756852e-5, 6.524135877568538e-5, 6.52413587756854e-5, 6.524135877568591e-5, 6.524135877568605e-5, 6.524342777251993e-5, 6.524860188228701e-5, 6.524860188228725e-5, 6.524860188228816e-5, 6.524860188228911e-5, 6.524860188229002e-5, 6.524860188229037e-5, 6.524860188229122e-5, 6.525006230061477e-5, 6.525339167675779e-5, 6.525339167675827e-5, 6.525339167675889e-5, 6.525339167675951e-5, 6.525339167676103e-5, 6.525339167676138e-5, 6.525339167676171e-5, 6.525474072912147e-5, 6.525860663827404e-5, 6.525860663827472e-5, 6.525860663827518e-5, 6.525860663827555e-5, 6.525860663827605e-5, 6.525860663827648e-5, 6.525860663827675e-5, 6.52591626814788e-5, 6.526013019229868e-5, 6.526013019229912e-5, 6.526013019229922e-5, 6.526013019229958e-5, 6.526013019230004e-5, 6.526013019230009e-5, 6.526013019230177e-5, 6.526099062056582e-5, 6.52624408151195e-5, 6.526244081511984e-5, 6.526244081512022e-5, 6.52624408151204e-5, 6.526244081512041e-5, 6.526244081512153e-5, 6.5262440815122e-5, 6.526314345272595e-5, 6.526475194895277e-5, 6.526475194895315e-5, 6.52647519489546e-5, 6.526475194895485e-5, 6.526475194895578e-5, 6.526475194895683e-5, 6.526475194895848e-5, 6.526505541949284e-5, 6.526562447485161e-5, 6.52656244748525e-5, 6.526562447485252e-5, 6.526562447485253e-5, 6.526562447485413e-5, 6.52656244748546e-5, 6.526562447485573e-5, 6.526593714028592e-5, 6.526663547419664e-5, 6.526663547419717e-5, 6.526663547419717e-5, 6.526663547419736e-5, 6.526663547419771e-5, 6.526663547419836e-5, 6.526663547420029e-5, 6.526669269474585e-5, 6.526676761140412e-5, 6.526676761140456e-5, 6.526676761140472e-5, 6.526676761140483e-5, 6.526676761140504e-5, 6.526676761140517e-5, 6.526676761140693e-5, 6.526687469665222e-5, 6.526695238206563e-5, 6.526695238206634e-5, 6.526695238206664e-5, 6.526695238206723e-5, 6.526695238206783e-5, 6.52669523820681e-5, 6.526695238206875e-5, 6.526737128195117e-5, 6.526766076694955e-5, 6.526766076694968e-5, 6.526766076695036e-5, 6.526766076695057e-5, 6.526766076695063e-5, 6.526766076695142e-5, 6.526766076695352e-5, 6.526768603878903e-5, 6.526770896500671e-5, 6.526770896500774e-5, 6.526770896500786e-5, 6.526770896500805e-5, 6.526770896500814e-5, 6.52677089650083e-5, 6.526770896500916e-5, 6.526831700073359e-5, 6.526846765466574e-5, 6.526846765466578e-5, 6.526846765466628e-5, 6.526846765466632e-5, 6.526846765466637e-5, 6.526846765466646e-5, 6.526846765466785e-5, 6.618608327151317e-5, 6.618608327151346e-5, 6.618608327151374e-5, 6.618608327151436e-5, 6.618608327151455e-5, 6.618608327151542e-5, 6.618608327151572e-5, 6.618626004704281e-5, 6.618635771085123e-5, 6.618635771085134e-5, 6.618635771085218e-5, 6.618844874274854e-5, 6.619178886001254e-5, 6.619178886001314e-5, 6.619178886001329e-5, 6.619178886001345e-5, 6.619178886001372e-5, 6.61917888600138e-5, 6.619178886001426e-5, 6.619477653670146e-5, 6.620129606564784e-5, 6.620129606564798e-5, 6.620129606564806e-5, 6.620129606564815e-5, 6.620129606564831e-5, 6.62012960656487e-5, 6.620129606564872e-5, 6.620341017930306e-5, 6.620692119780572e-5, 6.6206921197806e-5, 6.620692119780656e-5, 6.620692119780664e-5, 6.620692119780675e-5, 6.620692119780683e-5, 6.620692119780725e-5, 6.62104163320957e-5, 6.621861237443082e-5, 6.621861237443105e-5, 6.621861237443166e-5, 6.621861237443286e-5, 6.621861237443341e-5, 6.621861237443365e-5, 6.621861237443417e-5, 6.622049977965009e-5, 6.622498969170621e-5, 6.622498969170686e-5, 6.622498969170714e-5, 6.622498969170747e-5, 6.622498969170785e-5, 6.62249896917079e-5, 6.622498969170802e-5, 6.622672631229373e-5, 6.623086056994128e-5, 6.623086056994177e-5, 6.623086056994222e-5, 6.623086056994309e-5, 6.623086056994344e-5, 6.623086056994394e-5, 6.623086056994398e-5, 6.62322217128721e-5, 6.623572644685297e-5, 6.623572644685359e-5, 6.623572644685384e-5, 6.62357264468543e-5, 6.623572644685457e-5, 6.6235726446855e-5, 6.623572644685713e-5, 6.623667921818564e-5, 6.623944342282435e-5, 6.623944342282447e-5, 6.623944342282477e-5, 6.623944342282485e-5, 6.623944342282514e-5, 6.623944342282541e-5, 6.623944342282624e-5, 6.623999463905936e-5, 6.624154323155392e-5, 6.624154323155408e-5, 6.624154323155505e-5, 6.62415432315551e-5, 6.624154323155532e-5, 6.624154323155596e-5, 6.624154323155699e-5, 6.624187126179298e-5, 6.624254006607275e-5, 6.624254006607299e-5, 6.624254006607299e-5, 6.624254006607313e-5, 6.62425400660744e-5, 6.624254006607538e-5, 6.62425400660763e-5, 6.624287165505928e-5, 6.624353094430398e-5, 6.62435309443048e-5, 6.624353094430501e-5, 6.624353094430519e-5, 6.624353094430681e-5, 6.624353094430767e-5, 6.6243530944308e-5, 6.624374218356945e-5, 6.624422130610734e-5, 6.624422130610815e-5, 6.624422130610816e-5, 6.624422130610907e-5, 6.624422130610961e-5, 6.624422130610961e-5, 6.624422130610971e-5, 6.624426562913143e-5, 6.624431873435846e-5, 6.624431873435863e-5, 6.624431873435907e-5, 6.62443187343594e-5, 6.624431873435983e-5, 6.624431873436007e-5, 6.6244318734362e-5, 6.624442104998735e-5, 6.624448919017105e-5, 6.624448919017129e-5, 6.62444891901714e-5, 6.624448919017209e-5, 6.624448919017212e-5, 6.624448919017246e-5, 6.624448919017289e-5, 6.624483217361201e-5, 6.624498138146291e-5, 6.624498138146311e-5, 6.62449813814634e-5, 6.624498138146363e-5, 6.624498138146384e-5, 6.624498138146464e-5, 6.624498138146515e-5, 6.624547570831401e-5, 6.624573797434563e-5, 6.624573797434635e-5, 6.624573797434639e-5, 6.62457379743466e-5, 6.624573797434709e-5, 6.624573797434767e-5, 6.624573797434861e-5, 6.624580154097973e-5, 6.624584934882879e-5, 6.624584934882916e-5, 6.62458493488296e-5, 6.62458493488296e-5, 6.624584934882974e-5, 6.624584934883023e-5, 6.624584934883172e-5, 6.788274822355589e-5, 6.78827482235563e-5, 6.788274822355631e-5, 6.788274822355683e-5, 6.7882748223557e-5, 6.788274822355759e-5, 6.788274822355803e-5, 6.788347348085442e-5, 6.788423574460884e-5, 6.788423574460891e-5, 6.788423574460953e-5, 6.788499208204587e-5, 6.788640271243041e-5, 6.788640271243083e-5, 6.78864027124312e-5, 6.78864027124315e-5, 6.788640271243152e-5, 6.78864027124318e-5, 6.788640271243213e-5, 6.788969177801452e-5, 6.789547808304269e-5, 6.789547808304273e-5, 6.78954780830429e-5, 6.789547808304292e-5, 6.789547808304346e-5, 6.789547808304353e-5, 6.789547808304371e-5, 6.789799960058514e-5, 6.79041625612901e-5, 6.790416256129016e-5, 6.790416256129032e-5, 6.790416256129063e-5, 6.790416256129126e-5, 6.790416256129142e-5, 6.790416256129245e-5, 6.790537564783141e-5, 6.790697555647234e-5, 6.790697555647319e-5, 6.790697555647329e-5, 6.790697555647365e-5, 6.790697555647373e-5, 6.790697555647386e-5, 6.790697555647437e-5, 6.791038581496376e-5, 6.791525164125878e-5, 6.791525164125922e-5, 6.791525164125952e-5, 6.791525164125956e-5, 6.791525164125969e-5, 6.791525164126065e-5, 6.791525164126082e-5, 6.791792045765445e-5, 6.792500311837997e-5, 6.792500311838024e-5, 6.792500311838056e-5, 6.792500311838113e-5, 6.792500311838115e-5, 6.792500311838155e-5, 6.792500311838173e-5, 6.792615971830455e-5, 6.792908515332285e-5, 6.792908515332338e-5, 6.792908515332342e-5, 6.792908515332412e-5, 6.792908515332434e-5, 6.792908515332442e-5, 6.792908515332453e-5, 6.793005518378981e-5, 6.79323985616989e-5, 6.79323985616994e-5, 6.793239856169976e-5, 6.793239856170007e-5, 6.793239856170047e-5, 6.793239856170088e-5, 6.793239856170094e-5, 6.793303075577981e-5, 6.793417996330003e-5, 6.79341799633004e-5, 6.793417996330079e-5, 6.793417996330092e-5, 6.793417996330103e-5, 6.793417996330106e-5, 6.793417996330125e-5, 6.793488351702838e-5, 6.793607708863105e-5, 6.793607708863124e-5, 6.793607708863141e-5, 6.793607708863171e-5, 6.79360770886324e-5, 6.793607708863275e-5, 6.793607708863285e-5, 6.793667561357539e-5, 6.793828923945205e-5, 6.793828923945235e-5, 6.793828923945246e-5, 6.793828923945343e-5, 6.793828923945384e-5, 6.793828923945397e-5, 6.793828923945457e-5, 6.793832423637447e-5, 6.793836281856916e-5, 6.793836281856999e-5, 6.793836281857e-5, 6.793836281857012e-5, 6.793836281857031e-5, 6.793836281857077e-5, 6.793836281857102e-5, 6.793863137995278e-5, 6.793883369057362e-5, 6.793883369057424e-5, 6.793883369057439e-5, 6.793883369057449e-5, 6.79388336905751e-5, 6.793883369057563e-5, 6.79388336905759e-5, 6.793917475241339e-5, 6.79395162025338e-5, 6.793951620253447e-5, 6.793951620253467e-5, 6.793951620253506e-5, 6.793951620253508e-5, 6.793951620253533e-5, 6.793951620253535e-5, 6.793960151628912e-5, 6.793967700932417e-5, 6.793967700932469e-5, 6.793967700932485e-5, 6.793967700932536e-5, 6.793967700932607e-5, 6.793967700932612e-5, 6.793967700932654e-5, 6.7940109242269e-5, 6.794029411368008e-5, 6.794029411368041e-5, 6.794029411368045e-5, 6.79402941136806e-5, 6.79402941136806e-5, 6.794029411368073e-5, 6.79402941136824e-5, 6.794060329913464e-5, 6.794073393914434e-5, 6.794073393914458e-5, 6.794073393914477e-5, 6.794073393914499e-5, 6.794073393914544e-5, 6.79407339391455e-5, 6.794073393914628e-5, 6.954391815715242e-5, 6.954391815715276e-5, 6.954391815715284e-5, 6.954427539149984e-5, 6.954547753036911e-5, 6.95454775303695e-5, 6.954547753036962e-5, 6.954547753037058e-5, 6.9545477530371e-5, 6.954547753037104e-5, 6.954547753037155e-5, 6.95469113130552e-5, 6.954909955300186e-5, 6.954909955300222e-5, 6.954909955300247e-5, 6.954909955300288e-5, 6.954909955300304e-5, 6.954909955300331e-5, 6.954909955300338e-5, 6.955181701240543e-5, 6.95561911711372e-5, 6.955619117113817e-5, 6.955619117113869e-5, 6.955619117113901e-5, 6.955619117113927e-5, 6.955619117113948e-5, 6.955619117113975e-5, 6.955910439642455e-5, 6.956448822314121e-5, 6.956448822314129e-5, 6.956448822314156e-5, 6.956448822314175e-5, 6.956448822314228e-5, 6.956448822314242e-5, 6.956448822314251e-5, 6.95672897645288e-5, 6.957346284496718e-5, 6.957346284496788e-5, 6.957346284496805e-5, 6.957346284496807e-5, 6.957346284496845e-5, 6.957346284496866e-5, 6.957346284496875e-5, 6.95754795855106e-5, 6.957945271470029e-5, 6.957945271470118e-5, 6.957945271470171e-5, 6.957945271470187e-5, 6.957945271470313e-5, 6.957945271470321e-5, 6.957945271470331e-5, 6.958165524044138e-5, 6.95873233940532e-5, 6.958732339405436e-5, 6.958732339405449e-5, 6.958732339405521e-5, 6.958732339405526e-5, 6.958732339405616e-5, 6.958732339405674e-5, 6.958839323972996e-5, 6.959063565042711e-5, 6.959063565042726e-5, 6.95906356504278e-5, 6.959063565042828e-5, 6.959063565042901e-5, 6.959063565042919e-5, 6.95906356504298e-5, 6.959166586750525e-5, 6.959330826880227e-5, 6.959330826880257e-5, 6.959330826880267e-5, 6.959330826880345e-5, 6.959330826880459e-5, 6.959330826880588e-5, 6.959330826880611e-5, 6.959461619191643e-5, 6.959733197300474e-5, 6.959733197300535e-5, 6.959733197300539e-5, 6.959733197300569e-5, 6.959733197300599e-5, 6.959733197300646e-5, 6.959733197300649e-5, 6.959799032708773e-5, 6.960009587730936e-5, 6.96000958773098e-5, 6.96000958773099e-5, 6.96000958773101e-5, 6.960009587731032e-5, 6.960009587731134e-5, 6.960009587731136e-5, 6.960023633718645e-5, 6.960053786498154e-5, 6.960053786498238e-5, 6.960053786498253e-5, 6.960053786498265e-5, 6.960053786498279e-5, 6.9600537864983e-5, 6.9600537864983e-5, 6.960056252737925e-5, 6.96005871247518e-5, 6.96005871247521e-5, 6.960058712475236e-5, 6.960058712475247e-5, 6.960058712475264e-5, 6.960058712475277e-5, 6.96005871247531e-5, 6.960132292513122e-5, 6.960194327571482e-5, 6.96019432757151e-5, 6.960194327571512e-5, 6.960194327571524e-5, 6.960194327571524e-5, 6.960194327571562e-5, 6.960194327571576e-5, 6.960210105148283e-5, 6.960229351113051e-5, 6.960229351113104e-5, 6.96022935111313e-5, 6.960229351113132e-5, 6.960229351113186e-5, 6.960229351113196e-5, 6.960229351113285e-5, 6.960241354446907e-5, 6.960250281928201e-5, 6.960250281928202e-5, 6.960250281928224e-5, 6.960250281928269e-5, 6.960250281928303e-5, 6.960250281928332e-5, 6.960250281928481e-5, 6.960317647306594e-5, 6.960347974086516e-5, 6.960347974086578e-5, 6.96034797408661e-5, 6.960347974086646e-5, 6.960347974086658e-5, 6.960347974086677e-5, 6.96034797408677e-5, 6.960360521852263e-5, 6.960368906101726e-5, 6.960368906101776e-5, 6.960368906101829e-5, 6.96036890610184e-5, 6.960368906101863e-5, 6.9603689061019e-5, 6.960368906101933e-5, 7.115350797904934e-5, 7.115350797904988e-5, 7.115350797905058e-5, 7.115469014655929e-5, 7.11644501806919e-5, 7.116445018069257e-5, 7.116445018069296e-5, 7.116445018069318e-5, 7.116445018069342e-5, 7.116445018069364e-5, 7.116445018069448e-5, 7.116615137353488e-5, 7.116989785109827e-5, 7.116989785109841e-5, 7.116989785109935e-5, 7.116989785109945e-5, 7.116989785109951e-5, 7.11698978511001e-5, 7.116989785110028e-5, 7.117242119641075e-5, 7.117692463301148e-5, 7.117692463301215e-5, 7.117692463301237e-5, 7.11769246330127e-5, 7.117692463301298e-5, 7.117692463301308e-5, 7.117692463301326e-5, 7.118022638182415e-5, 7.118666940941128e-5, 7.11866694094119e-5, 7.1186669409412e-5, 7.118666940941243e-5, 7.11866694094125e-5, 7.118666940941252e-5, 7.118666940941271e-5, 7.118970399295032e-5, 7.119661297886221e-5, 7.119661297886232e-5, 7.119661297886244e-5, 7.119661297886308e-5, 7.119661297886329e-5, 7.119661297886339e-5, 7.119661297886435e-5, 7.119909040590498e-5, 7.120587542227924e-5, 7.120587542228018e-5, 7.120587542228022e-5, 7.120587542228145e-5, 7.12058754222817e-5, 7.120587542228221e-5, 7.120587542228239e-5, 7.120750190337499e-5, 7.121200985857485e-5, 7.121200985857515e-5, 7.121200985857534e-5, 7.121200985857561e-5, 7.12120098585759e-5, 7.121200985857599e-5, 7.12120098585771e-5, 7.121310602372139e-5, 7.121537488864721e-5, 7.121537488864799e-5, 7.121537488864811e-5, 7.12153748886488e-5, 7.121537488864912e-5, 7.121537488864983e-5, 7.121537488865025e-5, 7.121661983489884e-5, 7.12198120950558e-5, 7.121981209505654e-5, 7.121981209505681e-5, 7.121981209505716e-5, 7.12198120950575e-5, 7.121981209505782e-5, 7.121981209505839e-5, 7.122046801963744e-5, 7.122236934336868e-5, 7.122236934337044e-5, 7.12223693433705e-5, 7.122236934337095e-5, 7.122236934337175e-5, 7.122236934337308e-5, 7.12223693433732e-5, 7.122265168741347e-5, 7.122310028075669e-5, 7.122310028075705e-5, 7.122310028075726e-5, 7.122310028075778e-5, 7.122310028075787e-5, 7.122310028075799e-5, 7.122310028075827e-5, 7.122357998629196e-5, 7.122426456890318e-5, 7.122426456890336e-5, 7.122426456890336e-5, 7.122426456890356e-5, 7.122426456890366e-5, 7.122426456890373e-5, 7.12242645689055e-5, 7.122465620541785e-5, 7.122524406856513e-5, 7.122524406856589e-5, 7.122524406856593e-5, 7.122524406856615e-5, 7.122524406856649e-5, 7.122524406856666e-5, 7.122524406856745e-5, 7.122555727529031e-5, 7.122628853824276e-5, 7.122628853824332e-5, 7.122628853824375e-5, 7.122628853824386e-5, 7.12262885382444e-5, 7.122628853824448e-5, 7.122628853824465e-5, 7.122630804985692e-5, 7.122633151359906e-5, 7.122633151359968e-5, 7.122633151359988e-5, 7.122633151359988e-5, 7.122633151360002e-5, 7.122633151360033e-5, 7.122633151360219e-5, 7.122641330905987e-5, 7.122647661341128e-5, 7.122647661341137e-5, 7.122647661341167e-5, 7.122647661341174e-5, 7.122647661341209e-5, 7.12264766134121e-5, 7.122647661341812e-5, 7.122655226520305e-5, 7.122659919535342e-5, 7.122659919535388e-5, 7.122659919535468e-5, 7.122659919535491e-5, 7.122659919535501e-5, 7.122659919535521e-5, 7.122659919535545e-5, 7.122684442262258e-5, 7.122692342214743e-5, 7.122692342214795e-5, 7.122692342214859e-5, 7.122692342214875e-5, 7.1226923422149e-5, 7.122692342214918e-5, 7.12269234221493e-5]

    λ *= 1e-12 / hbar

    f = Figure()
    ax = Axis(f[1,1])
    scatter!(ax, λ)
    display(f)
end

function main()
    n_ε = 12
    n_θ = 38

    Uee = 0.0
    Vimp = 0.0

    fit_file = joinpath(data_dir, "Uee_Vimp_fit.info")
    if isfile(fit_file)
        open(fit_file,"r") do f
            for line in eachline(f)
                key, value = split(line, ':')
                key == "Uee" && (Uee = parse(Float64, value))
                key == "Vimp" && (Vimp = parse(Float64, value))
            end
        end
    end
    Uee == 0.0 && Vimp == 0.0 && return nothing

    temps = 2.0:1.0:14.0

    # LudwigIO.get_property("Rh", data_dir, material, 12.0, n_ε, n_θ, Uee, Vimp; n_ε = n_ε, n_θ = n_θ, n_bands = 3)

    # plot_optical_conductivity(n_ε, n_θ, 8.0)
    # optical_conductivity(n_ε, n_θ, Uee, 4.7 * Vimp, 8.0)
    # for T in temps
    # end

    # fit_ρ(n_ε, n_θ, temps, model_1, Uee, Vimp)
    # plot_η(n_ε, n_θ)
    plot_ρ_strain(n_ε, n_θ)
    # plot_lifetimes(n_ε, n_θ; addendum = "ee_only")
    # plot_lifetime_matthiesen(n_ε, n_θ, "η")
    # scaled_impurity_model(n_ε, n_θ)
    # scaled_impurity_model(n_ε, n_θ; addendum = "simple_impurity")

    # ϵ = -0.02
    # LudwigIO.write_property_to_file("ρ", material*"_uniaxial_strain_ϵ_$(ϵ)", data_dir, n_ε, n_θ, Uee, Vimp, temps; addendum = "ϵ_$(ϵ)", imp_stem = impurity_stem)

    # ϵ = -0.03
    # LudwigIO.write_property_to_file("ρ", material*"_uniaxial_strain_ϵ_$(ϵ)", data_dir, n_ε, n_θ, Uee, Vimp, temps; addendum = "ϵ_$(ϵ)", imp_stem = impurity_stem)

    # LudwigIO.write_property_to_file("ηB1g", material, data_dir, n_ε, n_θ, Uee, Vimp, temps; addendum = "simple_impurity", imp_stem = impurity_stem)
    # LudwigIO.write_property_to_file("ηB2g", material, data_dir, n_ε, n_θ, Uee, Vimp, temps; addendum = "simple_impurity", imp_stem = impurity_stem)
end 

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))
data_dir = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "model_2")
plot_dir = joinpath(@__DIR__, "..", "plots", "Sr2RuO4")
exp_dir  = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "experiment")
impurity_stem = ""

main()