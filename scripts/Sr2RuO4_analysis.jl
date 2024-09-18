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

function get_σ(L, k, v, E, dV, T)
    σ = Ludwig.conductivity(L, v, E, dV, T)
    return σ / c
end

function get_σxx(L, k, v, E, dV, T)
    σxx = Ludwig.longitudinal_conductivity(L, first.(v), E, dV, T)
    return σxx / c
end

function get_ρ(L, k, v, E, dV, T)
    σxx = Ludwig.longitudinal_conductivity(L, first.(v), E, dV, T)
    return c / σxx
end

function get_ηB1g(L, k, v, E, dV, T)
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

function get_ηB2g(L, k, v, E, dV, T)
    ℓ = length(k)
    Dxy = zeros(Float64, ℓ)
    for i in eachindex(k)
        μ = (i-1)÷(ℓ÷3) + 1 # Band index
        Dxy[i] = dxy_μ(k[i], μ)
    end

    return Ludwig.ηB2g(L, E, dV, Dxy, T) / (a^2 * c)
end

function get_τeff_σ(L, k, v, E, dV, T)
    return Ludwig.σ_lifetime(L, v, E, dV, T)
end

function get_τeff_η(L, k, v, E, dV, T)
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
            ρ[i] = LudwigIO.get_property("ρ", data_dir, material, t[i], n_ε, n_θ, p[1], p[2]) * 1e8
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

    xlims!(ax, 0.0, 14.0)
    ylims!(ax, 0.0, 1.0)
    domain = 0.0:0.1:14.0

    scatter!(ax, lupien_data[:, 1], lupien_data[:, 2], color = :black)
    scatter!(ax, t, ρ, color = :red)
    lines!(ax, domain, model(domain, fit.param), color = :red)
    lines!(ax, domain, l_model(domain, lfit.param))
    display(f)

    outfile = joinpath(plot_dir, "23 August 2024", "ρ_with_fit.png")
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
            ylabel = L"\eta / \eta_{B1g}^{(0)}",
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
    
    outfile = joinpath(@__DIR__, "..", "plots", "ηB1g_vs_ηB2g.png")
    # save(outfile, f)

    
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
    outfile = joinpath(@__DIR__, "..", "plots", "ηB1g.png")
    # save(outfile, f)

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
    
    ρ_fit = curve_fit(ρ_model, T, ρ, [1e-2, 0.1, 2.0])
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

    save(joinpath(plot_dir, "23 August 2024", "ρ_and_η_fit.png"), f)
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
    save(joinpath(plot_dir, "τ_eff.png"), f)

end

function scaled_impurity_model(n_ε, n_θ)
    t, ρ, _ = LudwigIO.read_property_from_file(joinpath(data_dir, "ρ_impurity_only_$(n_ε)x$(n_θ).dat"))
    ρ *= 1e8

    lupien_file = joinpath(exp_dir, "rhovT_Lupien_digitized.dat")
    lupien_data = readdlm(lupien_file)
    lfit = curve_fit(model_1, lupien_data[10:40, 1], lupien_data[10:40, 2], [0.0, 0.0001, 2.0], lower = [0.0, 0.0, 0.0])

    scale_fit = curve_fit(model_0, t, model_1(t, lfit.param) ./ ρ,  [0.65, 0.1])
    
    τ = model_1(t, lfit.param) ./ ρ
    # τ = model_0(t, scale_fit.param) # Lifetime scaling that goes as T^2

    domain = 0.0:0.1:14.0
    f = Figure()
    ax = Axis(f[1,1])
    xlims!(ax, 0.0, 14.0)
    ylims!(ax, 0.0, 1.0)
    scatter!(ax, lupien_data[:, 1], lupien_data[:, 2])
    scatter!(ax, t, ρ .* τ, color = :red)
    display(f)

    # return nothing


    visc_file = joinpath(exp_dir,"B1gViscvT_new.dat")
    data = readdlm(visc_file)
    rfit = curve_fit(model_3, data[1:300, 1], data[1:300, 2], [0.1866, 1.2, 0.1, 2.0])
    data[:, 2] .-= rfit.param[1] # Subtract off systematic offset

    t, η, _ = LudwigIO.read_property_from_file(joinpath(data_dir, "ηB1g_impurity_only_$(n_ε)x$(n_θ).dat"))
    η ./= τ
    fit = curve_fit(model_0, t, 1 ./ η, [1.0, 1.0], lower = [0.0, 0.0])
    @show fit.param

    η_model(t, p) = p[1] .+ p[2] * t.^rfit.param[4]
    t1, η1, _  = LudwigIO.read_property_from_file(joinpath(data_dir, "ηB1g_$(n_ε)x$(n_θ).dat"))  
    fit1 = curve_fit(model_1, t1, 1 ./ η1, [1.0, 0.1, 2.0])
    η0 = 1 / fit1.param[1]

    f = Figure(size = (1000, 600), fontsize = 24)
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

    lines!(ax, domain.^2, model_0(domain, fit.param) / fit.param[1], color = :grey, label = "Single Rate")
    scatter!(ax, t.^2, 1 ./ (fit.param[1] * η), color = :grey)

    lines!(ax, domain.^2, model_1(domain, fit1.param) / fit1.param[1], color = :red, label = "Ludwig.jl")
    scatter!(ax, t1.^2, fit1.param[1]^(-1) ./ η1, color = :red)

    axislegend(ax, position = :lt)

    display(f)
    outfile = joinpath(plot_dir, "single_rate_viscosity.png")
    # save(outfile, f)
end

function main()
    n_ε = 12
    n_θ = 38

    Uee  = 0.07441459021549257
    Vimp = 7.538183350719422e-5

    temps = 2.0:0.5:14.0

    # fit_ρ(n_ε, n_θ, temps, model_1, Uee, Vimp)

    # plot_η(n_ε, n_θ)
    # plot_lifetimes(n_ε, n_θ)

    # scaled_impurity_model(n_ε, n_θ)

    LudwigIO.write_property_to_file("τeff_σ", material, data_dir, n_ε, n_θ, Uee, 0.0, temps; addendum = "ee_only")

end 

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))
data_dir = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "model_2")
plot_dir = joinpath(@__DIR__, "..", "plots", "Sr2RuO4")
exp_dir  = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "experiment")

main()