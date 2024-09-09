using Ludwig
using HDF5
using StaticArrays
using CairoMakie, LaTeXStrings, Colors
using LinearAlgebra
using LsqFit
using DelimitedFiles
using ProgressBars
import Statistics: mean

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

function symmetrize(L, dV, E, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    D = diagm(dV .* fd .* (1 .- fd))

    return 0.5 * (L + inv(D) * L' * D)
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
    # L *= 0.5 * Uee^2

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

function load(file, T; symmetrized = false)
    h5open(file, "r") do fid
        g = fid["data"]

        L = 2pi * read(g, "L") 
        k = reinterpret(SVector{2,Float64}, vec(transpose(read(g, "momenta"))))
        v = reinterpret(SVector{2,Float64}, vec(transpose(read(g, "velocities"))))
        E = read(g, "energies")
        dV = read(g, "dVs")
        corners = read(g, "corners")
        corner_ids = read(g, "corner_ids")

        symmetrized && (L = symmetrize(L, dV, E, kb * T))

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

function impurity_only(n_ε, n_θ, Vimp)
    t = 2.0:0.5:14.0

    σ = Vector{Float64}(undef, length(t))
    η = Vector{Float64}(undef, length(t))
    τ_σ = Vector{Float64}(undef, length(t))
    τ_η = Vector{Float64}(undef, length(t))

    for i in eachindex(t)
        impfile = joinpath(data_dir, "Sr2RuO4_unitary_imp_$(t[i])_$(n_ε)x$(n_θ).h5")
        Γ, k, v, E, dV, _, _ = load(impfile, t[i]; symmetrized = false)

        Γ *= Vimp^2

        Dxx = zeros(Float64, ℓ)
        Dyy = zeros(Float64, ℓ)
        for i in 1:ℓ
            μ = (i-1)÷(ℓ÷3) + 1 # Band index
            Dxx[i] = dii_μ(k[i], 1, μ, 0.0)
            Dyy[i] = dii_μ(k[i], 2, μ, 0.0)
        end

        σ[i] = Ludwig.longitudinal_conductivity(Γ, first.(v), E, dV, kb * t[i]) / c
        η[i] = Ludwig.ηB1g(Γ, E, dV, Dxx, Dyy, kb * t[i]) / (a^2 * c)
        τ_σ[i] = Ludwig.σ_lifetime(Γ, v, E, dV, kb * t[i])
        τ_η[i] = Ludwig.η_lifetime(Γ, Dxx, Dyy, E, dV, kb * t[i])
        @show σ[i], η[i], τ_σ[i], τ_η[i]
    end

    @show σ
    @show η
    @show τ_σ
    @show τ_η

    f = Figure(fontsize = 20)
    ax = Axis(f[1,1], ylabel = L"\tau_\text{eff}^{-1}\,(\mathrm{ps}^{-1})", xlabel = L"T\, (\mathrm{K})", 
    xticks = [4, 16, 25, 36, 49, 64, 81, 100, 144, 169, 196],
                xtickformat = values -> [L"%$(Int(sqrt(x)))^2" for x in values])
                xlims!(ax, 0, 200)
    scatter!(ax, t.^2, 1e-12 ./ σ_τ, label = L"\tau_\sigma")
    scatter!(ax, t.^2, 1e-12 ./ η_τ, label = L"\tau_\eta")
    axislegend(ax, position = :lt)
    display(f)

end

function get_property(prop::String, T, n_ε::Int, n_θ::Int, Uee::Real, Vimp::Real, δ = 0.0; include_impurity::Bool = true)
    eefile = joinpath(data_dir, "Sr2RuO4_$(Float64(T))_$(n_ε)x$(n_θ).h5")
    
    Γ, k, v, E, dV, _, _= load(eefile, T, symmetrized = true)
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
            Dxx[i] = dii_μ(k[i], 1, μ, δ)
            Dyy[i] = dii_μ(k[i], 2, μ, δ)
        end

        return Ludwig.ηB1g(Γ, E, dV, Dxx, Dyy, T) / (a^2 * c)
    elseif prop == "τ_eff"
        # Effective lifetime as calculated from the conductivity
        τ1 = Ludwig.σ_lifetime(Γ, v, E, dV, T)

        Dxx = zeros(Float64, ℓ)
        Dyy = zeros(Float64, ℓ)
        for i in 1:ℓ
            μ = (i-1)÷(ℓ÷3) + 1 # Band index
            Dxx[i] = dii_μ(k[i], 1, μ, δ)
            Dyy[i] = dii_μ(k[i], 2, μ, δ)
        end
        τ2 = Ludwig.η_lifetime(Γ, Dxx, Dyy, E, dV, T)
        return τ1, τ2
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

# function τ_eff(n_ε, n_θ, Uee, Vimp)
#     T = 2.0:0.5:14.0

#     file = joinpath(data_dir, "τ_eff.dat")
#     open(file, "w") do f
#         write(f, "#τ_eff from conductivity")
#         write(f, "\n#n_ε = $(n_ε), n_θ = $(n_θ)")
#         write(f, "\n#Uee = $(Uee) eV")
#         write(f, "\n#√n * Vimp = $(Vimp) eV m^{-3/2}")
#         write(f, "\n#T(K) τ_eff (ps)")
#         for i in eachindex(T)
#             τ = get_property("τ_eff", T[i], n_ε, n_θ, Uee, Vimp, include_impurity = false) * 1e12
#             write(f, "\n$(T[i]) $(τ)")
#         end
#     end 
# end

function display_heatmap(file, T)
    Γ, k, v, E, dV, corners, corner_ids = load(file, T)

    T = kb * T


    fd = f0.(E, T) # Fermi dirac on grid points
    D = diagm(sqrt.(dV .* fd .* (1 .- fd)))
    M = D * Γ * inv(D)

    ℓ = size(Γ)[1]
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

    # λ = eigvals(M)
    # @show λ[1:4]
    # f = Figure()
    # ax = Axis(f[1,1])
    # scatter!(ax, real.(λ))
    # display(f)
end

function modes(T, n_ε, n_θ, Uee, Vimp; include_impurity = false)
    file = joinpath(data_dir, "Sr2RuO4_$(T)_$(n_ε)x$(n_θ).h5")

    L, k, v, E, dV, corners, corner_ids = load(file, T; symmetrized = true)
    L *= 0.5 * Uee^2

    if include_impurity
        impfile = joinpath(data_dir, "Sr2RuO4_unitary_imp_$(T)_$(n_ε)x$(n_θ).h5")
        Limp, _, _, _, _, _, _ = load(impfile, T)
        L += Limp * Vimp^2
    end

    T *= kb

    @time eigenvalues  = eigvals(L)
    eigenvalues *= 1e-9 / hbar

    f = Figure(size = (700, 500), fontsize = 24)
    ax = Axis(f[1,1],
            xlabel = "Mode Index",
            ylabel = L"\lambda \,(\mathrm{ps}^{-1})"
    )
    for i in 1:50
        if i  < 5
            scatter!(ax, i, eigenvalues[i], 
            color = :gray, 
            )
        else
            scatter!(ax, i, eigenvalues[i], 
            #color = parities[i] > 0 ? :orange : :blue, 
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

    # outfile = joinpath(plot_dir, "23 August 2024", "Sr2RuO4_spectrum_1_to_50.png")
    # save(outfile, f)

    # return nothing
    
    @time eigenvectors = eigvecs(L)
    parities = Vector{Float64}(undef, length(eigenvalues))
    for i in eachindex(eigenvalues)
        parities[i] = parity(eigenvectors[:, i])
    end

    @show eigenvalues[1:4]

    quads = Vector{Vector{SVector{2, Float64}}}(undef, 0)
    for i in 1:size(corner_ids)[1]
        push!(quads, map(x -> SVector{2}(corners[x, :]), corner_ids[i, :]))
    end

    N = 4
    
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

function ρ_fit(n_ε, n_θ)
    # Fit to Lupien's data
    lupien_file = joinpath(exp_dir, "rhovT_Lupien_digitized.dat")
    lupien_data = readdlm(lupien_file)
    
    lupien_T = lupien_data[11:60,1]
    lupien_ρ = lupien_data[11:60,2]

    lupien_model(t, p) = p[1] .+ p[2] * t.^2 #p[3]
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

function get_ρ(n_ε, n_θ, Uee, Vimp, T)
    σ = get_property("σxx", T, n_ε, n_θ, Uee, Vimp; include_impurity = true)
    println("ρ = $(1e8 / σ) μΩ-cm")
end

function plot_ρ(n_ε, n_θ, Uee, Vimp)
    t = 2.0:0.5:12.0

    # σ = Vector{Float64}(undef, length(t))
    # for i in eachindex(t)
    #     σ[i] = get_property("σxx", t[i], n_ε, n_θ, Uee, Vimp; include_impurity = true)
    #     @show 10^8 / σ[i]
    # end

    # ρ = 10^8  ./ σ # Convert to μΩ-cm
    # @show ρ

    ρ = [0.10959056505116326, 0.12151019463132146, 0.13504856338798898, 0.15016845154757302, 0.16672467041152575, 0.1847198405479588, 0.2041851433762483, 0.22516814159799386, 0.24777750498999843, 0.27197760942785343, 0.29784506878946493, 0.32543193395073594, 0.35473522979103506, 0.38579389146608156, 0.4186462443045558, 0.4534385594028692, 0.49009715381486624, 0.5286627463900118, 0.5691567274598581, 0.6116340508555088, 0.6561653206886255]
    

    model(t, p) = p[1] .+ p[2] * t.^2

    fit = curve_fit(model, t, ρ, [0.0, 0.0001, 2.0], lower = [0.0, 0.0, 0.0])
    # @show fit.param[2] / fit.param[1]

    f = Figure(fontsize = 20)
    ax = Axis(f[1,1],
              xlabel = L"T(\mathrm{K})",
              ylabel = L"\rho (\mathrm{\mu\Omega \,cm})")

    xlims!(ax, 0.0, 14.0)
    ylims!(ax, 0.0, 1.0)

    lupien_file = joinpath(exp_dir, "rhovT_Lupien_digitized.dat")
    lupien_data = readdlm(lupien_file)
    lfit = curve_fit(model, lupien_data[10:40, 1], lupien_data[10:40, 2], [0.0, 0.0001, 2.0], lower = [0.0, 0.0, 0.0])
    @show lfit.param[2] / lfit.param[1]
    scatter!(ax, lupien_data[:, 1], lupien_data[:, 2], color = :black)
    domain = 0.0:0.1:30.0

    # for r in 0.1:0.1:1.0
    #     data_file = joinpath(data_dir, "ρ_r_$(r).dat")

    #     T = Float64[]
    #     ρ = Float64[]
    #     open(data_file, "r") do file
    #         for line in readlines(file)
    #             startswith(line, "#") && continue
    #             Ti, ρi = split(line, ",")
    #             push!(T, parse(Float64, Ti))
    #             push!(ρ, parse(Float64, ρi))
    #         end
    #     end

        
    #     fit = curve_fit(model, T, ρ, [0.0, 0.0001, 2.0], lower = [0.0, 0.0, 0.0])
        
    #     lines!(ax, domain, (model(domain, fit.param)), color = r, colorrange = (0.0, 1.0), colormap = :thermal)
    #     scatter!(ax, T, ρ, color = r, colorrange = (0.1, 1.0), colormap = :thermal)

    #     display(f)

    # end

    # scatter!(ax, t, ρ, color = :red)
    lines!(ax, domain, model(domain, fit.param), color = :red)
    # lines!(ax, domain, model(domain, lfit.param))
    display(f)

    outfile = joinpath(plot_dir, "23 August 2024", "ρ_with_fit.png")
    # save(outfile, f)

end

function get_η(n_ε, n_θ, Uee, Vimp, T)
    δ = 0.0
    η = get_property("ηB1g", T, n_ε, n_θ, Uee, Vimp, δ; include_impurity = true)
    println("$(n_ε)x$(n_θ), $(η)")
end

function plot_η()
    # data_file = joinpath(data_dir, "ηB1g_sans_crystal_field_splitting.dat")
    

    # Fit to Brad's data
    visc_file = joinpath(exp_dir,"B1gViscvT_new.dat")
    data = readdlm(visc_file)
    rmodel(t, p) = p[1] .+ 1 ./ (p[2] .+ p[3] * t.^p[4])#2)
    rfit = curve_fit(rmodel, data[1:341, 1], data[1:341, 2], [0.1866, 1.2, 0.1, 2.0])
    @show rfit.param
    @show rfit.param[3] / rfit.param[2]
    data[:, 2] .-= rfit.param[1] # Subtract off systematic offset

    f = Figure()
        ax = Axis(f[1,1], 
                aspect = 1.0,
                ylabel = L"\eta^{-1} ((\mathrm{Pa\,s})^{-1})",
                xlabel = L"T^2(\mathrm{K^2})",
                xticks = [4, 9, 16, 25, 36, 49, 64, 81, 100, 144, 169, 196],
                xtickformat = values -> [L"%$(Int(sqrt(x)))^2" for x in values],
                yticks = 0:2:40
                )
                xlims!(0.0, 14^2)
        ylims!(ax, 0.0, 40)
        xlims!(ax, 0.0, 13^2)
        domain = 0.0:0.1:14.0
        scatter!(ax, data[:, 1].^2, 1 ./ data[:, 2], color = :black)
        # lines!(ax, domain.^2, 1 ./ (rmodel(domain, rfit.param) .- rfit.param[1]))

    for r in 0.1:0.1:1.0
        data_file = joinpath(data_dir, "ηB1g_r_$(r)_δ_0.0.dat")

        T = 14.0:-2.0:2.0
        ηB1g = Float64[]
        open(data_file, "r") do file
            for line in readlines(file)
                startswith(line, "#") && continue
                push!(ηB1g, parse(Float64, line))
            end
        end

        model(t, p) = 1 ./ (p[1] .+ p[2] * t.^p[3])
        fit = curve_fit(model, T, ηB1g, [1.0, 0.1, 2.0])
        @show fit.param
        @show fit.param[2] / fit.param[1]
        
        lines!(ax, domain.^2, 1 ./ (model(domain, fit.param)), color = r, colorrange = (0.1, 1.0), colormap = :thermal)
        scatter!(ax, T.^2, 1 ./ ηB1g, color = r, colorrange = (0.1, 1.0), colormap = :thermal)

        # mean_ratio = sum((rmodel(domain, rfit.param) .- rfit.param[1]) ./ (model(domain, fit.param))) / length(domain)
        # @show mean_ratio

        display(f)

    end
        # outfile = joinpath(@__DIR__, "..", "plots", "η_unitary.png")
        # save(outfile, f)

end

function η_fit(Uee, Vimp)
    temps = 2.0:2.0:14.0

    visc_file = joinpath(exp_dir,"B1gViscvT_new.dat")
    data = readdlm(visc_file)
    rmodel(t, p) = p[1] .+ 1 ./ (p[2] .+ p[3] * t.^p[4])
    rfit = curve_fit(rmodel, data[1:341, 1], data[1:341, 2], [0.1866, 1.2, 0.1, 2.0])

    model(t, p) = p[1] .+ p[2] * t.^p[3]

    # @show model(temps, rfit.param[2:end])
    # return nothing

    Vfit_model(t, p) = begin 
        @show p
        ηinv = Vector{Float64}(undef, length(t))
        for i in eachindex(t)
            ηinv[i] = 1 / get_property("ηB1g", t[i], n_ε, n_θ, Uee, Vimp, p[1]; include_impurity = true)
            @show ηinv[i]
        end

        χ = sqrt(sum((ηinv .- model(t, rfit.param[2:end])).^2) / length(t))
        @show χ
        ηinv
    end

    guess = [0.4]
    Vfit = curve_fit(Vfit_model, temps, model(temps, rfit.param[2:end]), guess, lower = [0.0])
    @show Vfit.param

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

function plot_ρ_and_η(n_ε, n_θ, Uee, Vimp)
    T = 2.0:0.5:12.0

    # ρ = Vector{Float64}(undef, length(T))
    # η = Vector{Float64}(undef, length(T))
    # for i in eachindex(T)
    #     ρ[i] = 1e8 / get_property("σxx", T[i], n_ε, n_θ, Uee, Vimp; include_impurity = true)
    #     η[i] = get_property("ηB1g", T[i], n_ε, n_θ, Uee, Vimp; include_impurity = true)
    # end

    # @show ρ
    # @show η

    # ρ = [0.14572187053941366, 0.1583948568367494, 0.1728425862769762, 0.18901282110967638, 0.20667432212019982, 0.2257887321860733, 0.24635849964177697, 0.2684105636171452, 0.29204875574948685, 0.31721993847220736, 0.34400235558405456, 0.3724488327795807, 0.4025541823326671, 0.4343602379900469, 0.4679118286333832, 0.5033662819541096, 0.5406422926834203, 0.5797838939355607, 0.6208156109261806, 0.6637970814224016, 0.7088032591626268]
    # η = [0.1632109937471985, 0.14104169141627979, 0.12153028700859449, 0.104572203676835, 0.09046997281210196, 0.07878866672121872, 0.06909612045431311, 0.06101620796348812, 0.054204113192094124, 0.04846692488169224, 0.043586830506469956, 0.03940454789543453, 0.035803078584045474, 0.03268040808617452, 0.029947369120300646, 0.027530396713247595, 0.02539869935048196, 0.02350404774632313, 0.021813544105868457, 0.020296558737697944, 0.01892834501990456]

    ρ = [0.1184466057991059, 0.13021347042993292, 0.143596080709527, 0.15854907572090784, 0.17489943326990434, 0.19263351064614767, 0.21177117405749082, 0.2323511844774808, 0.2544772463142182, 0.27810943094383744, 0.3033228627991061, 0.33016842764032833, 0.3586429551527171, 0.38878524297657274, 0.4206348950232471, 0.45433677221108826, 0.48981748828118055, 0.527117413936166, 0.5662584554497896, 0.6072952203987327, 0.6502972674520632]
    η = [0.1958796881072882, 0.16627525782177574, 0.14109624269862953, 0.11989784757382546, 0.10269560654437497, 0.08873094916369476, 0.07733182090541726, 0.06795287218998933, 0.06012861084233261, 0.05359280940948254, 0.04806950193915352, 0.043360191266380674, 0.03932100989922829, 0.03582971528067413, 0.032782009364159224, 0.030092616428403475, 0.027724258460274275, 0.02562214168020945, 0.023748640955960847, 0.022069190007005773, 0.020555898057792395]

    
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
    

    ρ_model(t, p) = p[1] .+ p[2] * t.^2
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

function lifetimes(n_ε, n_▓, Uee, Vimp)
    t = 2.0:0.5:14.0

    # σ_τ = Vector{Float64}(undef, length(t))
    # η_τ = Vector{Float64}(undef, length(t))
    # for i in eachindex(t)
    #     σ_τ[i], η_τ[i] = get_property("τ_eff", t[i], n_ε, n_θ, Uee, Vimp, include_impurity = false)
    # end
    # @show σ_τ
    # @show η_τ

    # With impurity 
    # σ_τ = [1.5379426604419398e-11, 1.4153664096410503e-11, 1.2974288764480237e-11, 1.1868653223645244e-11, 1.085768150490269e-11, 9.941054095654228e-12, 9.113114627721576e-12, 8.366103226176869e-12, 7.690377533893804e-12, 7.081362547616889e-12, 6.53107027355719e-12, 6.033135019923649e-12, 5.582724094619536e-12, 5.174610054588276e-12, 4.8042319709706505e-12, 4.4664228478470644e-12, 4.158960002210765e-12, 3.8786306267985055e-12, 3.622692805796296e-12, 3.3884820801602004e-12, 3.1736591921403806e-12, 2.9761941634076086e-12, 2.794276054874725e-12, 2.6275855240600728e-12, 2.4721297922739584e-12]
    # η_τ = [2.248378867683754e-11, 1.946472905613255e-11, 1.6795271439185808e-11, 1.447625912345739e-11, 1.2539689018403753e-11, 1.09303620359893e-11, 9.591979482517447e-12, 8.473947321202503e-12, 7.529911842016506e-12, 6.7337727226534815e-12, 6.055771930112863e-12, 5.474114416173493e-12, 4.972727140608846e-12, 4.537548807223898e-12, 4.156413631726802e-12, 3.818893787055368e-12, 3.5206570916361886e-12, 3.255576395242323e-12, 3.01897651414712e-12, 2.8062912217134144e-12, 2.613698490838313e-12, 2.438966209794292e-12, 2.279390037396542e-12, 2.1357649647625993e-12, 2.001359872133798e-12] 

    # Without impurity
    σ_τ = [1.6026181361056132e-10, 1.0257874532699684e-10, 7.122884944705212e-11, 5.231135463761146e-11, 4.003171850295472e-11, 3.161152212547963e-11, 2.558739567328745e-11, 2.1127772272118413e-11, 1.7728227576577122e-11, 1.5084432121081086e-11, 1.298584640055659e-11, 1.1291975448698414e-11, 9.906714199279813e-12, 8.75925855970228e-12, 7.798313437706531e-12, 6.983388764992457e-12, 6.288043711259425e-12, 5.689913133654263e-12, 5.171843460288032e-12, 4.719938197011751e-12, 4.3231518819058414e-12, 3.972660830595949e-12, 3.661360636080351e-12, 3.3853421254420406e-12, 3.135942743061621e-12]
    η_τ = [1.4601236133977752e-10, 9.341611693458268e-11, 6.483038412978151e-11, 4.7540576559707795e-11, 3.633253502772071e-11, 2.8654909104088672e-11, 2.3166338391120787e-11, 1.9104520287446257e-11, 1.6004547583011295e-11, 1.359607106660144e-11, 1.1684719218168684e-11, 1.0142255776852013e-11, 8.881562328189892e-12, 7.837937214022785e-12, 6.963471153891217e-12, 6.220614430388409e-12, 5.5871948493785754e-12, 5.042819826813676e-12, 4.571816415517399e-12, 4.160945361616479e-12, 3.799466277893721e-12, 3.480169068299188e-12, 3.1959823789679095e-12, 2.9454210472643614e-12, 2.718162848164906e-12]

    @show σ_τ ./ η_τ

    f = Figure(fontsize = 20)
    ax = Axis(f[1,1], ylabel = L"\tau_\text{eff}^{-1}\,(\mathrm{ps}^{-1})", xlabel = L"T\, (\mathrm{K})", 
    xticks = [4, 16, 25, 36, 49, 64, 81, 100, 144, 169, 196],
                xtickformat = values -> [L"%$(Int(sqrt(x)))^2" for x in values])
                xlims!(ax, 0, 200)
    scatter!(ax, t.^2, 1e-12 ./ σ_τ, label = L"\tau_\sigma")
    scatter!(ax, t.^2, 1e-12 ./ η_τ, label = L"\tau_\eta")
    axislegend(ax, position = :lt)
    display(f)
    # save(joinpath(plot_dir, "23 August 2024", "τ_eff.png"),f)

end

function γ_lifetimes(n_ε, n_▓, Uee, Vimp, include_impurity)
    t = 2.0:0.5:14.0

    σ_τ = Vector{Float64}(undef, length(t))
    η_τ = Vector{Float64}(undef, length(t))
    for i in eachindex(t)
        println("T = $(t[i])")
        file = joinpath(data_dir, "Sr2RuO4_γ_$(t[i])_$(n_ε)x$(n_θ).h5")

        L, k, v, E, dV, corners, corner_ids = load(file, t[i]; symmetrized = true)
        ℓ = size(L)[1]
        L *= 0.5 * Uee^2

        # if include_impurity
        #     impfile = joinpath(data_dir, "Sr2RuO4_unitary_imp_$(T)_$(n_ε)x$(n_θ).h5")
        #     Γimp, _, _, _, _, _, _ = load(impfile, T)
        #     Γ += Γimp * Vimp^2
        # end

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

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))
data_dir = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "model_2")
plot_dir = joinpath(@__DIR__, "..", "plots", "Sr2RuO4")
exp_dir  = joinpath(@__DIR__, "..", "data", "Sr2RuO4", "experiment")

n_ε = 14
n_θ = 44
Uee = 0.07517388226660576
Vimp = 8.647920506354473e-5

# get_ρ(14, 40, Uee, Vimp, 12.0)
for n_θ ∈ 38:2:40
    get_η(n_ε, n_θ, Uee, Vimp, 12.0)
end

# display_heatmap(joinpath(data_dir, "Sr2RuO4_12.0_12x38.h5"), 12)

# modes(12.0, n_ε, n_θ, Uee, Vimp, include_impurity = false)
# separate_band_conductivities(n_ε, n_θ, Uee, Vimp)
# lifetimes(n_ε, n_θ, Uee, Vimp)
# γ_lifetimes(n_ε, n_θ, Uee, Vimp, false)
# impurity_only(n_ε, n_θ, Vimp)

ℓ = 4 * (n_ε - 1) * (n_θ - 1)

# i = 2 * ℓ + Int( ((n_ε - 1) + 1) * (n_θ ÷ 2 - 1.5))
# @show i

# ρ_fit(n_ε, n_θ)
# plot_ρ_and_η(n_ε, n_θ, Uee, Vimp)

# visualize_rows(3460, 12.0, n_ε, n_θ)

# η_fit(Uee, Vimp)

# plot_η()
# get_η(n_ε, n_θ, Uee, Vimp)

# get_ρ(n_ε, n_θ, Uee, Vimp)
# plot_ρ(12, 38, Uee, Vimp)

# mirror_gamma_modes(12.0, n_ε, n_θ, Uee)