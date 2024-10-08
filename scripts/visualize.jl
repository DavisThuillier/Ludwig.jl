using Ludwig
using CairoMakie, LaTeXStrings
using LinearAlgebra
using StaticArrays
using ForwardDiff

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

function deformation_potentials()
    N = 1000

     # Generate grid of 1st quadrant 
     x = LinRange(0.0, 0.5, N)


     for δ in 0.0:0.05:0.5
        f = Figure(fontsize = 20)
        xticks = map(x -> (x // 4), 0:1:16)
        ax = Axis(f[1,1], 
                xlabel = L"θ",
                ylabel = L"D_{xx}^2",
                xticks = xticks,
                xtickformat = values -> rational_format.(values),
                title = latexstring("\$\\delta = $(δ) \\mathrm{ eV}\$")
                )
        labels = [L"\alpha", L"\beta", L"\gamma"]

        for μ in 2:3
            E = map(x -> bands[μ]([x[1], x[2]]), collect(Iterators.product(x, x)))
        
            c = Ludwig.find_contour(x, x, E) # Generate Fermi surface contours
            points = c.isolines[1].points
            θ = map(x -> atan(x[2], x[1]), points) / pi
            if θ[2] < θ[1] # Fix orientation
                reverse!(θ)
                reverse!(points)
            end

            points = vcat(points, reverse(points))
            points = vcat(points, reverse(points))

            θ = vcat(θ, θ .+ 0.5) 
            θ = vcat(θ, θ .+ 1.0)

            lines!(ax, θ[3:end-1], dii_μ.(points[3:end-1], 1, μ, δ).^2, label = labels[μ])
            # lines!(ax, θ, label = labels[μ])
        end
        axislegend(ax)
        display(f)
    end

    #  outfile = joinpath(@__DIR__, "..", "plots", "deformation_potentials.png")
    #  save(outfile, f)
end

function get_contour(k1, k2, x, N)
    level = bands[3](k1) + bands[3](k2)

    E = map(x -> bands[3]([x[1], x[2]]) + bands[3]([x[1], x[2]] .- k1 .- k2), collect(Iterators.product(x, x)))

    c = Ludwig.find_contour(x, x, E, level)
    return c
end

function delta(k1, k2, N)
    x = LinRange(-0.5, 0.5, N)

    f = Figure()
    ax = Axis(f[1,1], aspect = 1.0)
    xlims!(ax, -0.5, 0.5)
    ylims!(ax, -0.5, 0.5)

    for δkx = -0.005:0.005:0.005
        for δky = -0.005:0.005:0.005
            c = get_contour(k1 + [δkx, δky], k2, x, N)

            for iso in c.isolines
                lines!(ax, iso.points, color = map(x -> norm(ForwardDiff.gradient(bands[3], x)), iso.points), colorrange = (-1, 1))
                @show iso.arclength
            end

            c = get_contour(k1, k2 + [δkx, δky], x, N)

            for iso in c.isolines
                lines!(ax, iso.points, color = map(x -> norm(ForwardDiff.gradient(bands[3], x)), iso.points), colorrange = (-1, 1))
            end
            # Colorbar(f[1,2], h)
            display(f)
        end
    end
end

function main()
    T = 12 * kb
    n_ε = 12
    n_θ = 60

    mesh = Ludwig.multiband_mesh(bands, orbital_weights, T, n_ε, n_θ)
    ℓ = length(mesh.patches)

    corner_ids = map(x -> x.corners, mesh.patches)

    quads = Vector{Vector{SVector{2, Float64}}}(undef, 0)
    for i in 1:size(corner_ids)[1]
        push!(quads, map(x -> mesh.corners[x], corner_ids[i]))
    end

    f = Figure(size = (1200,1200), fontsize = 24)
    ax = Axis(f[1,1],
              aspect = 1.0,
              limits = (-0.5,0.5,-0.5,0.5),
              xlabel = L"k_x",
              xticks = map(x -> (x // 8), 0:1:16),
              xtickformat = values -> rational_format.(values),
              ylabel = L"k_y",
              yticks = map(x -> (x // 8), 0:1:16),
              ytickformat = values -> rational_format.(values),
              xlabelsize = 30,
              ylabelsize = 30
    )

    p = poly!(ax, quads, color = map(x -> x.energy, mesh.patches), colormap = :viridis, strokecolor = :black, strokewidth = 0.2)

    # Colorbar(f[1,2], p, label = L"\varepsilon - \mu (\mathrm{eV})", labelsize = 30)
    display(f)

    save(joinpath(plot_dir,"SRO_t5_$(t5)_12.0_K.png"), f)
end

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))
plot_dir = joinpath(@__DIR__, "..", "plots", "Sr2RuO4")

for p in 0.02:0.01:0.13
    global t5 = p 
    main()
end
# deformation_potentials()

