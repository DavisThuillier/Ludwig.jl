using Ludwig
using CairoMakie, LaTeXStrings
using LinearAlgebra
using StaticArrays

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

function main()
    T = 12 * kb
    n_ε = 12
    n_θ = 38
    # mesh = Ludwig.multiband_mesh(bands, orbital_weights, T, n_ε, n_θ)
    # ℓ = length(mesh.patches)

    # corner_ids = map(x -> x.corners, mesh.patches)

    # quads = Vector{Vector{SVector{2, Float64}}}(undef, 0)
    # for i in 1:size(corner_ids)[1]
    #     push!(quads, map(x -> mesh.corners[x], corner_ids[i]))
    # end

    x = LinRange(-0.5, 0.5, 1000)
    E = map(x -> ε1([x[1], x[2]]), collect(Iterators.product(x, x)))
    c = Ludwig.find_contour(x,x,E)

    f = Figure(size = (1000,1000))
    ax = Axis(f[1,1],
              aspect = 1.0,
            #   limits = (-0.5,0.5,-0.5,0.5)
    )
    
    h = heatmap!(ax, x,x ,E)
    for iso in c.isolines
        lines!(ax, iso.points)
    end

    # p = poly!(ax, quads, color = map(x -> x.energy, mesh.patches), colormap = :viridis)

    # xs = map(x -> x.momentum[1], mesh.patches)
    # ys = map(x -> x.momentum[2], mesh.patches)
    # us = map(x -> inv(x.jinv)[1,1] / norm(inv(x.jinv)[1,:]), mesh.patches)
    # vs = map(x -> inv(x.jinv)[1,2] / norm(inv(x.jinv)[1, :]), mesh.patches)
    # us = map(x -> x.v[1] / norm(x.v), grid)
    # vs = map(x -> x.v[2] / norm(x.v), grid)
    # arrows!(ax, xs, ys, us, vs, lengthscale = 0.01, arrowsize = 3, linecolor = :black, arrowcolor = :black)

    # scatter!(ax, map(x -> x.momentum, mesh.patches))
    Colorbar(f[1,2], h)
    display(f)
end

function form_factors()
    T = 12 * kb
    n_ε = 12
    n_θ = 38
    mesh = Ludwig.multiband_mesh(bands, orbital_weights, T, n_ε, n_θ)

    xticks = map(x -> (x // 4), -8:8)

    N = 60
    krange = LinRange(-0.5, 0.5, N)

    for i in 1:6:2*length(mesh.patches)÷3
        weights = Ludwig.multiband_weight(mesh.patches[i], vertex_pk, N)

        k = mesh.patches[i].momentum
        θ = Ludwig.get_angle(k)
        for μ in 1:1
            f = Figure()
            ax = Axis(f[1,1], 
                      aspect = 1.0, 
                      title  = latexstring("Band $(mesh.patches[i].band_index), \$\\theta = \$ $(round(θ, digits = 4))"),
                      xlabel = L"k_x",
                      ylabel = L"k_y",
                      xticks = xticks,
                      xtickformat = values -> rational_format.(values),
                      yticks = xticks,
                      ytickformat = values -> rational_format.(values)
                      )
            h = heatmap!(ax, krange, krange, weights[:, :, μ], colorrange = (-1.0, 1.0))
            Colorbar(f[1,2], h, )
            display(f)
        end
    end

    
end

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))

# form_factors()
# main()
deformation_potentials()
