using Ludwig
using CairoMakie
using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))

function rational_format(x)
    x = Rational(x)
    if x == 0
        return L"0"
    elseif x == 1
        return L"\pi"
    elseif denominator(x) == 1
        L"%$(numerator(x)) \pi"
    else 
        return L"\frac{%$(numerator(x))}{%$(denominator(x))} \pi"
    end
end

function deformation_potentials()
    N = 1000

     # Generate grid of 1st quadrant 
     x = LinRange(0.0, 0.5, N)
     f = Figure(fontsize = 20)
     xticks = map(x -> (x // 4), 0:1:16)
     ax = Axis(f[1,1], 
               xlabel = L"θ",
               ylabel = L"D_{xx}^2",
               xticks = xticks,
               xtickformat = values -> rational_format.(values)
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

        lines!(ax, θ[3:end-1], dii_μ.(points[3:end-1], 1, μ).^2, label = labels[μ])
        # lines!(ax, θ, label = labels[μ])
     end
     axislegend(ax)
     display(f)

    #  outfile = joinpath(@__DIR__, "..", "plots", "deformation_potentials.png")
    #  save(outfile, f)
end

function main()
    T = 12 * kb
    n_ε = 9
    n_θ = 60
    mesh, _ = Ludwig.multiband_mesh(hamiltonian, T, n_ε, n_θ)

    corner_ids = map(x -> x.corners, mesh.patches)
    quads = Vector{Vector{SVector{2, Float64}}}(undef, 0)
    for i in 1:size(corner_ids)[1]
        push!(quads, map(x -> mesh.corners[x], corner_ids[i]))
    end

    f = Figure(size = (1000,1000))
    ax = Axis(f[1,1],
              aspect = 1.0,
              limits = (-0.5,0.5,-0.5,0.5)
    )
    
    ℓ = length(mesh.patches)
    # Dxx = Vector{Float64}(undef, ℓ)
    # Dyy = Vector{Float64}(undef, ℓ)
    # index = Vector{Float64}(undef, ℓ)
    # for i in 1:ℓ
    #     μ = (i-1)÷(ℓ÷3) + 1 # Band index
        
    #     if μ == 2; μ += 1
    #     elseif μ == 3; μ -= 1
    #     end

    #     index[i] = μ
    #     @show μ

    #     Dxx[i] = dii_μ(mesh.patches[i].momentum, 1, μ)
    #     Dyy[i] = dii_μ(mesh.patches[i].momentum, 2, μ)
    # end

    # for patch in mesh.patches
    #     display(patch.W)
    #     return nothing
    # end
    # for p in mesh.patches
    #     @show p.w
    # end

    p = poly!(ax, quads, color = map(x -> argmax(x.energies), mesh.patches), colormap = :viridis)

    xs = map(x -> x.momentum[1], mesh.patches)
    ys = map(x -> x.momentum[2], mesh.patches)
    us = map(x -> inv(x.jinv)[1,1] / norm(inv(x.jinv)[1,:]), mesh.patches)
    vs = map(x -> inv(x.jinv)[1,2] / norm(inv(x.jinv)[1, :]), mesh.patches)
    # us = map(x -> x.v[1] / norm(x.v), grid)
    # vs = map(x -> x.v[2] / norm(x.v), grid)
    # arrows!(ax, xs, ys, us, vs, lengthscale = 0.01, arrowsize = 3, linecolor = :black, arrowcolor = :black)

    
    Colorbar(f[1,2], p)
    display(f)
end

main()
# deformation_potentials()