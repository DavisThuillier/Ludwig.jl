using Ludwig
using CairoMakie
using LinearAlgebra

include(joinpath(@__DIR__, "materials", "Sr2RuO4.jl"))

bands = [bands[1]]

function main()
    T = kb * 12.0
    n_ε = 4
    n_θ = 20 

    grid    = Vector{Patch}(undef, 0)
    corners = Vector{SVector{2, Float64}}(undef, 0)
    for (i, h) in enumerate(bands)
        mesh, Δε = Ludwig.generate_mesh(h, T, n_ε, n_θ, 2000)
        grid = vcat(grid, map(x -> Patch(
                                    x.momentum, 
                                    x.energy,
                                    x.v,
                                    x.dV,
                                    x.de,
                                    x.jinv, 
                                    x.djinv,
                                    x.corners .+ length(corners)
                                ), mesh.patches)
        )
        corners = vcat(corners, mesh.corners)
    end
    @show size(corners)

    f = Figure()
    ax = Axis(f[1,1], aspect = 1.0, limits = (-0.5, 0.5, -0.5, 0.5))
    quads = []
    for (i, p) in enumerate(grid)
        push!(quads, map(x -> corners[x], p.corners))
    end
    poly!(ax, quads, color = map(x -> x.dV, grid), colormap = :viridis)

    xs = map(x -> x.momentum[1], grid)
    ys = map(x -> x.momentum[2], grid)
    us = map(x -> x.jinv[1,2] / norm(x.jinv[:, 2]), grid)
    vs = map(x -> x.jinv[2,2] / norm(x.jinv[:, 2]), grid)
    # us = map(x -> x.v[1] / norm(x.v), grid)
    # vs = map(x -> x.v[2] / norm(x.v), grid)
    arrows!(ax, xs, ys, us, vs, lengthscale = 0.01, arrowsize = 2, linecolor = :black, arrowcolor = :black)

    us = map(x -> x.jinv[1,1] / norm(x.jinv[:, 1]), grid)
    vs = map(x -> x.jinv[2,1] / norm(x.jinv[:, 1]), grid)
    arrows!(ax, xs, ys, us, vs, lengthscale = 0.01, arrowsize = 2, linecolor = :black, arrowcolor = :black)
    
    display(f)
    
end

main()