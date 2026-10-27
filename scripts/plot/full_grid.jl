using Pkg
Pkg.activate(@__DIR__)

using Revise
using Ludwig
using CairoMakie
using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "..", "materials", "graphene.jl"))

function main()
    point_group = PointGroup(generators, primitives)
    crystal = Crystal(primitives, point_group)
    ibz = irreducible_brillouin_zone(crystal)
    bz = brillouin_zone(crystal)
    # @show reciprocal_lattice_vectors(crystal)
    
    N_s = 20
    N_ε = 36
    T   = kb * 40
    energy_window = 12.0 * T

    nodes, weights, residuals = Ludwig.bz_quadrature_rule(dispersion, velocity, crystal, energy_window, N_s, N_ε; initial_depth = 2, refinement_depth = 18, progress = true)
    @show length(nodes)
    @show residuals
    
    kmax = 4π/ 3
    @time mesh = level_set(dispersion, 0.0, (-kmax, kmax), (-kmax, kmax), 10;   
        refinement_depth = 18,
        mask = bz, weld = true,
    )
    mesh_vertices = vertices(mesh)
    segment_points = [Point2f(mesh_vertices[i][1], mesh_vertices[i][2])
                      for simplex in simplices(mesh) for i in simplex]



    fig = Figure(size = (500, 500))
    ax = Axis(fig[1, 1];
        aspect = DataAspect(), 
        xlabel = L"k_x", 
        ylabel = L"k_y",
        xgridvisible = false,
        ygridvisible = false
    
    )
    # xlims!(ax, 0.9 * (4π/3), 4π/3)
    # ylims!(ax, -0.05 * (4π/3), 0.05 * 4π/3)
    xlims!(ax, 0.9 * (2π/3), 1.05 * 2π/3)
    ylims!(ax, 0.95 * 2π/sqrt(3), 2π/sqrt(3))

    bz = brillouin_zone(crystal)
    bz_vertices = vertices(bz)
    sort!(bz_vertices, by = corner -> atan(corner[2], corner[1]))

    bz_vertices = vcat(bz_vertices, [first(bz_vertices)])
    lines!(ax, bz_vertices; color= :black)

    ibz_vertices = vertices(ibz)
    ibz_vertices = vcat(ibz_vertices, [first(ibz_vertices)])
    lines!(ax, ibz_vertices; color= :black)

    linesegments!(ax, segment_points; color = :orange, linewidth = 2)

    scatter!(ax, nodes; color = weights, markersize = 3, colorscale = log)

    display(fig)
        
end

main()
