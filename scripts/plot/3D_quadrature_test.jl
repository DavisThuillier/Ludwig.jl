using Pkg
Pkg.activate(@__DIR__)

using Ludwig
using GLMakie
using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "..", "materials", "simple_cubic.jl"))

function main()

    point_group = PointGroup(generators, primitives)
    crystal = Crystal(primitives, point_group)
    
    ibz = Ludwig.irreducible_brillouin_zone(crystal)
    ibz_vertices = vertices(ibz)
    ibz_vertices = vcat(ibz_vertices, [first(ibz_vertices)])

    T = 100 # kelvin
    energy_window = 12.0 * kb * T
    N_s = 5
    N_ε = 4

    nodes, weights, residuals = Ludwig.bz_quadrature_rule(dispersion, velocity, crystal, energy_window, N_s, N_ε; initial_depth = 7, refinement_depth = 11, progress = true, method = :maxvol, α = 1.0)

    fig = Figure(size = (700, 700))
    ax = Axis3(fig[1, 1];
        aspect = :data,
        xlabel = L"k_x",
        ylabel = L"k_y",
        zlabel = L"k_z",
        # title = "",
    )
        
    scatter!(ax, nodes; color = weights, colorscale = log)

    display(fig)
end

main()
