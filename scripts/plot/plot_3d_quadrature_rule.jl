using Pkg
Pkg.activate(@__DIR__)

using Ludwig
using GLMakie
using StaticArrays
using DelimitedFiles

include(joinpath(@__DIR__, "local_settings.jl"))
const SETTINGS = LOCAL_SETTINGS.plot_3d_quadrature_rule

include(joinpath(@__DIR__, "data_io.jl"))

"""
    build_fermi_surface_mesh(ε, crystal; initial_depth = 4, refinement_depth = 6)

Return a coarse triangulation of the Fermi surface (`ε = 0`) clipped to the IBZ of `crystal`.
"""
function build_fermi_surface_mesh(ε, crystal; initial_depth = 4, refinement_depth = 6)
    ibz = irreducible_brillouin_zone(crystal)
    bounds = Ludwig.axis_aligned_bounding_box(ibz)
    return level_set(ε, 0.0, bounds..., initial_depth; refinement_depth, mask = ibz, weld = false)
end

function main(nodes, weights, N_ibz)
    @show length(nodes) N_ibz
    # nodes = nodes[1:N_ibz]
    # weights = weights[1:N_ibz]

    point_group = PointGroup(generators, primitives)
    crystal = Crystal(primitives, point_group)
    surface = build_fermi_surface_mesh(dispersion, crystal)
    surface_points = Point3f.(vertices(surface))
    surface_simplices = simplices(surface)
    faces = [surface_simplices[i][j] for i in eachindex(surface_simplices), j in 1:3]

    fig = Figure(size = (500, 500))
    ax = Axis3(fig[1, 1];
        aspect = :data,
        xlabel = L"k_x",
        ylabel = L"k_y",
        xgridvisible = false,
        ygridvisible = false
    )

    SETTINGS.xlims === nothing || xlims!(ax, SETTINGS.xlims...)
    SETTINGS.ylims === nothing || ylims!(ax, SETTINGS.ylims...)
    SETTINGS.zlims === nothing || zlims!(ax, SETTINGS.zlims...)

    # Muted, near-transparent sheet as a reference backdrop; the weight-coloured nodes draw on top.
    mesh!(ax, surface_points, faces; color = (:gray, 0.25), transparency = true)

    node_plot = scatter!(ax, nodes; color = weights, markersize = 6, colorscale = log10)
    Colorbar(fig[1, 2], node_plot; label = L"w")

    display(fig)
end

# Resolve the material from the rule's own header before the plot, so the overlaid Fermi surface
# matches the rule. The material is `include`d at top level (not inside `main`) so its `dispersion`
# method predates `main`'s world age and stays callable from within the mesh build.
const RULE = load_quadrature_rule(joinpath(@__DIR__, "..", "meshes", SETTINGS.quad_file))
include(joinpath(@__DIR__, "..", "materials", basename(RULE.header.material_file)))
main(RULE.nodes, RULE.weights, RULE.header.N_ibz)
