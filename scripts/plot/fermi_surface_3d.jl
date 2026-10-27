using Pkg
Pkg.activate(@__DIR__)

using Ludwig
using GLMakie
using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "local_settings.jl"))
const SETTINGS = LOCAL_SETTINGS.fermi_surface_3d
include(joinpath(@__DIR__, "..", "materials", SETTINGS.material_file))


function main()
    point_group = PointGroup(generators, primitives)
    crystal = Crystal(primitives, point_group)
    
    ibz = Ludwig.irreducible_brillouin_zone(crystal)
    ibz_vertices = vertices(ibz)
    ibz_vertices = vcat(ibz_vertices, [first(ibz_vertices)])

    @time mesh = level_set(dispersion, 0.0, (0, π), (0, π), (0, π), 2; refinement_depth=6)

    mesh_vertices = vertices(mesh)
    surface_points = [Point3f(v[1], v[2], v[3]) for v in mesh_vertices]
    faces = [simplices(mesh)[i][j] for i in eachindex(simplices(mesh)), j in 1:3]
    @show length(mesh_vertices)

    # return nothing

    fig = Figure(size = (700, 700))
    ax = Axis3(fig[1, 1];
        aspect = :data,
        xlabel = L"k_x",
        ylabel = L"k_y",
        zlabel = L"k_z",
        title = "Nearest-neighbour tight-binding Fermi surface",
    )
    mesh!(ax, surface_points, faces; color = (:orange, 0.5), transparency = true)

    display(fig)
    path = joinpath(@__DIR__, "..", "plots", SETTINGS.output)
    save(path, fig)
    return nothing
end

main()
