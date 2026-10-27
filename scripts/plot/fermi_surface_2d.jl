using Pkg
Pkg.activate(@__DIR__)

using Ludwig: level_set, merge_vertices, vertices, simplices, integrate_over_mesh, ConvexPolytope
using CairoMakie
using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "local_settings.jl"))
const SETTINGS = LOCAL_SETTINGS.fermi_surface_2d

function main()
    dispersion(k) = -2 * (cos(k[1]) + cos(k[2])) -1.0 # square-lattice NN tight binding
    T = SETTINGS.T
    fermi_dirac(k) = begin
        reduced_energy = dispersion(k) / T
        return 1 / (exp(reduced_energy) + 1)
    end

    initial_depth = 6
    for refinement_depth ∈ 6:2:24
        mesh = level_set(dispersion, 0.0, (0, π), (0, π), initial_depth; refinement_depth)

        mesh_vertices = vertices(mesh)
        # segment_points = [Point2f(mesh_vertices[i][1], mesh_vertices[i][2])
        #                   for simplex in simplices(mesh) for i in simplex]
        # @show length(mesh_vertices)

        # The occupation is 1/2 everywhere on the ε = 0 contour, so this integral converges to half the
        # contour's arc length; it tracks how the marching mesh converges with refinement depth.
        fermi_contour_integral = integrate_over_mesh(k -> fermi_dirac(k), mesh; nchunks = 5, threaded = true)
        @show fermi_contour_integral
    end

    # return nothing

    ibz = ConvexPolytope(SVector{2,Float64}[(0.0, 0.0), (π, 0.0), (π, π)])

    @time mesh = level_set(dispersion, 0.0, (0, π), (0, π), 6; refinement_depth=17)
    _ = level_set(dispersion, 0.0, (0, π), (0, π), 6; refinement_depth=17, mask=ibz) #warmup
    @time mesh = level_set(dispersion, 0.0, (0, π), (0, π), 6; refinement_depth=17, mask=ibz)

    mesh_vertices = vertices(mesh)
    segment_points = [Point2f(mesh_vertices[i][1], mesh_vertices[i][2])
                      for simplex in simplices(mesh) for i in simplex]
    @show length(mesh_vertices)
    # junctions = vertices(merge_vertices(mesh; atol = 1e-9))
    # junction_points = [Point2f(v[1], v[2]) for v in junctions]

    fig = Figure(size = (500, 500))
    ax = Axis(fig[1, 1];
        aspect = DataAspect(), 
        xlabel = L"k_x", 
        ylabel = L"k_y",
        title = "Nearest-neighbour tight-binding Fermi surface",
        xgridvisible = false,
        ygridvisible = false
    )
    linesegments!(ax, segment_points; color = :green)
    # scatter!(ax, junction_points; color = :black, markersize = 2)

    SETTINGS.xlims === nothing || xlims!(ax, SETTINGS.xlims...)
    SETTINGS.ylims === nothing || ylims!(ax, SETTINGS.ylims...)

    display(fig)
    # path = joinpath(@__DIR__, "..", "plots", "fermi_surface_2d.png")
    # save(path, fig)
    return nothing
end

main()
