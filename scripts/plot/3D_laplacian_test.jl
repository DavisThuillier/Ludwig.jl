using Pkg
Pkg.activate(@__DIR__)

using Ludwig
using GLMakie
using LinearAlgebra
using StaticArrays
using LaTeXStrings

include(joinpath(@__DIR__, "..", "materials", "simple_cubic.jl"))

function main()
    point_group = PointGroup(generators, primitives)
    crystal = Crystal(primitives, point_group)

    ibz = Ludwig.irreducible_brillouin_zone(crystal)

    kmax = π
    buffer = Array{Float64,3}(undef, 129, 129, 129)
    @time mesh = level_set(dispersion, 0.0, (0, kmax), (0, kmax), (0, kmax), 7; refinement_depth=17,
        samples = buffer, mask = ibz, weld = true
    )

    mesh_vertices = vertices(mesh)
    faces = [simplices(mesh)[i][j] for i in eachindex(simplices(mesh)), j in 1:3]
    surface_points = [Point3f(v[1], v[2], v[3]) for v in mesh_vertices]

    N = 20
    density(k) = 1 / norm(velocity(k))
    ϕ = Ludwig.mesh_harmonics(mesh, density, N)

    for i ∈ 1:N
        moment = integrate_over_mesh(ϕ[:, i] .* density.(mesh.vertices), mesh)
        @show abs(moment)

        fig = Figure(size = (700, 700))
        ax = Axis3(fig[1, 1];
            aspect = :data,
            xlabel = L"k_x",
            ylabel = L"k_y",
            zlabel = L"k_z",
            title = latexstring("i = $(i)"),
        )
        mesh!(ax, surface_points, faces; color = ϕ[:, i])
        display(fig)
    end
end

main()
