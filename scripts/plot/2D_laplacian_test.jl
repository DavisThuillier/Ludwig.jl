using Pkg
Pkg.activate(@__DIR__)

using Ludwig
using CairoMakie
using LinearAlgebra
using StaticArrays
using KrylovKit
using LaTeXStrings

include(joinpath(@__DIR__, "..", "materials", "graphene.jl"))

function main(; plot_stars = false)
    point_group = PointGroup(generators, primitives)
    crystal = Crystal(primitives, point_group)
    
    ibz = Ludwig.irreducible_brillouin_zone(crystal)
    ibz_vertices = vertices(ibz)
    ibz_vertices = vcat(ibz_vertices, [first(ibz_vertices)])

    kmax = 4π/3
    @time mesh = level_set(dispersion, -0.2, (0, kmax), (0, kmax), 3; refinement_depth=10,
        mask = ibz, weld = true
    )

    N = 1000
    density(k) = 1 / norm(velocity(k))
    @time ϕ = Ludwig.mesh_harmonics(mesh, density, N)

    @time nodes, weights, residual = Ludwig.isoenergy_quadrature_rule(mesh, velocity, ϕ; ibz)
    @show residual
    @show weights

    for i ∈ 1:N
        moment = integrate_over_mesh(ϕ[:, i] .* density.(mesh.vertices), mesh)
        @show abs(moment)

        fig = Figure()
        ax = Axis(fig[1,1];
            aspect = DataAspect(),
            title = latexstring("i = $(i)")
        )
        scatter!(ax, mesh.vertices ; color = ϕ[:, i], markersize = 2)
        # scatter!(ax, nodes; color = :red)
        display(fig)
    end
end

main()
