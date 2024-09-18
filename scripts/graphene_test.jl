using StaticArrays
using CairoMakie
using Ludwig
using Ludwig.Lattices
using LinearAlgebra

function main()
    a1 = [0.5, sqrt(3) / 2.0]
    a2 = [-0.5, sqrt(3) / 2.0]

    l = Lattices.Lattice(a1, a2)
    rlv = Lattices.reciprocal_lattice_vectors(l)
    T = inv(rlv) * rlv
    bz = Lattices.get_bz(l)
    ibz = Lattices.get_ibz(l)

    G = Lattices.point_group(l)

    basis = Matrix{Float64}(undef, 2, 0)
    for i in eachindex(ibz)
        j = i == length(ibz) ? 1 : i + 1

        if ibz[i] == [0.0, 0.0] || ibz[j] == [0.0, 0.0]
            basis = hcat(basis, ibz[i] - ibz[j])
        end
    end

    mesh = MarchingTriangles.ibz_mesh(l, 80)
    E = map(bands[1], mesh.points)
    energies = collect(-8.0:1.0:0.0)
    Δε = 0.1

    # triangles = MarchingTriangles.get_triangles(mesh, E)

    f = Figure()
    ax = Axis(f[1,1], aspect = 1.0)
    poly!(ax, bz, color = :black)
    poly!(ax, map(x -> SVector{2}(T * x), ibz), color = :gray)
    # scatter!(ax, map(x -> SVector{2}(T * x), mesh.points), color = :black)

    # for i in eachindex(triangles)
    #     lines!(ax, map(x -> SVector{2}(T * mesh.points[x]), vcat(collect(triangles[i]), first(triangles[i]))), color = :black, linewidth = 0.4)
    # end

    c = MarchingTriangles.contours(mesh, E, energies, Δε)
    
    for g in G
        O = Groups.get_matrix_representation(g)

        for bundle in c
            for iso in bundle.isolines
                lines!(ax, map(x -> SVector{2}(O * x), iso.points), color = bundle.level, colorrange = (minimum(E), maximum(E)))
            end
        end
        display(f)
    end

    # arrows!([0.0, 0.0], [0.0, 0.0], rlv[1, :], rlv[2, :], color = [:blue, :red])
    # Colorbar(f[1,2], colorrange = (minimum(E), maximum(E)), colormap = :viridis)
    display(f)
        
end

include(joinpath(@__DIR__, "materials", "graphene.jl"))

main()