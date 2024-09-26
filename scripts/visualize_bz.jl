using Ludwig
using CairoMakie
using LinearAlgebra
using StaticArrays
using DelaunayTriangulation

function main()
    a1 = [1.0, 0.0]
    a2 = [0.5, sqrt(3) / 2.0]

    l = Lattices.Lattice(Array(hcat(a2, a1)'))

    @show Lattices.lattice_type(l)
    rlv = Lattices.reciprocal_lattice_vectors(l)

    bz = map(x -> SVector{2, Float64}(x), Lattices.get_bz(l))

    x_range, y_range = Ludwig.get_bounding_box(bz)

    f = Figure()
    ax = Axis(f[1,1], aspect = 1.0)
    xlims!(ax, x_range...)
    ylims!(ax, y_range...)
    poly!(ax, bz)

    N = 17
    xs = LinRange(x_range..., N)
    ys = LinRange(y_range..., N)
    
    grid = SVector{2,Float64}[]
    for kx in xs
        for ky in ys
            k = SVector{2}([kx, ky])
            Lattices.in_polygon(k, bz) && push!(grid, k)
        end 
    end
    scatter!(ax, grid, color = map(x -> bands[2](x), grid), markersize = 1)

    E = map(x -> bands[2]([x[1], x[2]]), collect(Iterators.product(xs, ys)))
    
    α = 6.0
    T = 400
    e_threshold = α * kb * T # Half-width of Fermi tube
    @show e_threshold
    e_min = max(-e_threshold, 0.999 * minimum(E))
    e_max = min(e_threshold, 0.999 * maximum(E))
    energies = collect(LinRange(e_min, e_max, 11))

    c = Ludwig.contours(xs, ys, E, energies)
    for i in eachindex(c)
        for iso in c[i].isolines
            lines!(ax, filter(k -> Lattices.in_polygon(k, bz), iso.points), color = energies[i], colorrange = (e_min, e_max))
        end
    end
    Colorbar(f[1,2], colorrange = (e_min, e_max))

    arrows!([0.0, 0.0], [0.0, 0.0], rlv[1, :], rlv[2, :], color = [:blue, :red])
    display(f)
end

include(joinpath(@__DIR__, "materials", "graphene.jl"))
main()