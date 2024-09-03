using Ludwig
using CairoMakie
using LinearAlgebra
using StaticArrays

function main()
    a1 = [1.0, 0.0]
    a2 = [0.5, sqrt(3) / 2.0]

    l = Lattices.Lattice(Array(hcat(a2, a1)'))

    @show lattice_type(l)
    rlv = Lattices.reciprocal_lattice_vectors(l)

    T = inv(rlv) * rlv

    bz = map(x -> SVector{2, Float64}(T * x), Lattices.get_bz(l))

    x_range, y_range = Ludwig.get_bounding_box(bz)

    f = Figure()
    ax = Axis(f[1,1], aspect = 1.0)
    xlims!(ax, x_range...)
    ylims!(ax, y_range...)
    poly!(ax, bz)

    N = 101
    xs = LinRange(x_range..., N)
    ys = LinRange(y_range..., N)
    
    grid = []
    for kx in xs
        for ky in ys
            k = SVector{2}([kx, ky])
            # push!(grid, k)
            Lattices.in_polygon(k, bz) && push!(grid, k)
        end 
    end

    scatter!(ax, grid, color = :black, markersize = 2)
    arrows!([0.0, 0.0], [0.0, 0.0], (T * rlv)[1, :], (T * rlv)[2, :], color = [:blue, :red])
    display(f)
end

main()