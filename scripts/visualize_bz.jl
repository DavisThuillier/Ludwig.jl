using Ludwig
using CairoMakie
using LinearAlgebra

function main()
    a1 = [1.0, 0.0]
    # a2 = [-0.5, 1.0]
    a2 = [0.5, sqrt(3) / 2.0]

    l = Lattices.Lattice(Array(hcat(a2, a1)'))
    bz = Lattices.get_bz(l)
    rlv = Lattices.reciprocal_lattice_vectors(l)

    f = Figure()
    ax = Axis(f[1,1], aspect = 1.0)
    poly!(ax, bz)
    
    k = [-0.51, -0.51]
    kᵩ = Lattices.map_to_bz(k, rlv)

    scatter!(ax, k[1], k[2])
    scatter!(ax, kᵩ[1], kᵩ[2])

    arrows!([0.0, 0.0], [0.0, 0.0], rlv[1, :], rlv[2, :], color = [:blue, :red])
    display(f)
end

main()