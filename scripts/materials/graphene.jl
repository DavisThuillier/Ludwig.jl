const material::String = "graphene"

const a::Float64 = 2.46e-10  # Lattice constant in m
const t::Float64 = 2.8 # Nearest neighbor hopping in eV
const graphene = Ludwig.Lattices.Lattice([1.0, 0.0], [0.5, 0.5*sqrt(3)])

function Δ(k)
    return exp(-2π * im * k[1]) * (1 + 2 * exp(3π * im * k[1]) * cos(sqrt(3) * π * k[2]))
end

function ε1(k, μ = 0.0)
    return μ - t * sqrt(1 + 4 * cos(pi * k[1])^2 + 4 * cos(pi * k[1]) * cos(pi * sqrt(3) * k[2]) )
end

function ε2(k, μ = 0.0)
    return μ + t * sqrt(1 + 4 * cos(pi * k[1])^2 + 4 * cos(pi * k[1]) * cos(pi * sqrt(3) * k[2]) )
end

function orbital_weights(k)
    u = Δ(k)
    u /= abs(u)

    return (1 / sqrt(2)) * [-u 1.0; u 1.0]
end

"""
    vertex_factor(p1, p2)
"""
vertex_pp(p1::Patch, p2::Patch) = dot(p1.w, p2.w)


function vertex_pk(p::Patch, k, μ)
    u = Δ(k)
    u /= abs(u)

    if μ == 1
        return (- u * p.w[1] + p.w[2]) / sqrt(2)
    else
        return (u * p.w[1] + p.w[2]) / sqrt(2)
    end
end