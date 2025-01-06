const material::String = "graphene"

const a::Float64 = 2.46e-10  # Lattice constant in m
const t::Float64 = 2.8 # Nearest neighbor hopping in eV

function ε1(k, μ = 0.0)
    return μ - t * sqrt(1 + 4 * cos(pi * k[1])^2 + 4 * cos(pi * k[1]) * cos(pi * sqrt(3) * k[2]) )
end

function ε2(k, μ = 0.0)
    return μ + t * sqrt(1 + 4 * cos(pi * k[1])^2 + 4 * cos(pi * k[1]) * cos(pi * sqrt(3) * k[2]) )
end

orbital_weights(k) = (1 / sqrt(2)) * [1.0 1.0; 1.0 -1.0]

bands = [ε1, ε2]