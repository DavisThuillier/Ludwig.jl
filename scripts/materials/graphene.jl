const material::String = "graphene"

const a::Float64 = 2.46e-10  # Lattice constant in m
const t::Float64 = 2.8 # Nearest neighbor hopping in eV
const μ::Float64 = 0.0

function ε1(k)
    return μ - t * sqrt(1 + 4 * cos(pi * k[1])^2 + 4 * cos(pi * k[1]) * cos(pi * sqrt(3) * k[2]) )
end

function ε2(k)
    return μ + t * sqrt(1 + 4 * ((cos(k[1]/2.0))^2 + cos(k[1]/2.0) * cos(sqrt(3) * k[2] / 2.0) ))
end

orbital_weights(k) = (1 / sqrt(2)) * [1.0 1.0; 1.0 -1.0]

bands = [ε1, ε2]