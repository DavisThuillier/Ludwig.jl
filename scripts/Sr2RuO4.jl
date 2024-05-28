

const c::Float64  = 12.68e-10 # Interlayer distance in m
const a::Float64  = 3.90e-10  # Lattice constant in m
const tα::Float64 = 0.165 * 0.60 # α band hopping in eV
const tβ::Float64 = 0.165 * 0.78 # β band hopping in eV
const tγ::Float64 = 0.119 * 0.65 # γ band hopping in eV
const tpγ::Float64 = 0.392
const t3::Float64 = 0.08
const t5::Float64 = 0.13
const μαβ::Float64 = 1.08
const μγ::Float64  = 1.48

#########
### γ ###
#########

function ham_γ(k)
    return - 2.0 * cos(2pi*k[1]) - 2.0 * cos(2pi * k[2]) - 4 * tpγ * cos(2pi * k[1]) * cos(2pi * k[2]) - μγ
end

exz(k) = -2.0 * cos(2pi*k[1]) - 2.0 * t3 * cos(2pi*k[2]) - μαβ
eyz(k) = -2.0 * t3 * cos(2pi*k[1]) - 2.0 * cos(2pi*k[2]) - μαβ
V(k)   = 4.0 * t5 * sin(2pi*k[1]) * sin(2pi*k[2])

function ham_α(k)
    x = exz(k)
    y = eyz(k)
    return 0.5 * ( (x + y) - sqrt( (x - y)^2 + 4 * V^2 ) )
end

function ham_β(k)
    x = exz(k)
    y = eyz(k)
    return 0.5 * ( (x + y) + sqrt( (x - y)^2 + 4 * V^2 ) )
end

band_dict = Dict("gamma" => ("γ", ham_γ, tγ), "alpha" => ("α", ham_α, tα), "beta" => ("β", ham_β, tβ))

function main()

end

main()