const material::String = "Sr2RuO4"
const c::Float64  = 12.68e-10 # Interlayer distance in m
const a::Float64  = 3.90e-10  # Lattice constant in m

#########
### γ ###
#########
const tγ::Float64     = 0.07735      # γ band nearest hopping in eV
const tpγ::Float64    = 0.0303212    # eV
const μγ::Float64     = 0.114478     # eV

function ham_γ(k)
    return - 2.0 * tγ * (cos(2pi*k[1]) + cos(2pi*k[2])) - 4*tpγ*cos(2pi*k[1]) * cos(2pi*k[2]) - μγ
end

###############
### α and β ###
###############
const tα::Float64     = 0.099  # α band hopping in eV
const tβ::Float64     = 0.1278 # β band hopping in eV

# In units of respective band nearest neighbor hopping:
const t3::Float64     = 0.08
const t5::Float64     = 0.13
const μαβ::Float64    = 1.08

exz(k) = -2.0 * cos(2pi*k[1]) - 2.0 * t3 * cos(2pi*k[2]) - μαβ
eyz(k) = -2.0 * t3 * cos(2pi*k[1]) - 2.0 * cos(2pi*k[2]) - μαβ
V(k)   = 4.0 * t5 * sin(2pi*k[1]) * sin(2pi*k[2])

function ham_α(k)
    x = exz(k)
    y = eyz(k)
    return 0.5 * ( (x + y) - sqrt( (x - y)^2 + 4 * V(k)^2 ) ) * tα
end

function ham_β(k)
    x = exz(k)
    y = eyz(k)
    return 0.5 * ( (x + y) + sqrt( (x - y)^2 + 4 * V(k)^2 ) ) * tβ
end

bands = [ham_α, ham_β, ham_γ]