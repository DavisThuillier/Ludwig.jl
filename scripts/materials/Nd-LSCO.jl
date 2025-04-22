const material::String = "Nd-LSCO"
# Tight binding parameters from arXiv:2011.13054
const t::Float64 = 0.160 # ± 30 eV
# In units of t
const μ::Float64 = -0.8243 # ± 0.0005 
const tp::Float64 = - 0.1364 # ± 0.0005
const tpp::Float64 = 0.0682 # ± 0.0005
const tz::Float64 = 0.0651 # ± 0.0005
const a::Float64 = 3.75e-10
const c::Float64 = 13.2e-10

function ε_full(k::AbstractVector)
    return - μ - 2.0 * (cos(2pi*k[1]) + cos(2pi*k[2])) - 4 * tp * cos(2pi*k[1])*cos(2pi*k[2]) - 2 * tpp * (cos(4pi * k[1]) + cos(4pi * k[2])) - 2 * tz * cos(pi*k[1]) * cos(pi*k[2]) * cos(pi*k[3]) * (cos(2pi*k[1]) - cos(2pi*k[2]))^2
end

## k_z = 0 slice
function ε(k::AbstractVector)
    return - μ - 2.0 * (cos(2pi*k[1]) + cos(2pi*k[2])) - 4 * tp * cos(2pi*k[1])*cos(2pi*k[2]) - 2 * tpp * (cos(4pi * k[1]) + cos(4pi * k[2])) - 2 * tz * cos(pi*k[1]) * cos(pi*k[2]) * (cos(2pi*k[1]) - cos(2pi*k[2]))^2
end

bands = [ε]
