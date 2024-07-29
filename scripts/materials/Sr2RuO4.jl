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

const alph::Float64 = 7.604
const alph_p::Float64 = 7.604

function dxx_γ(k)
    return 2 * alph * tγ * cos(2pi*k[1]) + 2 * alph_p * tpγ * cos(2pi*k[1]) * cos(2pi*k[2])
end

function dyy_γ(k)
    return 2 * alph * tγ * cos(2pi*k[2]) + 2 * alph_p * tpγ * cos(2pi*k[1]) * cos(2pi*k[2]) 
end

function dxy_γ(k)
    return - 2 * alph_p * tpγ * sin(2pi*k[1]) * sin(2pi*k[2])
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

# Computes the xx or yy deformation potential for the α band. μ is a band index where 1 is α and 2 is β
function dii_μ(k, i::Int, μ::Int)
    if μ == 1 || μ == 2
        x = exz(k)
        y = eyz(k)
        Δ = sqrt( 0.25 * (x - y)^2 + V(k)^2 ) # Twice the band gap between α and β

        d = alph * (1 + t3) * cos(2pi*k[i]) + (-1)^(μ+i) * (alph * (1 - t3)^2 * cos(2pi*k[i]) * (cos(2pi*k[1]) - cos(2pi*k[2])) ) / Δ
        (μ == 1) ? (return d*tα) : return (d*tβ) 
    elseif μ == 3 # γ band
        return 2 * alph * tγ * cos(2pi*k[i]) + 2 * alph_p * tpγ * cos(2pi*k[1]) * cos(2pi*k[2])
    else
        return 0 # Invalid band index
    end
end

function hamiltonian(k)
    h = zeros(Float64, length(bands), length(bands))
    h[1,1] = exz(k)
    h[1,2] = V(k)
    h[2,2] = eyz(k)
    h[2,1] = h[1,2]
    h[1:2,1:2] *= sqrt(tα*tβ)
    h[3,3] = ham_γ(k)
    return h
end

# Closed form eigenvectors for Sr2RuO4
function eigenvecs!(W::MMatrix{3, 3, Float64}, k)
    W[1,2] = (exz(k) - eyz(k)) / (2 * V(k)) # Store in cell of W which will be set to 0 later
    W[2,2] = sqrt(1.0 + W[1,2]^2) # Eigenvectors for α,β are of the form (W[1,1] ∓ W[1,2], 1)
    W[3,1] = sqrt(1.0 + (W[1,2] - W[2,2])^2) # Norm of first eigenvector
    W[3,3] = sqrt(1.0 + (W[1,2] + W[2,2])^2) # Norm of first eigenvector

    W[1,1] = (W[1,2] - W[2,2]) / W[3,1]; W[2,1] = 1.0 / W[3,1]; W[3,1] = 0.0
    W[1,3] = (W[1,2] + W[2,2]) / W[3,3]; W[2,3] = 1.0 / W[3,3]; W[3,3] = 0.0
    W[1,2] = 0.0; W[2,2] = 0.0; W[3,2] = 1.0
    return nothing
end

bands = [ham_α, ham_β, ham_γ]