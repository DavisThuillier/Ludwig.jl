const material::String = "Sr2RuO4"
const c::Float64  = 12.68e-10 # Interlayer distance in m
const a::Float64  = 3.90e-10  # Lattice constant in m
const νxy::Float64 = 0.39 # Poisson's ratio for Sr2RuO4

#########
### γ ###
#########
const tγ::Float64     = 0.07735      # γ band nearest hopping in eV
const tpγ::Float64    = 0.0303212    # eV
const μγ::Float64     = 0.114478     # eV

function ham_γ(k, ϵ = 0.0)
    return - 2.0 * tγ * ((1 - ϵ) * cos(2pi*k[1]) + (1 + νxy * ϵ / 2.0) * cos(2pi*k[2])) - 4 * tpγ * (1 - (1 - νxy) * ϵ / 2.0) * cos(2pi*k[1]) * cos(2pi*k[2]) - μγ
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

exz(k, ϵ = 0.0) = -2.0 * (1 - ϵ) * cos(2pi*k[1]) - 2.0 * t3 * (1 + νxy * ϵ) * cos(2pi*k[2]) - μαβ
eyz(k, ϵ = 0.0) = -2.0 * t3 * (1 - ϵ) * cos(2pi*k[1]) - 2.0 * (1 + νxy * ϵ) * cos(2pi*k[2]) - μαβ
V(k, ϵ = 0.0)   = 4.0 * t5 * (1 - (1 - νxy) * ϵ / 2.0) * sin(2pi*k[1]) * sin(2pi*k[2])

function ham_α(k, ϵ = 0.0)
    x = exz(k, ϵ)
    y = eyz(k, ϵ)
    return 0.5 * ( (x + y) - sqrt( (x - y)^2 + 4 * V(k)^2 ) ) * tα
end

function ham_β(k, ϵ = 0.0)
    x = exz(k, ϵ)
    y = eyz(k, ϵ)
    return 0.5 * ( (x + y) + sqrt( (x - y)^2 + 4 * V(k)^2 ) ) * tβ
end

function orbital_weights(k)
    W = zeros(Float64, 3, 3)

    Δ1 = exz(k) - eyz(k)
    Δ2 = 2 * V(k)
    ζ = sqrt(Δ1^2 + Δ2^2)

    nα = sqrt(Δ2^2 + (Δ1 - ζ)^2) # Norm of the eigenvector corresponding to α band

    W[1,1] = sign(Δ2) * (Δ1 - ζ) / nα 
    W[2,1] = abs(Δ2) / nα 
    W[1,2] = W[2,1]
    W[2,2] = - W[1,1]
    W[3,3] = 1.0

    return W
end

"""
    vertex_factor(p1, p2)

Compute ``F_{k1, k2}`` for patches `p1` and `p2` specifically optimized for Sr2RuO4.
"""
function vertex_pp(p1::Patch, p2::Patch)
    if p1.band_index < 3 # i.e. p1 ∈ {α, β}
        if p2.band_index < 3
            return dot(p1.w, p2.w)
        else
            return 0.0
        end
    else
        if p2.band_index < 3
            return 0.0
        else
            return 1.0 # \gamma \gamma
        end
    end
end

function vertex_pk(p::Patch, k, μ::Int)
    if p.band_index < 3 # i.e. p ∈ {α, β}
        if μ < 3
            Δ1 = (exz(k) - eyz(k)) 
            Δ2 = 2 * V(k)
            ζ = sqrt(Δ1^2 + Δ2^2) 

            w1 = Δ1 - ζ # First element of the α band weight
            if μ == 1 # α band
                return sign(Δ2) * (p.w[1] * w1 + p.w[2] * Δ2) / sqrt(Δ2^2 + w1^2)
            else # β band
                return sign(Δ2) * (p.w[1] * Δ2 - p.w[2] * w1) / sqrt(Δ2^2 + w1^2)
            end
        else
            return 0.0
        end
    else
        if μ < 3
            return 0.0
        else
            return 1.0
        end
    end
end

bands = [ham_α, ham_β, ham_γ]