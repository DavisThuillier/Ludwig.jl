using ForwardDiff

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

# Computes the xx or yy band structure deformation potential for the α band. μ is a band index where 1 is α, 2 is β, and 3 in γ
function dbs_ii_μ(k, i::Int, μ::Int, δ = 0.0)
    if μ == 1 || μ == 2
        (μ == 1) ? (δ /= tα) : (δ /= tβ)
        x = exz(k)
        y = eyz(k)
        Δ = sqrt( 0.25 * (x - y)^2 + V(k)^2 ) # Twice the band gap between α and β
        
        d = alph * (1 + t3) * cos(2pi*k[i]) + (-1)^(μ+1) * ( (-1)^(i + 1) * (1 - t3) * ( alph * (1 - t3) * cos(2pi*k[i]) - δ / 2.0) * (cos(2pi*k[1]) - cos(2pi*k[2])) + 8 * t5^2 * alph_p * (sin(2pi*k[1]) * sin(2pi*k[2]))^2) / Δ

        (μ == 1) ? (return d*tα) : return (d*tβ) 
    elseif μ == 3 # γ band
        return 2 * alph * tγ * cos(2pi*k[i]) + 2 * alph_p * tpγ * cos(2pi*k[1]) * cos(2pi*k[2])
    else
        return 0 # Invalid band index
    end
end

function dep_ii_μ(k, i::Int, μ::Int, δ = 0.0)
    dbs = dbs_ii_μ(k, i, μ, δ)
    v = ForwardDiff.gradient(bands[μ], k)
    dep = dbs + Ludwig.e_mass * (a/Ludwig.hc)^2 * v[i]^2
    return dep
end

function dbs_xy_μ(k, μ::Int)
    if μ == 1 || μ == 2
        x = exz(k)
        y = eyz(k)
        Δ = sqrt( 0.25 * (x - y)^2 + V(k)^2 ) # Twice the band gap between α and β

        d = (-1)^(μ) * t5^2 * alph_p * cos(2pi*k[1]) * cos(2pi*k[2]) * sin(2pi*k[1]) * sin(2pi*k[2]) / Δ

        (μ == 1) ? (return d*tα) : return (d*tβ) 
    elseif μ == 3 # γ band
        return - 4 * alph_p * tpγ * sin(2pi*k[1]) * sin(2pi*k[2])
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