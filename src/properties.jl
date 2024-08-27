"""
    inner_product(a, b, L, w)

Compute the weighted inner product ``\\langle a | L^(-1) | b\\rangle`` with the weight vector `w`.
"""
function inner_product(a, b, L, w)
    ϕ = bicgstabl(L, b)
    prod = 0.0
    for i in eachindex(a)
        prod += a[i] * w[i] * ϕ[i]
    end
    return prod
end

"""
    conductivity(L, v, E, dV, T)
"""
function conductivity(L, v, E, dV, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    weight = fd .* (1 .- fd) .* dV

    σ = Matrix{ComplexF64}(undef, 2, 2)

    σ[1,1] = inner_product(first.(v), first.(v), L, weight)
    σ[1,2] = inner_product(first.(v), last.(v), L, weight)
    σ[2,1] = inner_product(last.(v), first.(v), L, weight)
    σ[2,2] = inner_product(last.(v), last.(v), L, weight)

    return (G0 / (2π)) * (σ / T) 
end

function longitudinal_conductivity(L, vx, E, dV, T)
    fd = f0.(E, T) # Fermi dirac on grid points

    σxx = inner_product(vx, vx, L, fd .* (1 .- fd) .* dV)

    return (G0 / (2π)) * (σxx / T) 
end

"""
    ηB1g(L, E, dVs, Dxx, Dyy, T)
"""
function ηB1g(L, E, dV, Dxx, Dyy, T)
    fd = f0.(E, T) # Fermi dirac on grid points

    prefactor = 2 * hbar * e_charge / T

    η = prefactor * 0.25 * inner_product(Dxx .- Dyy, Dxx .- Dyy, L, fd .* (1 .- fd) .* dV)

    return η
end

function ηB2g(L, E, dV, Dxy, T)
    fd = f0.(E, T) # Fermi dirac on grid points

    prefactor = 2 * hbar * e_charge / T

    η = prefactor * inner_product(Dxy, Dxy, L, fd .* (1 .- fd) .* dV)

    return η
end

"""
    σ_lifetime(L, v, E, dV, T)
"""
function σ_lifetime(L, v, E, dV, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd)
    vx = first.(v)

    norm = 0.0
    for i in eachindex(vx)
        norm += w[i] * dV[i] * vx[i]^2
    end
    σ = inner_product(vx, vx, L, w .* dV)

    τ_eff = real( σ / norm)
    
    return τ_eff * hbar
end

function η_lifetime(L, Dxx, Dyy, E, dV, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd)

    D = Dxx .- Dyy

    norm = 0.0
    for i in eachindex(D)
        norm += w[i] * dV[i] * D[i]^2
    end
    η = inner_product(D, D, L, w .* dV)

    τ_eff = real( η / norm)
    
    return τ_eff * hbar
end

function spectrum(Γ, E, dVs, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd) # Energy derivative of FD on grid points

    # M = diagm(sqrt.(w .* dVs)) * Γ * (diagm(1 ./ sqrt.(w .* dVs)))

    eigenvalues  = eigvals(M)

    return eigenvalues
end