"""
    conductivity(L, v, E, dV, T)
"""
function conductivity(L, v, E, dV, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd) # Energy derivative of FD on grid points

    σ = Matrix{ComplexF64}(undef, 2, 2)

    ϕx = bicgstabl(L, first.(v))
    ϕy = bicgstabl(L, last.(v))

    σ[1,1] = dot(first.(v) .* w .* dV, ϕx)
    σ[1,2] = dot(first.(v) .* w .* dV, ϕy)
    σ[2,1] = dot(last.(v) .* w .* dV, ϕx)
    σ[2,2] = dot(last.(v) .* w .* dV, ϕy)

    return (G0 / (2π)) * (σ / T) 
end

"""
    ηB1g(L, E, dVs, Dxx, Dyy, T)
"""
function ηB1g(L, E, dV, Dxx, Dyy, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd) # Energy derivative of FD on grid points

    Winv = diagm(1 ./ (w .* dV)) 
    G = L * Winv # G is a symmetric matrix
    geigvecs = eigvecs(G)
    τ = 1 ./ eigvals(G) # Lifetimes of modes
    τ[1] = 0.0 # Enforce that overlap with the particle-conserving mode is null

    ϕxx = Vector{ComplexF64}(undef, length(τ))
    ϕyy = Vector{ComplexF64}(undef, length(τ))
    for i in eachindex(τ)
        ϕxx[i] = dot(Dxx, geigvecs[:, i])
        ϕyy[i] = dot(Dyy, geigvecs[:, i])
    end

    ηxxxx = dot(ϕxx, ϕxx .* τ)
    ηxxyy = dot(ϕxx, ϕyy .* τ)
    η = (ηxxxx - ηxxyy)/2

    prefactor = 2 * hbar * e_charge / T

    return real(η) * prefactor
end

"""
    σ_lifetime(Γ, v, E, dV, T)
"""
function σ_lifetime(Γ, v, E, dV, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd) # Energy derivative of FD on grid points
    vx = first.(v)

    norm = 0.0
    for i in eachindex(vx)
        norm += w[i] * dV[i] * vx[i]^2
    end
    ϕx = bicgstabl(Γ, vx)
    σ = dot(vx .* w .* dVs, ϕx)

    τ_eff = real( σ / norm)
    
    return τ_eff
end

function spectrum(Γ, E, dVs, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd) # Energy derivative of FD on grid points

    M = diagm(sqrt.(w .* dVs)) * Γ * (diagm(1 ./ sqrt.(w .* dVs)))

    eigenvalues  = eigvals(M)

    return eigenvalues
end