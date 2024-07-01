function conductivity(Γ, v, E, dV, T::Real)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd) # Energy derivative of FD on grid points

    σ = Matrix{ComplexF64}(undef, 2, 2)

    ϕx = bicgstabl(Γ, first.(v))
    ϕy = bicgstabl(Γ, last.(v))

    σ[1,1] = dot(first.(v) .* w .* dV, ϕx)
    σ[1,2] = dot(first.(v) .* w .* dV, ϕy)
    σ[2,1] = dot(last.(v) .* w .* dV, ϕx)
    σ[2,2] = dot(last.(v) .* w .* dV, ϕy)

    return (G0 / (2π)) * (σ / T) 
end

# function viscosity(Γ, E, dVs, Dxx, Dyy, Dxy, T::Real)
#     fd = f0.(E, T) # Fermi dirac on grid points
#     w = fd .* (1 .- fd) # Energy derivative of FD on grid points

#     Winv = diagm(1 ./ (w .* dVs)) 
#     G = Γ * Winv # G is a symmetric matrix
#     geigvecs = eigvecs(G)
#     τ = 1 ./ eigvals(G) # Lifetimes of modes
#     τ[1] = 0.0 # Enforce that overlap with the particle-conserving mode is null

#     ϕxx = Vector{ComplexF64}(undef, length(τ))
#     ϕyy = Vector{ComplexF64}(undef, length(τ))
#     for i in eachindex(τ)
#         ϕxx[i] = dot(Dxx, geigvecs[:, i])
#         ϕyy[i] = dot(Dyy, geigvecs[:, i])
#     end

#     ηxxxx = dot(ϕxx, ϕxx .* τ)
#     ηxxyy = dot(ϕxx, ϕyy .* τ)
#     ηB1g = (ηxxxx - ηxxyy)/2

#     ϕxy = bicgstabl(G, Dxy)
#     ηxyxy = dot(Dxy, ϕxy)
#     ηB2g = ηxyxy

#     prefactor = 2 * hbar * e_charge / T

#     ηB1g *= prefactor
#     ηB2g *= prefactor

#     return real.(ηB1g), real.(ηB2g)
# end

function viscosity(Γ, E, dV, Dxx, Dyy, Dxy, T::Real)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd) # Energy derivative of FD on grid points

    ℓ = size(Γ)[1]
    ξ0 = ones(Float64, ℓ) / sqrt(ℓ) # Particle conserving mode

    # Dxx -= dot(Dxx, ξ0) * ξ0
    # Dyy -= dot(Dyy, ξ0) * ξ0
    # Dxy -= dot(Dxy, ξ0) * ξ0

    ϕxx, _ = bicgstabl(Γ, Dxx, 10, log = true)
    ϕyy = bicgstabl(Γ, Dyy)
    ϕxy = bicgstabl(Γ, Dxy)

    # Viscosities as inner product
    ηxxxx = dot(Dxx .* w .* dV, ϕxx)
    ηxxyy = dot(Dxx .* w .* dV, ϕyy)
    ηxyxy = dot(Dxy .* w .* dV, ϕxy)

    @show ηxxxx

    ηB1g = (ηxxxx - ηxxyy)/2
    ηB2g = ηxyxy

    prefactor = 2 * hbar * e_charge / T

    ηB1g *= prefactor
    ηB2g *= prefactor

    return real.(ηB1g), real.(ηB2g)
end

function σ_lifetime(Γ, v, E, dVs, T::Real)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd) # Energy derivative of FD on grid points
    vx = first.(v)

    norm = 0.0
    for i in eachindex(vx)
        norm += w[i] * dVs[i] * vx[i]^2
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