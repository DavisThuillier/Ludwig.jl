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

function viscosity(Γ, k, E, dVs, dxx::Function, dyy::Function, dxy::Function, T::Real)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd) # Energy derivative of FD on grid points

    Winv = diagm(1 ./ (w .* dVs)) 
    G = Γ * Winv # G is a symmetric matrix
    geigvecs = eigvecs(G)
    τ = 1 ./ eigvals(G) # Lifetimes of modes
    τ[1] = 0.0 # Enforce that overlap with the particle-conserving mode is null
    
    Dxx = dxx.(k)
    Dyy = dyy.(k)
    Dxy = dxy.(k)


    ϕxx = Vector{ComplexF64}(undef, length(τ))
    ϕyy = Vector{ComplexF64}(undef, length(τ))
    for i in eachindex(τ)
        ϕxx[i] = dot(Dxx, geigvecs[:, i])
        ϕyy[i] = dot(Dyy, geigvecs[:, i])
    end
    
    ηxxxx = dot(ϕxx, ϕxx .* τ)
    ηxxyy = dot(ϕxx, ϕyy .* τ)
    ηB1g = (ηxxxx - ηxxyy)/2

    ϕxy = bicgstabl(G, Dxy)
    ηxyxy = dot(Dxy, ϕxy)
    ηB2g = ηxyxy

    prefactor = 8 * π^2 * hbar / T

    ηB1g *= prefactor
    ηB2g *= prefactor

    return real.(ηB1g), real.(ηB2g)
end