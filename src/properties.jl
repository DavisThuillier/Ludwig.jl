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

    return (G0 / π) * (σ / T) 
end