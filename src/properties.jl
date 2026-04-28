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
    electrical_conductivity(L, v, E, dV, T [, ω = 0.0, q = [0.0, 0.0]])

Compute the conductivity tensor using ``\\sigma_{ij}(ω, \\mathbf{q}) = 2 e^2 \\langle v_i | (L - i\\omega + i\\mathbf{q}\\cdot\\mathbf{v})^{-1} | v_j \\rangle``.

For correct conversion to SI units, `E` and `T` must be expressed in units of eV, `dV` must be in units of ``(1/a)^2``, where ``a`` is the lattice constant, and `v` must be in units of ``(a/\\hbar) \\,\\mathrm{eV}``.
"""
function electrical_conductivity(L, v, E, dV, T, ω = 0.0, q = [0.0, 0.0])
    fd = f0.(E, T) # Fermi dirac on grid points
    weight = fd .* (1 .- fd) .* dV

    σ = Matrix{ComplexF64}(undef, 2, 2)

    if q != [0.0, 0.0]
        if ω == 0.0
            L′ = L - im * diagm(dot.(Ref(q), v))
        else
            L′ = L - im*ω*I + im * diagm(dot.(Ref(q), v))
        end
    else
        if ω != 0.0
            L′ = L - im*ω*I
        else
            L′ = L
        end
    end

    σ[1,1] = inner_product(first.(v), first.(v), L′, weight)
    σ[1,2] = inner_product(first.(v), last.(v), L′, weight)
    σ[2,1] = inner_product(last.(v), first.(v), L′, weight)
    σ[2,2] = inner_product(last.(v), last.(v), L′, weight)

    return (G0 / (2π)) * (σ / T)
end

"""
    longitudinal_electrical_conductivity(L, vx, E, dV, T [, ω])

Compute ``\\sigma_{xx}`` only.
"""
function longitudinal_electrical_conductivity(L, vx, E, dV, T, ω = 0.0)
    fd = f0.(E, T) # Fermi dirac on grid points

    if ω != 0.0
        L′ = L - im*ω*I
    else
        L′ = L
    end

    σxx = inner_product(vx, vx, L, fd .* (1 .- fd) .* dV)

    return (G0 / (2π)) * (σxx / T)
end

"""
    thermal_conductivity(L, v, E, dV, T)

Compute the thermal conductivity tensor using
``\\kappa_{ij} = \\langle \\varepsilon v_i | L^{-1} | \\varepsilon v_j \\rangle``,
where the inner product is weighted by ``f^{(0)}(1 - f^{(0)}) \\Delta V``.

Returns a 2×2 matrix. Unit conversion must be applied by the caller.
"""
function thermal_conductivity(L, v, E, dV, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    weight = fd .* (1 .- fd) .* dV

    jex = E .* first.(v) # Energy current in x-direction
    jey = E .* last.(v)  # Energy current in y-direction

    λ = Matrix{ComplexF64}(undef, 2, 2)
    λ[1,1] = inner_product(jex, jex, L, weight)
    λ[1,2] = inner_product(jex, jey, L, weight)
    λ[2,1] = inner_product(jey, jex, L, weight)
    λ[2,2] = inner_product(jey, jey, L, weight)

    return λ
end

"""
    thermal_conductivity(L, v, E, dV, T, i::Int, j::Int)

Compute the single component ``\\kappa_{ij}`` of the thermal conductivity tensor.
`i` and `j` must be 1 or 2.
"""
function thermal_conductivity(L, v, E, dV, T, i::Int, j::Int)
    if !((0 < i < 3) && (0 < j < 3)) 
        throw(BoundsError((i,j), " indices of tensor must be 1 or 2."))
    end
    fd = f0.(E, T) # Fermi dirac on grid points
    weight = fd .* (1 .- fd) .* dV

    j1 = E .* map(x -> x[i], v)
    j2 = E .* map(x -> x[j], v)

    λ = inner_product(j1, j2, L, weight)
    return λ
end

"""
    thermoelectric_conductivity(L, v, E, dV, T)

Compute the thermoelectric conductivity tensor using
``\\epsilon_{ij} = \\langle v_i | L^{-1} | \\varepsilon v_j \\rangle``,
where the inner product is weighted by ``f^{(0)}(1 - f^{(0)}) \\Delta V``.

Returns a 2×2 matrix. Unit conversion must be applied by the caller.
"""
function thermoelectric_conductivity(L, v, E, dV, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    weight = fd .* (1 .- fd) .* dV

    jx  = first.(v)
    jy  = last.(v)
    jex = E .* jx # Energy current in x-direction
    jey = E .* jy  # Energy current in y-direction

    ϵ = Matrix{ComplexF64}(undef, 2, 2)
    ϵ[1,1] = inner_product(jx, jex, L, weight)
    ϵ[1,2] = inner_product(jx, jey, L, weight)
    ϵ[2,1] = inner_product(jy, jex, L, weight)
    ϵ[2,2] = inner_product(jy, jey, L, weight)

    return ϵ
end

"""
    thermoelectric_conductivity(L, v, E, dV, T, i::Int, j::Int)

Compute the single component ``\\epsilon_{ij}`` of the thermoelectric conductivity tensor.
`i` and `j` must be 1 or 2.
"""
function thermoelectric_conductivity(L, v, E, dV, T, i::Int, j::Int)
    if !((0 < i < 3) && (0 < j < 3)) 
        throw(BoundsError((i,j), " indices of tensor must be 1 or 2."))
    end
    fd = f0.(E, T) # Fermi dirac on grid points
    weight = fd .* (1 .- fd) .* dV

    j1 = map(x -> x[i], v)
    je2 = E .* map(x -> x[j], v)
    ϵ = inner_product(j1, je2, L, weight)

    return ϵ
end

"""
    peltier_tensor(L, v, E, dV, T)

Compute the Peltier tensor using
``\\tau_{ij} = \\langle \\varepsilon v_i | L^{-1} | v_j \\rangle``,
where the inner product is weighted by ``f^{(0)}(1 - f^{(0)}) \\Delta V``.

Returns a 2×2 matrix. Unit conversion must be applied by the caller.
"""
function peltier_tensor(L, v, E, dV, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    weight = fd .* (1 .- fd) .* dV

    jx  = first.(v)
    jy  = last.(v)
    jex = E .* jx # Energy current in x-direction
    jey = E .* jy  # Energy current in y-direction

    τ = Matrix{ComplexF64}(undef, 2, 2)
    τ[1,1] = inner_product(jex, jx, L, weight)
    τ[1,2] = inner_product(jex, jy, L, weight)
    τ[2,1] = inner_product(jey, jx, L, weight)
    τ[2,2] = inner_product(jey, jy, L, weight)

    return τ
end

"""
    peltier_tensor(L, v, E, dV, T, i::Int, j::Int)

Compute the single component ``\\tau_{ij}`` of the Peltier tensor.
`i` and `j` must be 1 or 2.
"""
function peltier_tensor(L, v, E, dV, T, i::Int, j::Int)
    if !((0 < i < 3) && (0 < j < 3)) 
        throw(BoundsError((i,j), " indices of tensor must be 1 or 2."))
    end
    fd = f0.(E, T) # Fermi dirac on grid points
    weight = fd .* (1 .- fd) .* dV

    je1 = E .* map(x -> x[i], v)
    j2  = map(x -> x[j], v)
    τ  = inner_product(je1, j2, L, weight)

    return τ
end

"""
    ηB1g(L, E, dVs, Dxx, Dyy, T)

Compute the B1g viscosity from the deformation potentials `Dxx` and `Dyy`.

For correct conversion to SI units, `E`, `Dxx`, `Dyy`, and `T` must be expressed in units of eV and `dV` must be in units of ``(1/a)^2``, where ``a`` is the lattice constant.
"""
function ηB1g(L, E, dV, Dxx, Dyy, T)
    fd = f0.(E, T) # Fermi dirac on grid points

    prefactor = 2 * hbar * e_charge / T # hbar * e_charge converts hbar to units of J.s

    η = prefactor * 0.25 * inner_product(Dxx .- Dyy, Dxx .- Dyy, L, fd .* (1 .- fd) .* dV)

    return η
end

"""
    ηB2g(L, E, dVs, Dxy, T)

Compute the B2g viscosity from the deformation potentials `Dxx` and `Dyy`.

For correct conversion to SI units, `E`, `Dxy`, and `T` must be expressed in units of eV and `dV` must be in units of ``(1/a)^2``, where ``a`` is the lattice constant.
"""
function ηB2g(L, E, dV, Dxy, T)
    fd = f0.(E, T) # Fermi dirac on grid points

    prefactor = 2 * hbar * e_charge / T # hbar * e_charge converts hbar to units of J.s

    η = prefactor * inner_product(Dxy, Dxy, L, fd .* (1 .- fd) .* dV)

    return η
end

"""
    σ_lifetime(L, v, E, dV, T)

Compute the effective scattering lifetime corresponding the conductivity.
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

"""
    η_lifetime(L, D, E, dV, T)

Compute the effective scattering lifetime corresponding the viscosity η = <D|L^-1|D>.
"""
function η_lifetime(L, D, E, dV, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd)

    norm = 0.0
    for i in eachindex(D)
        norm += w[i] * dV[i] * D[i]^2
    end
    η = inner_product(D, D, L, w .* dV)

    τ_eff = η / norm

    return τ_eff * hbar
end

###
### Hall Resistivity (FIXME: incomplete)
###

delta(i,j) = i == j ? 1.0 : 0.0
delta(v::Vector, i::Int, j::Int) = v[i] == v[j] ? 1.0 : 0.0

function momentum_derivative(n_ε, n_θ, n_bands, k, v, E)
    D = Matrix{Float64}(undef, length(k), length(k))
    for _j in eachindex(k)
        εi, θi 
    end
end

function hall_coefficient(L, k, v, E, dV, T; kwargs)
    momentum_derivative(kwargs[:n_ε], kwargs[:n_θ], kwargs[:n_bands], k, v, E)
end