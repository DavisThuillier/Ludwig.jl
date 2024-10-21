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
    conductivity(L, v, E, dV, T [, ω = 0.0, q = [0.0, 0.0]])

Compute the conductivity tensor using ``\\sigma_{ij}(ω, \\mathbf{q}) = 2 e^2 \\langle v_i | (L - i\\omega + i\\mathbf{q}\\cdot\\mathbf{v})^{-1} | v_j \\rangle``.

For correct conversion to SI units, `E` and `T` must be expressed in units of eV, dV must be in units of ``(2pi / a)^2,`` where ``a`` is the lattice constant, and `v` must be in units of ``(a / h) eV``.
"""
function conductivity(L, v, E, dV, T, ω = 0.0, q = [0.0, 0.0])
    fd = f0.(E, T) # Fermi dirac on grid points
    weight = fd .* (1 .- fd) .* dV

    σ = Matrix{ComplexF64}(undef, 2, 2)

    if ω != 0.0 || q != [0.0, 0.0]
        L′ = L - im*ω*I + im * diagm(dot.(Ref(q), v))
    else
        L′ = L
    end

    σ[1,1] = inner_product(first.(v), first.(v), L′, weight)
    σ[1,2] = inner_product(first.(v), last.(v), L′, weight)
    σ[2,1] = inner_product(last.(v), first.(v), L′, weight)
    σ[2,2] = inner_product(last.(v), last.(v), L′, weight)

    return (G0 / (2π)) * (σ / T)
end

"""
    longitudinal_conductivity(L, vx, E, dV, T [, ω])

Compute only the ``\\sigma_{xx}`` component of the conductivity tensor.
"""
function longitudinal_conductivity(L, vx, E, dV, T, ω = 0.0)
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
    ηB1g(L, E, dVs, Dxx, Dyy, T)

Compute the B1g viscosity from the deformation potentials `Dxx` and `Dyy`.

For correct conversion to SI units, `E`, `Dxx`, `Dyy`, and `T` must be expressed in units of eV and dV must be in units of ``(2pi / a)^2,`` where ``a`` is the lattice constant.
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

For correct conversion to SI units, `E`, `Dxy`, and `T` must be expressed in units of eV and dV must be in units of ``(2pi / a)^2,`` where ``a`` is the lattice constant.
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
    η_lifetime(L, Dxx, Dyy, E, dV, T)

Compute the effective scattering lifetime corresponding the conductivity.
"""
function η_lifetime(L, Dxx, Dyy, E, dV, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    w = fd .* (1 .- fd)

    D = Dxx .- Dyy

    norm = 0.0
    for i in eachindex(D)
        norm += w[i] * dV[i] * D[i]^2
    end
    η = inner_product(D, D, L, w .* dV)

    τ_eff = η / norm

    return τ_eff * hbar
end

delta(i,j) = i == j ? 1.0 : 0.0
delta(v::Vector, i::Int, j::Int) = v[i] == v[j] ? 1.0 : 0.0

function _momentum_derivative(n_ε, n_θ, n_bands, k, v, E)
    D = Matrix{Float64}(undef, length(k), length(k))
    for _j in eachindex(k)
        εi, θi 
    end
end

function hall_coefficient(L, k, v, E, dV, T; kwargs)
    _momentum_derivative(kwargs[:n_ε], kwargs[:n_θ], kwargs[:n_bands], k, v, E)
end