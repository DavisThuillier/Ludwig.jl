"""
    boltzmann_weight(E, dV, T)

Return the Fermi-window quadrature weight ``f^{(0)}(\\varepsilon)\\,(1 - f^{(0)}(\\varepsilon))\\,\\Delta V`` on each
grid point with energy `E[i]`, momentum-space patch area `dV[i]`, and temperature `T`.

This is the natural weight for transport inner products in the Boltzmann formalism: it sets
the size of the Fermi window and the integration measure on the Fermi surface.

# Examples
```jldoctest
julia> Ludwig.boltzmann_weight([0.0], [1.0], 1.0)
1-element Vector{Float64}:
 0.25
```
"""
function boltzmann_weight(E, dV, T)
    fd = f0.(E, T)
    return fd .* (1 .- fd) .* dV
end

"""
    inner_product(a, b, L, w)

Compute the weighted inner product ``\\langle a | L^{-1} | b\\rangle`` with the weight vector `w`.

Solves ``L \\phi = b`` with `IterativeSolvers.bicgstabl` and returns ``\\sum_i a_i\\, w_i\\, \\phi_i``.

# Returns
A scalar; complex when `L` has a complex part (e.g. ``-i\\omega I``), real otherwise.

# Examples
```julia
# Given a collision matrix L on `n` patches and per-patch weights `w`,
# compute the conductivity-like inner product `<v_x | L^{-1} | v_x>`.
σxx = inner_product(vx, vx, L, w)
```
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

Compute the conductivity tensor using
``\\sigma_{ij}(\\omega, \\mathbf{q}) = 2 e^2 \\langle v_i | (L - i\\omega + i\\mathbf{q}\\cdot\\mathbf{v})^{-1} | v_j \\rangle``.

For correct conversion to SI units, `E` and `T` must be expressed in units of eV, `dV` must be in units of ``(1/a)^2``, where ``a`` is the lattice constant, and `v` must be in units of ``(a/\\hbar) \\,\\mathrm{eV}``.

# Returns
A 2×2 `Matrix{ComplexF64}` of conductivities. Imaginary parts vanish at ``\\omega = 0`` and
``\\mathbf{q} = 0``.
"""
function electrical_conductivity(L, v, E, dV, T, ω = 0.0, q = [0.0, 0.0])
    weight = boltzmann_weight(E, dV, T)

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

# Returns
A complex scalar with the same prefactor convention as
[`electrical_conductivity`](@ref); imaginary part vanishes at ``\\omega = 0``.
"""
function longitudinal_electrical_conductivity(L, vx, E, dV, T, ω = 0.0)
    L′ = ω != 0.0 ? (L - im*ω*I) : L
    σxx = inner_product(vx, vx, L′, boltzmann_weight(E, dV, T))
    return (G0 / (2π)) * (σxx / T)
end

"""
    thermal_conductivity(L, v, E, dV, T)

Compute the thermal conductivity tensor using
``\\kappa_{ij} = \\langle \\varepsilon v_i | L^{-1} | \\varepsilon v_j \\rangle``,
where the inner product is weighted by ``f^{(0)}(1 - f^{(0)}) \\Delta V``.

# Returns
A 2×2 `Matrix{ComplexF64}`. Unit conversion must be applied by the caller.
"""
function thermal_conductivity(L, v, E, dV, T)
    weight = boltzmann_weight(E, dV, T)

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

# Returns
A complex scalar; same units as the tensor-valued method.
"""
function thermal_conductivity(L, v, E, dV, T, i::Int, j::Int)
    if !((0 < i < 3) && (0 < j < 3))
        throw(ArgumentError("indices of tensor must be 1 or 2; got ($i, $j)."))
    end
    weight = boltzmann_weight(E, dV, T)

    j1 = E .* map(x -> x[i], v)
    j2 = E .* map(x -> x[j], v)

    return inner_product(j1, j2, L, weight)
end

"""
    thermoelectric_conductivity(L, v, E, dV, T)

Compute the thermoelectric conductivity tensor using
``\\epsilon_{ij} = \\langle v_i | L^{-1} | \\varepsilon v_j \\rangle``,
where the inner product is weighted by ``f^{(0)}(1 - f^{(0)}) \\Delta V``.

# Returns
A 2×2 `Matrix{ComplexF64}`. Unit conversion must be applied by the caller.
"""
function thermoelectric_conductivity(L, v, E, dV, T)
    weight = boltzmann_weight(E, dV, T)

    jx  = first.(v)
    jy  = last.(v)
    jex = E .* jx # Energy current in x-direction
    jey = E .* jy # Energy current in y-direction

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

# Returns
A complex scalar; same units as the tensor-valued method.
"""
function thermoelectric_conductivity(L, v, E, dV, T, i::Int, j::Int)
    if !((0 < i < 3) && (0 < j < 3))
        throw(ArgumentError("indices of tensor must be 1 or 2; got ($i, $j)."))
    end
    weight = boltzmann_weight(E, dV, T)

    j1  = map(x -> x[i], v)
    je2 = E .* map(x -> x[j], v)
    return inner_product(j1, je2, L, weight)
end

"""
    peltier_tensor(L, v, E, dV, T)

Compute the Peltier tensor using
``\\tau_{ij} = \\langle \\varepsilon v_i | L^{-1} | v_j \\rangle``,
where the inner product is weighted by ``f^{(0)}(1 - f^{(0)}) \\Delta V``.

# Returns
A 2×2 `Matrix{ComplexF64}`. Unit conversion must be applied by the caller.
"""
function peltier_tensor(L, v, E, dV, T)
    weight = boltzmann_weight(E, dV, T)

    jx  = first.(v)
    jy  = last.(v)
    jex = E .* jx # Energy current in x-direction
    jey = E .* jy # Energy current in y-direction

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

# Returns
A complex scalar; same units as the tensor-valued method.
"""
function peltier_tensor(L, v, E, dV, T, i::Int, j::Int)
    if !((0 < i < 3) && (0 < j < 3))
        throw(ArgumentError("indices of tensor must be 1 or 2; got ($i, $j)."))
    end
    weight = boltzmann_weight(E, dV, T)

    je1 = E .* map(x -> x[i], v)
    j2  = map(x -> x[j], v)
    return inner_product(je1, j2, L, weight)
end

"""
    ηB1g(L, E, dV, Dxx, Dyy, T)

Compute the B1g viscosity from the deformation potentials `Dxx` and `Dyy`.

For correct conversion to SI units, `E`, `Dxx`, `Dyy`, and `T` must be expressed in units of eV and `dV` must be in units of ``(1/a)^2``, where ``a`` is the lattice constant.

# Returns
A scalar viscosity in SI units (Pa·s).
"""
function ηB1g(L, E, dV, Dxx, Dyy, T)
    prefactor = 2 * hbar * e_charge / T # hbar * e_charge converts hbar to units of J.s
    return prefactor * 0.25 *
        inner_product(Dxx .- Dyy, Dxx .- Dyy, L, boltzmann_weight(E, dV, T))
end

"""
    ηB2g(L, E, dV, Dxy, T)

Compute the B2g viscosity from the deformation potential `Dxy`.

For correct conversion to SI units, `E`, `Dxy`, and `T` must be expressed in units of eV and `dV` must be in units of ``(1/a)^2``, where ``a`` is the lattice constant.

# Returns
A scalar viscosity in SI units (Pa·s).
"""
function ηB2g(L, E, dV, Dxy, T)
    prefactor = 2 * hbar * e_charge / T # hbar * e_charge converts hbar to units of J.s
    return prefactor * inner_product(Dxy, Dxy, L, boltzmann_weight(E, dV, T))
end

"""
    σ_lifetime(L, v, E, dV, T)

Compute the effective scattering lifetime corresponding to the conductivity.

Defined by ``\\tau_\\sigma = \\hbar\\, \\langle v_x | L^{-1} | v_x \\rangle / \\langle v_x | v_x \\rangle``,
with both inner products weighted by ``f^{(0)}(1 - f^{(0)}) \\Delta V``.

# Returns
A real scalar lifetime in SI units (seconds), assuming `E` and `T` are in eV.
"""
function σ_lifetime(L, v, E, dV, T)
    weight = boltzmann_weight(E, dV, T)
    vx = first.(v)

    norm_squared = 0.0
    for i in eachindex(vx)
        norm_squared += weight[i] * vx[i]^2
    end
    σ = inner_product(vx, vx, L, weight)

    return real(σ / norm_squared) * hbar
end

"""
    η_lifetime(L, D, E, dV, T)

Compute the effective scattering lifetime corresponding to the viscosity
``\\eta = \\langle D | L^{-1} | D \\rangle``.

# Returns
A scalar lifetime in SI units (seconds), assuming `E` and `T` are in eV.
"""
function η_lifetime(L, D, E, dV, T)
    weight = boltzmann_weight(E, dV, T)

    norm_squared = 0.0
    for i in eachindex(D)
        norm_squared += weight[i] * D[i]^2
    end
    η = inner_product(D, D, L, weight)

    return (η / norm_squared) * hbar
end
