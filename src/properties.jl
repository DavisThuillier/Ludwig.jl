"""
    boltzmann_weight(E, dV, T)

Return the Fermi-window quadrature weight
``f^{(0)}(\\varepsilon)\\,(1 - f^{(0)}(\\varepsilon))\\,\\Delta V / T`` on each grid point
with energy `E[i]`, momentum-space patch area `dV[i]`, and temperature `T`.

This is equivalent to ``(-\\partial f^{(0)}/\\partial \\varepsilon)\\,\\Delta V``, the
canonical Fermi-window weight of the linearized Boltzmann formalism. The factor of `1/T`
is included here so that every transport function in this file can write its inner
product as ``\\langle a | L^{-1} | b \\rangle`` with no further temperature factors in the
prefactor.

# Examples
```jldoctest
julia> Ludwig.boltzmann_weight([0.0], [1.0], 1.0)
1-element Vector{Float64}:
 0.25
```
"""
function boltzmann_weight(E, dV, T)
    fd = f0.(E, T)
    return fd .* (1 .- fd) .* dV ./ T
end

"Throw an `ArgumentError` if `(i, j)` are not a valid index pair into a 2×2 transport tensor."
@inline function check_tensor_indices(i::Int, j::Int)
    1 ≤ i ≤ 2 && 1 ≤ j ≤ 2 ||
        throw(ArgumentError("indices of tensor must be 1 or 2; got ($i, $j)."))
    return nothing
end

"""
    inner_product(a, b, L, w; solve = \\)

Compute the weighted inner product ``\\langle a | L^{-1} | b\\rangle`` with the weight vector `w`.

`solve` is any 2-argument callable `(L, b) -> ϕ` that returns a solution of ``L \\phi = b``.
The default `\\` dispatches on the type of `L`:

- a `Matrix` triggers a fresh dense LU factorization on each call;
- a precomputed `LinearAlgebra.Factorization` (e.g. `lu(L)`, `qr(L)`, `cholesky(L)`)
  reuses that factorization, avoiding the ``O(n^3)`` work on repeated calls;
- a `SparseMatrixCSC` dispatches to sparse `\\`;
- any user type with `Base.:\\` defined is supported.

To use an iterative solver, pass it explicitly. Iterative solvers from
`IterativeSolvers.jl` (`bicgstabl`, `gmres`, …) are not loaded by Ludwig itself,
so the caller is responsible for `using IterativeSolvers`:

```julia
using IterativeSolvers
inner_product(a, b, L, w; solve = bicgstabl)
inner_product(a, b, L, w; solve = (A, rhs) -> gmres(A, rhs; reltol = 1e-12))
```

# Returns
A scalar; complex when `L` has a complex part (e.g. ``-i\\omega I``), real otherwise.

# Examples
```julia
# Factor the collision matrix once and reuse across multiple inner products.
F = lu(L)
σxx = inner_product(vx, vx, F, w)
σxy = inner_product(vx, vy, F, w)
```
"""
function inner_product(a, b, L, w; solve = (\))
    ϕ = solve(L, b)
    prod = zero(eltype(ϕ)) * zero(eltype(a)) * zero(eltype(w))
    for i in eachindex(a)
        prod += a[i] * w[i] * ϕ[i]
    end
    return prod
end

# When the caller relies on the default `\` solver and we know we'll perform multiple
# solves against the same matrix, factorize once up-front. For any user-supplied solver
# we leave the matrix untouched so that iterative or matrix-free methods still work.
prepare_solve(L, solve) = solve === (\) ? factorize(L) : L

"""
    electrical_conductivity(L, v, E, dV, T [, ω = 0.0, q = [0.0, 0.0]]; solve = \\)

Compute the conductivity tensor using
``\\sigma_{ij}(\\omega, \\mathbf{q}) = 2 e^2 \\langle v_i | (L - i\\omega + i\\mathbf{q}\\cdot\\mathbf{v})^{-1} | v_j \\rangle``.

For correct conversion to SI units, `E` and `T` must be expressed in units of eV, `dV` must be in units of ``(1/a)^2``, where ``a`` is the lattice constant, and `v` must be in units of ``(a/\\hbar) \\,\\mathrm{eV}``.

`solve` is forwarded to [`inner_product`](@ref) and controls the linear-solver back end.
The default `\\` factorizes the modified collision matrix once and reuses the factorization
across all four tensor-component solves.

# Returns
A 2×2 `Matrix{ComplexF64}` of conductivities. Imaginary parts vanish at ``\\omega = 0`` and
``\\mathbf{q} = 0``.
"""
function electrical_conductivity(L, v, E, dV, T, ω = 0.0, q = [0.0, 0.0]; solve = (\))
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

    F = prepare_solve(L′, solve)

    σ[1,1] = inner_product(first.(v), first.(v), F, weight; solve)
    σ[1,2] = inner_product(first.(v), last.(v),  F, weight; solve)
    σ[2,1] = inner_product(last.(v),  first.(v), F, weight; solve)
    σ[2,2] = inner_product(last.(v),  last.(v),  F, weight; solve)

    return (G0 / (2π)) * σ
end

"""
    longitudinal_electrical_conductivity(L, vx, E, dV, T [, ω]; solve = \\)

Compute ``\\sigma_{xx}`` only.

`solve` is forwarded to [`inner_product`](@ref); see [`electrical_conductivity`](@ref).

# Returns
A complex scalar with the same prefactor convention as
[`electrical_conductivity`](@ref); imaginary part vanishes at ``\\omega = 0``.
"""
function longitudinal_electrical_conductivity(L, vx, E, dV, T, ω = 0.0; solve = (\))
    L′ = ω != 0.0 ? (L - im*ω*I) : L
    σxx = inner_product(vx, vx, L′, boltzmann_weight(E, dV, T); solve)
    return (G0 / (2π)) * σxx
end

"""
    thermal_conductivity(L, v, E, dV, T; solve = \\)

Compute the thermal conductivity tensor using
``\\kappa_{ij} = \\langle \\varepsilon v_i | L^{-1} | \\varepsilon v_j \\rangle``,
where the inner product is weighted by ``f^{(0)}(1 - f^{(0)}) \\Delta V / T``
(see `Ludwig.boltzmann_weight`).

`solve` is forwarded to [`inner_product`](@ref); see [`electrical_conductivity`](@ref).

# Returns
A 2×2 `Matrix{ComplexF64}`. Unit conversion must be applied by the caller.
"""
function thermal_conductivity(L, v, E, dV, T; solve = (\))
    weight = boltzmann_weight(E, dV, T)

    jex = E .* first.(v) # Energy current in x-direction
    jey = E .* last.(v)  # Energy current in y-direction

    F = prepare_solve(L, solve)

    λ = Matrix{ComplexF64}(undef, 2, 2)
    λ[1,1] = inner_product(jex, jex, F, weight; solve)
    λ[1,2] = inner_product(jex, jey, F, weight; solve)
    λ[2,1] = inner_product(jey, jex, F, weight; solve)
    λ[2,2] = inner_product(jey, jey, F, weight; solve)

    return λ
end

"""
    thermal_conductivity(L, v, E, dV, T, i::Int, j::Int; solve = \\)

Compute the single component ``\\kappa_{ij}`` of the thermal conductivity tensor.
`i` and `j` must be 1 or 2.

# Returns
A complex scalar; same units as the tensor-valued method.
"""
function thermal_conductivity(L, v, E, dV, T, i::Int, j::Int; solve = (\))
    check_tensor_indices(i, j)
    weight = boltzmann_weight(E, dV, T)

    j1 = E .* map(x -> x[i], v)
    j2 = E .* map(x -> x[j], v)

    return inner_product(j1, j2, L, weight; solve)
end

"""
    thermoelectric_conductivity(L, v, E, dV, T; solve = \\)

Compute the thermoelectric conductivity tensor using
``\\epsilon_{ij} = \\langle v_i | L^{-1} | \\varepsilon v_j \\rangle``,
where the inner product is weighted by ``f^{(0)}(1 - f^{(0)}) \\Delta V / T``
(see `Ludwig.boltzmann_weight`).

`solve` is forwarded to [`inner_product`](@ref); see [`electrical_conductivity`](@ref).

# Returns
A 2×2 `Matrix{ComplexF64}`. Unit conversion must be applied by the caller.
"""
function thermoelectric_conductivity(L, v, E, dV, T; solve = (\))
    weight = boltzmann_weight(E, dV, T)

    jx  = first.(v)
    jy  = last.(v)
    jex = E .* jx # Energy current in x-direction
    jey = E .* jy # Energy current in y-direction

    F = prepare_solve(L, solve)

    ϵ = Matrix{ComplexF64}(undef, 2, 2)
    ϵ[1,1] = inner_product(jx, jex, F, weight; solve)
    ϵ[1,2] = inner_product(jx, jey, F, weight; solve)
    ϵ[2,1] = inner_product(jy, jex, F, weight; solve)
    ϵ[2,2] = inner_product(jy, jey, F, weight; solve)

    return ϵ
end

"""
    thermoelectric_conductivity(L, v, E, dV, T, i::Int, j::Int; solve = \\)

Compute the single component ``\\epsilon_{ij}`` of the thermoelectric conductivity tensor.
`i` and `j` must be 1 or 2.

# Returns
A complex scalar; same units as the tensor-valued method.
"""
function thermoelectric_conductivity(L, v, E, dV, T, i::Int, j::Int; solve = (\))
    check_tensor_indices(i, j)
    weight = boltzmann_weight(E, dV, T)

    j1  = map(x -> x[i], v)
    je2 = E .* map(x -> x[j], v)
    return inner_product(j1, je2, L, weight; solve)
end

"""
    peltier_tensor(L, v, E, dV, T; solve = \\)

Compute the Peltier tensor using
``\\tau_{ij} = \\langle \\varepsilon v_i | L^{-1} | v_j \\rangle``,
where the inner product is weighted by ``f^{(0)}(1 - f^{(0)}) \\Delta V / T``
(see `Ludwig.boltzmann_weight`).

`solve` is forwarded to [`inner_product`](@ref); see [`electrical_conductivity`](@ref).

# Returns
A 2×2 `Matrix{ComplexF64}`. Unit conversion must be applied by the caller.
"""
function peltier_tensor(L, v, E, dV, T; solve = (\))
    weight = boltzmann_weight(E, dV, T)

    jx  = first.(v)
    jy  = last.(v)
    jex = E .* jx # Energy current in x-direction
    jey = E .* jy # Energy current in y-direction

    F = prepare_solve(L, solve)

    τ = Matrix{ComplexF64}(undef, 2, 2)
    τ[1,1] = inner_product(jex, jx, F, weight; solve)
    τ[1,2] = inner_product(jex, jy, F, weight; solve)
    τ[2,1] = inner_product(jey, jx, F, weight; solve)
    τ[2,2] = inner_product(jey, jy, F, weight; solve)

    return τ
end

"""
    peltier_tensor(L, v, E, dV, T, i::Int, j::Int; solve = \\)

Compute the single component ``\\tau_{ij}`` of the Peltier tensor.
`i` and `j` must be 1 or 2.

# Returns
A complex scalar; same units as the tensor-valued method.
"""
function peltier_tensor(L, v, E, dV, T, i::Int, j::Int; solve = (\))
    check_tensor_indices(i, j)
    weight = boltzmann_weight(E, dV, T)

    je1 = E .* map(x -> x[i], v)
    j2  = map(x -> x[j], v)
    return inner_product(je1, j2, L, weight; solve)
end

"""
    ηB1g(L, E, dV, Dxx, Dyy, T; solve = \\)

Compute the B1g viscosity from the deformation potentials `Dxx` and `Dyy`.

For correct conversion to SI units, `E`, `Dxx`, `Dyy`, and `T` must be expressed in units of eV and `dV` must be in units of ``(1/a)^2``, where ``a`` is the lattice constant.

# Returns
A scalar viscosity in SI units (Pa·s).
"""
function ηB1g(L, E, dV, Dxx, Dyy, T; solve = (\))
    prefactor = 2 * hbar * e_charge # hbar * e_charge converts hbar to units of J.s
    return prefactor * 0.25 *
        inner_product(Dxx .- Dyy, Dxx .- Dyy, L, boltzmann_weight(E, dV, T); solve)
end

"""
    ηB2g(L, E, dV, Dxy, T; solve = \\)

Compute the B2g viscosity from the deformation potential `Dxy`.

For correct conversion to SI units, `E`, `Dxy`, and `T` must be expressed in units of eV and `dV` must be in units of ``(1/a)^2``, where ``a`` is the lattice constant.

# Returns
A scalar viscosity in SI units (Pa·s).
"""
function ηB2g(L, E, dV, Dxy, T; solve = (\))
    prefactor = 2 * hbar * e_charge # hbar * e_charge converts hbar to units of J.s
    return prefactor * inner_product(Dxy, Dxy, L, boltzmann_weight(E, dV, T); solve)
end

"""
    σ_lifetime(L, v, E, dV, T; solve = \\)

Compute the effective scattering lifetime corresponding to the conductivity.

Defined by ``\\tau_\\sigma = \\hbar\\, \\langle v_x | L^{-1} | v_x \\rangle / \\langle v_x | v_x \\rangle``,
with both inner products weighted by ``f^{(0)}(1 - f^{(0)}) \\Delta V / T``
(see `Ludwig.boltzmann_weight`).

# Returns
A real scalar lifetime in SI units (seconds), assuming `E` and `T` are in eV.
"""
function σ_lifetime(L, v, E, dV, T; solve = (\))
    weight = boltzmann_weight(E, dV, T)
    vx = first.(v)

    norm_squared = 0.0
    for i in eachindex(vx)
        norm_squared += weight[i] * vx[i]^2
    end
    σ = inner_product(vx, vx, L, weight; solve)

    return real(σ / norm_squared) * hbar
end

"""
    η_lifetime(L, D, E, dV, T; solve = \\)

Compute the effective scattering lifetime corresponding to the viscosity
``\\eta = \\langle D | L^{-1} | D \\rangle``.

# Returns
A scalar lifetime in SI units (seconds), assuming `E` and `T` are in eV.
"""
function η_lifetime(L, D, E, dV, T; solve = (\))
    weight = boltzmann_weight(E, dV, T)

    norm_squared = 0.0
    for i in eachindex(D)
        norm_squared += weight[i] * D[i]^2
    end
    η = inner_product(D, D, L, weight; solve)

    return (η / norm_squared) * hbar
end
