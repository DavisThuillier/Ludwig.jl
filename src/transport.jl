###
### Transport coefficients
###
###
### Every coefficient here is a weighted correlator ⟨a|L⁻¹|b⟩ against the linearized collision
### operator `L`, so `correlator` is the only place a solve and a weighted sum happen.
###
### `v`, `energies`, and `weights` must be in the flat full-BZ ordering of `bz_quadrature_rule` --
### the ordering `full_collision_matrix` unfolds the stored IBZ block into. IBZ-only arrays passed
### against an unfolded `L` would break the tensor's symmetry while still looking plausible, so the
### lengths are checked on entry.
###
### Units follow the rest of the package: energies and temperature in eV, momenta in 1/a, velocities
### in a·eV (ħ = 1), and `weights` carrying the d^Dk/(2π)^D measure that `bz_quadrature_rule` applies.
### The conductivity prefactor is therefore 2π·G0 = 2e²/ħ rather than the G0/(2π) of a convention whose
### quadrature weights are bare d^Dk. 
###
### `L` is singular by construction: `enforce_particle_conservation!` makes the detailed-balance
### vector an exact left null vector, so every driving vector must be odd under the point group (as
### `v` and `ε·v` are) for the correlator to be meaningful, and a direct `\` carries a null-space
### component that only the final weighted sum removes.
###

### SI Unit Conversion Prefactors (up to division by unit cell volume Ω)
const electronic_prefactor = 2π * G0 # Converts to Siemens
const thermoelectric_prefactor = 2 * e_charge / hbar # Converts to Amperes
const thermal_prefactor = 2 * e_charge / hbar # Converts to J / s 

"""
    boltzmann_weight(energies, weights, temperature)

Fermi-window quadrature weight `f⁰(1 - f⁰)·w/T` on each grid point, equal to
`detailed_balance_factors(weights, energies, temperature) / temperature`.
"""
boltzmann_weight(energies, weights, temperature) =
    detailed_balance_factors(weights, energies, temperature) ./ temperature

"Throw an `ArgumentError` unless `(i, j)` indexes a `D`×`D` transport tensor."
@inline function check_tensor_indices(D::Int, i::Int, j::Int)
    (1 ≤ i ≤ D && 1 ≤ j ≤ D) ||
        throw(ArgumentError("transport tensor indices must lie in 1:$D; got ($i, $j)."))
    return nothing
end

"""
    check_transport_arguments(L, driving, energies, weights) -> Int

Number of grid points, after checking that `L` is square and that `driving`, `energies`, and
`weights` all have that length.
"""
function check_transport_arguments(L, driving, energies, weights)
    n = length(weights)
    length(energies) == n ||
        throw(DimensionMismatch("energies has length $(length(energies)), expected $n to match weights"))
    length(driving) == n ||
        throw(DimensionMismatch("driving vector has length $(length(driving)), expected $n to match weights"))
    if L isa Union{AbstractMatrix,Factorization}
        (size(L, 1) == n && size(L, 2) == n) ||
            throw(DimensionMismatch("L is $(size(L, 1))×$(size(L, 2)), expected $n×$n: the driving " *
                                    "vectors, energies, and weights must be on the full-BZ grid L acts " *
                                    "on, not the IBZ block"))
    end
    return n
end

###
### Weighted correlators
###

# Σᵢ aᵢ·weightᵢ·ϕᵢ, the single site where the weighted sum is formed. `eachindex` over all three
# arrays rejects mismatched lengths.
function weighted_dot(a, weight, ϕ)
    total = zero(eltype(a)) * zero(eltype(weight)) * zero(eltype(ϕ))
    for i in eachindex(a, weight, ϕ)
        total += a[i] * weight[i] * ϕ[i]
    end
    return total
end

# Factorize once when the caller relies on the default `\` and several solves share one `L`. A
# user-supplied solver, or an already-factorized `L`, is passed through untouched so that
# matrix-free and precomputed methods keep working.
prepare_solve(L, solve) = (solve === (\) && L isa AbstractMatrix) ? factorize(L) : L

"""
    correlator(a, b, L, weight; solve = \\)

Weighted correlator `⟨a|L⁻¹|b⟩ = Σᵢ aᵢ·weightᵢ·ϕᵢ` where `L·ϕ = b`.

`solve` is any callable `(L, b) -> ϕ`. The default `\\` dispatches on the type of `L`, so a
precomputed `LinearAlgebra.Factorization` is reused rather than refactorized; an iterative solver
such as `KrylovKit`'s can be passed as `solve = (A, rhs) -> linsolve(A, rhs)[1]`.
"""
correlator(a, b, L, weight; solve = (\)) = weighted_dot(a, weight, solve(L, b))

# Element type of a correlator against `L`: fixed by the operator, driving vector, and weights, not by
# whichever factorization the solver picks. Naming it keeps the tensor's type inferrable even though
# `factorize` has no inferrable return type.
correlator_type(L, driving, weight) =
    typeof(zero(eltype(L)) * zero(eltype(driving)) * zero(eltype(weight)))

# Driving vectors as one array per Cartesian component, in the order the tensors index them.
velocity_components(v::AbstractVector{SVector{D,T}}) where {D,T} =
    ntuple(i -> getindex.(v, i), Val(D))
energy_current_components(v::AbstractVector{SVector{D,T}}, energies) where {D,T} =
    ntuple(i -> energies .* getindex.(v, i), Val(D))

# One solve per component of `right`, reused across all D² correlators.
function transport_tensor(L, left::NTuple{D}, right::NTuple{D}, weight, solve) where {D}
    element_type = correlator_type(L, first(left), weight)
    prepared = prepare_solve(L, solve)
    ϕ = map(b -> solve(prepared, b), right)
    return SMatrix{D,D,element_type,D * D}([weighted_dot(left[i], weight, ϕ[j]) for i in 1:D, j in 1:D])
end

# `L` itself in the static limit, so that path never allocates a complex copy of the operator. `ω` and
# `q` are `nothing`-sentinelled rather than compared against zero so the real and complex paths are
# distinguished by type.
shifted_operator(L, v, ::Nothing, ::Nothing) = L
function shifted_operator(L, v, ω, q)
    L isa AbstractMatrix ||
        throw(ArgumentError("a finite ω or q shifts L, so the collision matrix itself is required; " *
                            "pass the matrix rather than a factorization or a matrix-free operator."))
    shifted = ω === nothing ? complex(L) : L - (im * ω) * I
    return q === nothing ? shifted : shifted + im * Diagonal(dot.(Ref(q), v))
end

###
### Electrical conductivity
###

"""
    conductivity(L, v, energies, weights, temperature; ω = nothing, q = nothing, solve = \\)

Electrical conductivity tensor `σᵢⱼ = 2e²⟨vᵢ|(L - iω + iq·v)⁻¹|vⱼ⟩` as a `D`×`D` `SMatrix`, in
Siemens·a^(2-D) — Siemens (sheet conductance) in 2D, Siemens per lattice constant in 3D.

`ω` (eV) and `q` (1/a) give the AC and finite-wavevector response and make the result complex; both
default to the static limit. `solve` is forwarded to [`correlator`](@ref).
"""
function conductivity(L, v::AbstractVector{SVector{D,Float64}}, energies, weights, temperature;
                      ω = nothing, q = nothing, solve = (\)) where {D}
    check_transport_arguments(L, v, energies, weights)
    weight = boltzmann_weight(energies, weights, temperature)
    currents = velocity_components(v)
    return electronic_prefactor * transport_tensor(shifted_operator(L, v, ω, q), currents, currents, weight, solve)
end

"""
    conductivity(L, v, energies, weights, temperature, i, j; ω = nothing, q = nothing, solve = \\)

Single component `σᵢⱼ` of the conductivity tensor, for any `i, j ∈ 1:D`.
"""
function conductivity(L, v::AbstractVector{SVector{D,Float64}}, energies, weights, temperature,
                      i::Int, j::Int; ω = nothing, q = nothing, solve = (\)) where {D}
    check_transport_arguments(L, v, energies, weights)
    check_tensor_indices(D, i, j)
    weight = boltzmann_weight(energies, weights, temperature)
    return electronic_prefactor * correlator(getindex.(v, i), getindex.(v, j),
                                  shifted_operator(L, v, ω, q), weight; solve)
end

"""
    effective_lifetime(L, a, energies, weights, temperature; solve = \\)

Effective scattering lifetime `ħ·⟨a|L⁻¹|a⟩/⟨a|a⟩` in seconds for the driving vector `a`, which may
be any channel: a velocity component gives the conductivity lifetime, a deformation potential the
viscous one.

`solve` is forwarded to [`correlator`](@ref).
"""
function effective_lifetime(L, a::AbstractVector, energies, weights, temperature; solve = (\))
    check_transport_arguments(L, a, energies, weights)
    weight = boltzmann_weight(energies, weights, temperature)
    return hbar * correlator(a, a, L, weight; solve) / weighted_dot(a, weight, a)
end

###
### Thermal transport
###

"""
    thermal_conductivity(L, v, energies, weights, temperature; solve = \\)

Thermal conductivity correlator `κᵢⱼ = ⟨εvᵢ|L⁻¹|εvⱼ⟩` as a `D`×`D` `SMatrix`.
"""
function thermal_conductivity(L, v::AbstractVector{SVector{D,Float64}}, energies, weights, temperature;
                              solve = (\)) where {D}
    check_transport_arguments(L, v, energies, weights)
    weight = boltzmann_weight(energies, weights, temperature)
    currents = energy_current_components(v, energies)
    return thermal_prefactor * transport_tensor(L, currents, currents, weight, solve)
end

"""
    thermal_conductivity(L, v, energies, weights, temperature, i, j; solve = \\)

Single component `κᵢⱼ` of the thermal conductivity correlator, for any `i, j ∈ 1:D`.
"""
function thermal_conductivity(L, v::AbstractVector{SVector{D,Float64}}, energies, weights, temperature,
                              i::Int, j::Int; solve = (\)) where {D}
    check_transport_arguments(L, v, energies, weights)
    check_tensor_indices(D, i, j)
    weight = boltzmann_weight(energies, weights, temperature)
    return thermal_prefactor * correlator(energies .* getindex.(v, i), energies .* getindex.(v, j), L, weight; solve)
end

"""
    thermoelectric_conductivity(L, v, energies, weights, temperature; solve = \\)

Thermoelectric correlator `εᵢⱼ = ⟨vᵢ|L⁻¹|εvⱼ⟩` as a `D`×`D` `SMatrix`.
"""
function thermoelectric_conductivity(L, v::AbstractVector{SVector{D,Float64}}, energies, weights, temperature;
                                     solve = (\)) where {D}
    check_transport_arguments(L, v, energies, weights)
    weight = boltzmann_weight(energies, weights, temperature)
    return thermoelectric_prefactor * transport_tensor(L, velocity_components(v), energy_current_components(v, energies),
                            weight, solve)
end

"""
    thermoelectric_conductivity(L, v, energies, weights, temperature, i, j; solve = \\)

Single component `εᵢⱼ` of the thermoelectric correlator, for any `i, j ∈ 1:D`.
"""
function thermoelectric_conductivity(L, v::AbstractVector{SVector{D,Float64}}, energies, weights, temperature,
                                     i::Int, j::Int; solve = (\)) where {D}
    check_transport_arguments(L, v, energies, weights)
    check_tensor_indices(D, i, j)
    weight = boltzmann_weight(energies, weights, temperature)
    return thermoelectric_prefactor * correlator(getindex.(v, i), energies .* getindex.(v, j), L, weight; solve)
end

"""
    peltier_tensor(L, v, energies, weights, temperature; solve = \\)

Peltier correlator `τᵢⱼ = ⟨εvᵢ|L⁻¹|vⱼ⟩` as a `D`×`D` `SMatrix`.
"""
function peltier_tensor(L, v::AbstractVector{SVector{D, Float64}}, energies, weights, temperature;
                        solve = (\)) where {D}
    check_transport_arguments(L, v, energies, weights)
    weight = boltzmann_weight(energies, weights, temperature)
    return thermoelectric_prefactor * transport_tensor(L, energy_current_components(v, energies), velocity_components(v),
                            weight, solve)
end

"""
    peltier_tensor(L, v, energies, weights, temperature, i, j; solve = \\)

Single component `τᵢⱼ` of the Peltier correlator, for any `i, j ∈ 1:D`.
"""
function peltier_tensor(L, v::AbstractVector{SVector{D,Float64}}, energies, weights, temperature,
                        i::Int, j::Int; solve = (\)) where {D}
    check_transport_arguments(L, v, energies, weights)
    check_tensor_indices(D, i, j)
    weight = boltzmann_weight(energies, weights, temperature)
    return thermoelectric_prefactor * correlator(energies .* getindex.(v, i), getindex.(v, j), L, weight; solve)
end
