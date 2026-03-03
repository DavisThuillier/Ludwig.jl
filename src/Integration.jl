module Integration

import StaticArrays: SVector, MVector
using LinearAlgebra
using ForwardDiff
using Interpolations

import ..Lattices: map_to_bz, Lattice, NoLattice, reciprocal_lattice_vectors, get_bz
import ..FSMesh: Patch, VirtualPatch, BZSymmetryMap
import ..Utilities: f0

export electron_electron, electron_impurity, electron_phonon, fill_from_ibz!

const ρ::Float64 = 4*6^(1/3)/pi
const ρ3::Float64 = 4*sqrt(2)/pi

"Volume of the 5-dimensional unit sphere"
const vol::Float64 = (8 * pi^2 / 15)

"""
    ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, εabc, v, T)

Compute the integral
```math
    \\mathcal{K}_{abc} \\equiv \\int_a \\int_b \\int_c (1 - f^{(0)}(\\mathbf{k}_a + \\mathbf{k}_b + \\mathbf{k}_c)) \\delta(\\varepsilon_a + \\varepsilon_b - \\varepsilon_c - \\varepsilon(\\mathbf{k}_a + \\mathbf{k}_b - \\mathbf{k}_c))
```
with group velocity `v` and energy `εabc` at temperature `T`.
```math
    \\int_i \\equiv \\frac{1}{a^2} \\int_{\\mathbf{k} \\in \\mathcal{P}_i} d^2\\mathbf{k}
```
is an integral over momenta in patch ``\\mathcal{P}_i``.

"""
function ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, v, εabc, T)
    δ = a.e + b.e - c.e - εabc

    # ζ[1], ζ[2] = v' * a.jinv
    # ζ[3], ζ[4] = v' * b.jinv
    # ζ[5], ζ[6] = - v' * c.jinv
    ζ[1] = v[1] * a.jinv[1,1] + v[2] * a.jinv[2,1]
    ζ[2] = v[1] * a.jinv[1,2] + v[2] * a.jinv[2,2]
    ζ[3] = v[1] * b.jinv[1,1] + v[2] * b.jinv[2,1]
    ζ[4] = v[1] * b.jinv[1,2] + v[2] * b.jinv[2,2]
    ζ[5] = - (v[1] * c.jinv[1,1] + v[2] * c.jinv[2,1])
    ζ[6] = - (v[1] * c.jinv[1,2] + v[2] * c.jinv[2,2])

    # u = [a.de/2.0, 0.0, b.de/2.0, 0.0, -c.de/2.0, 0.0] - ζ
    u[1] = a.de/2.0 - ζ[1]
    u[2] = - ζ[2]
    u[3] = b.de/2.0 - ζ[3]
    u[4] = - ζ[4]
    u[5] = (- c.de/2.0) - ζ[5]
    u[6] = - ζ[6]
    

    ρ < δ^2 / dot(u,u) && return 0.0 # Check for intersection of energy conserving 5-plane with coordinate space

    Δε = - δ * dot(ζ, u) / dot(u,u)
    r5 = (ρ - δ^2 / dot(u,u) )^(5/2)

    return vol * a.djinv * b.djinv * c.djinv * r5 * (1 - f0(εabc + Δε, T)) / norm(u)
end

function ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, k, εabc, ε::Function, T::Real)
    v::SVector{2,Float64} = ForwardDiff.gradient(ε, k)
    return ee_kernel!(ζ, u, a, b, c, v, εabc, T)
end 

function ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, k, εabc, itp::ScaledInterpolation, T::Real)
    v::SVector{2,Float64} = Interpolations.gradient(itp, k[1], k[2])
    return ee_kernel!(ζ, u, a, b, c, v, εabc, T)
end 

ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, d::VirtualPatch, T::Real) = ee_kernel!(ζ, u, a, b, c, d.v, d.e, T)

function ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, k, εabc, itp::ScaledInterpolation, invrlv, T::Real)
    v::SVector{2,Float64} = invrlv * Interpolations.gradient(itp, k[1], k[2])
    return ee_kernel!(ζ, u, a, b, c, v, εabc, T)
end

#################################
### Generic Scattering Vertex ###
#################################

"""
    electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, bands, T::Real, Weff_squared, rlv, bz; umklapp = true, kwargs...)

Compute the element (`i`,`j`) of the linearized Boltzmann collision operator for electron electron scattering.

The bands used to construct `grid` are callable using the interpolated dispersions in `itps`. The vector `f0s` stores the value of the Fermi-Dirac distribution at each patch center and can be calculated independent of `i` and `j`. The functions `Fpp` and `Fpk` are vertex factors defined for two Patch variables and for one Patch and one momentum vector respectively, using the orbital weight vectors defined `weights` evaluated at the patch centers of `grid`. 
"""
function electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, bands, T::Real, Weff_squared, rlv, bz; umklapp = true, kwargs...)
    Lij::Float64 = 0.0
    w123::Float64 = 0.0
    w124::Float64 = 0.0

    ζ  = MVector{6,Float64}(undef)
    u  = MVector{6,Float64}(undef)

    kij = grid[i].k + grid[j].k
    qij = grid[i].k - grid[j].k

    kijm = Vector{Float64}(undef, 2)
    kijm_rlb = Vector{Float64}(undef, 2)

    qimj = Vector{Float64}(undef, 2)
    qimj_rlb = Vector{Float64}(undef, 2)

    invrlv = inv(rlv) 
    is_function = map(x -> isa(x, Function), bands) # For determining how to evaluate bands

    for m in eachindex(grid)
        kijm .= kij .- grid[m].k 
        kijm_rlb .= mod.(invrlv * kijm, 1.0) # In reciprocal lattice basis, mapped to interpolation region

        for μ in eachindex(bands)
            if is_function[μ]
                p = VirtualPatch(
                    bands[μ](kijm),
                    umklapp ? map_to_bz(kijm, bz, rlv, invrlv) : kijm,
                    ForwardDiff.gradient(bands[μ], kijm),
                    μ
                )
            else # Otherwise assume this is an interpolation
                p = VirtualPatch(
                    bands[μ](kijm_rlb[1], kijm_rlb[2]),
                    umklapp ? map_to_bz(kijm, bz, rlv, invrlv) : kijm,
                    invrlv * Interpolations.gradient(bands[μ], kijm_rlb[1], kijm_rlb[2]),
                    μ
                )
            end
            w123 = Weff_squared(grid[i], grid[j], grid[m], p; kwargs)

            if w123 != 0
                Lij += w123 * ee_kernel!(ζ, u, grid[i], grid[j], grid[m], p, T) * f0s[j] * (1 - f0s[m])
            end
        end

        qimj .= qij .+ grid[m].k
        qimj_rlb .= mod.(invrlv * qimj, 1.0) # In reciprocal lattice basis

        for μ in eachindex(bands)
            if is_function[μ]
                p = VirtualPatch(
                    bands[μ](qimj),
                    umklapp ? map_to_bz(qimj, bz, rlv, invrlv) : qimj,
                    ForwardDiff.gradient(bands[μ], qimj),
                    μ
                )
            else
                p = VirtualPatch(
                    bands[μ](qimj_rlb[1], qimj_rlb[2]),
                    umklapp ? map_to_bz(qimj, bz, rlv, invrlv) : qimj,
                    Interpolations.gradient(bands[μ], qimj_rlb[1], qimj_rlb[2]),
                    μ
                )
            end

            w123 = Weff_squared(grid[i], grid[m], grid[j], p; kwargs)
            w124 = Weff_squared(grid[i], grid[m], p, grid[j]; kwargs)

            if w123 + w124 != 0
                Lij -= (w123 + w124) * ee_kernel!(ζ, u, grid[i], grid[m], grid[j], p, T) * f0s[m] * (1 - f0s[j])
            end
        end
    end

    return π * Lij / (grid[i].dV * (1 - f0s[i]))
end

electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, bands, T::Real, Weff_squared, l::Lattice; umklapp = true, kwargs...) = electron_electron(grid, f0s, i, j, bands, T, Weff_squared, reciprocal_lattice_vectors(l), get_bz(l); umklapp, kwargs...)

"""
    electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, ε::Function, T::Real, Weff_squared, l::NoLattice; kwargs...)

Compute the element (`i`,`j`) of the linearized Boltzmann collision operator for electron electron scattering, assuming an isotropic Fermi surface.

Passing the singleton `NoLattice` object identifies that the FS is isotropic and that umklapp is ignored. For the output, an explicit factor of (2π)^-6 is included since the momenta sampled are not rescaled as in the lattice case. The dispersion `ε` is thus taken to be a function of the norm of momentum only. The vector `f0s` stores the value of the Fermi-Dirac distribution at each patch center and can be calculated independent of `i` and `j`. `Weff_squared` is a user defined function of four Patch variables that computes the effective spinless quasiparticle scattering vertex. Additional parameters needed to evaluate `Weff_squared` can be passed through as keyword arguments. 
"""
function electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, ε::Function, T::Real, Weff_squared, l::NoLattice; kwargs...) 
    Lij::Float64 = 0.0
    w123::Float64 = 0.0
    w124::Float64 = 0.0

    ζ  = MVector{6,Float64}(undef)
    u  = MVector{6,Float64}(undef)

    kij = grid[i].k + grid[j].k
    qij = grid[i].k - grid[j].k

    kijm = Vector{Float64}(undef, 2)
    qimj = Vector{Float64}(undef, 2)

    for m in eachindex(grid)
        kijm .= kij .- grid[m].k 

        p = VirtualPatch(
            ε(norm(kijm)),
            kijm,
            norm(kijm) != 0.0 ? ForwardDiff.derivative(ε, norm(kijm)) * kijm / norm(kijm) : [0.0, 0.0],
            1
        )
        
        w123 = Weff_squared(grid[i], grid[j], grid[m], p; kwargs...)

        if w123 != 0
            Lij += w123 * ee_kernel!(ζ, u, grid[i], grid[j], grid[m], p, T) * f0s[j] * (1 - f0s[m])
        end

        qimj .= qij .+ grid[m].k

        p = VirtualPatch(
            ε(norm(qimj)),
            qimj,
            norm(qimj) != 0.0 ? ForwardDiff.derivative(ε, norm(qimj)) * qimj / norm(qimj) : [0.0, 0.0],
            1
        )
        w123 = Weff_squared(grid[i], grid[m], grid[j], p; kwargs...)
        w124 = Weff_squared(grid[i], grid[m], p, grid[j]; kwargs...)

        if w123 + w124 != 0
            Lij -= (w123 + w124) * ee_kernel!(ζ, u, grid[i], grid[m], grid[j], p, T) * f0s[m] * (1 - f0s[j])
        end
    end

    return π * Lij / (grid[i].dV * (1 - f0s[i])) / (2π)^6
end

"""
<<<<<<< HEAD
    electron_phonon(a::Patch, b::Patch, T::Real, g, ω, rlv, bz; kwargs...)

Compute the element ``(a, b)`` of the linearized Boltzmann collision operator for
electron-phonon scattering.

`g(k_b, k_a, q, ω0; kwargs...)` is the electron-phonon coupling matrix element, where
`q = k_b - k_a` is the phonon momentum and `ω0 = ω(q)` is the phonon frequency at that
momentum. `ω` must be differentiable via ForwardDiff.

Returns the scattering rate ``L_{ab}`` (same index convention and units as
[`electron_electron`](@ref)).
"""
function electron_phonon(a::Patch, b::Patch, T::Real, g, ω, rlv, bz; kwargs...)
    if abs(a.e - b.e) < a.de / 2
        return 0.0
    end
    R4_squared = 4*sqrt(2)/π
=======
    ep_kernel!(ζ, u, a::Patch, b::Patch, v, ω0, T)
>>>>>>> f95fd96 (Standardize argument style for collision operator functions.)

Compute the geometric kernel for electron-phonon scattering between patches `a` and `b`,
given phonon group velocity `v`, phonon frequency `ω0`, and temperature `T`. Uses
pre-allocated 4-element buffers `ζ` and `u` to avoid allocations.
"""
function ep_kernel!(ζ, u, a::Patch, b::Patch, v, ω0::Real, T::Real)
    ζ[1] = -(v[1] * a.jinv[1,1] + v[2] * a.jinv[2,1])
    ζ[2] = -(v[1] * a.jinv[1,2] + v[2] * a.jinv[2,2])
    ζ[3] = v[1] * b.jinv[1,1] + v[2] * b.jinv[2,1]
    ζ[4] = v[1] * b.jinv[1,2] + v[2] * b.jinv[2,2]

    Kij = 0.0
    for sign ∈ (-1, 1)
        δ = b.e - a.e + sign * ω0

        u[1] = -a.de/2.0 + sign * ζ[1]
        u[2] = sign * ζ[2]
        u[3] = b.de/2.0 + sign * ζ[3]
        u[4] = sign * ζ[4]

        ρ3 < δ^2 / dot(u, u) && continue # Check for intersection of energy conserving 3-plane with coordinate space
        ω_eff = ω0 - δ * dot(ζ, u) / dot(u, u)
        V = (4π^3 / 3) * (ρ3 - δ^2 / dot(u, u))^(3/2)

        Kij -= sign * V / (2 * cosh(ω_eff / T) - 2) / norm(u)
    end

    return Kij
end

"""
    electron_phonon(grid::Vector{Patch}, i::Int, j::Int, T::Real, g, ω, rlv, bz; kwargs...)

Compute the element (`i`,`j`) of the linearized Boltzmann collision operator for
electron-phonon scattering. `g(k_b, k_a, q, ω0)` is the electron-phonon coupling and
`ω(q)` is the phonon dispersion.
"""
function electron_phonon(grid::Vector{Patch}, i::Int, j::Int, T::Real, g, ω, rlv, bz; kwargs...)
    a = grid[i]
    b = grid[j]

    abs(a.e - b.e) < a.de / 2 && return 0.0

    invrlv = inv(rlv)
    q = map_to_bz(b.k - a.k, bz, rlv, invrlv)
    ω0 = ω(q)
    gij2 = abs(g(b.k, a.k, q, ω0; kwargs...))^2

    v = ForwardDiff.gradient(ω, q)
    ζ = MVector{4,Float64}(undef)
    u = MVector{4,Float64}(undef)

    f0i = f0(a.e, T)
    f0j = f0(b.e, T)
    return 4π * ep_kernel!(ζ, u, a, b, v, ω0, T) * gij2 * (f0j - f0i) * a.djinv * b.djinv / (a.dV * f0i * (1 - f0i))
end

electron_phonon(grid::Vector{Patch}, i::Int, j::Int, T::Real, g, ω, l::Lattice; kwargs...) = electron_phonon(grid, i, j, T, g, ω, reciprocal_lattice_vectors(l), get_bz(l); kwargs...)

function exact_volume(u, b, scale = 1)
    vertices = (map(x -> SVector{length(u), UInt8}(digits(x, base=2, pad = length(u))), 0:2^(length(u))-1))
    nonzero_indices = MVector{length(u), Int}(undef)
    α = 2 * u / scale # Normalize by scale to reduce numerical errors in the product of α
    prod_α::Float64 = 1.0
    d::Int = 0 # Number of nonzero elements of α
    for i in eachindex(u)
        if abs(u[i] / scale) > 1e-2 
            nonzero_indices[i] = 1 
            prod_α *= α[i]
            d += 1
        else
            nonzero_indices[i] = 0
        end
    end

    β = (- b + sum(u)) / scale

    if d > 1
        volume = 0.0
        for v in vertices
            σ = 0
            αv = 0.0 
            for i in eachindex(u)
                if nonzero_indices[i] == 1
                    σ += v[i]
                    αv += α[i]*v[i]
                end
            end
            if αv ≤ β
                if iseven(σ)
                    volume += (β - αv)^(d-1)
                else
                    volume -= (β - αv)^(d-1)
                end 
            end 
        end
        
        volume *= (2^d / factorial(d - 1)) / (scale * prod_α) # Really, volume / norm(u)
        volume = max(0.0, volume) # Set to zero if small negative error accumulated from sign oscillation

        return volume 
    elseif d == 1
        # Then, prod_α == α_i, the nonzero element
        if 0 ≤ β/prod_α ≤ 1
            return 2^(length(u) - 1)
        else
            return 0.0
        end
    else
        return 0.0
    end
end

"""
    Iab(a, b, V_squared)

Compute the scattering amplitude for an electron scattering from patch `a` to patch `b` with `V_squared` given by
```math
    | \\langle \\mathbf{k}_a | \\hat{V} | \\mathbf{k}_b \\rangle |^2 = |V(\\mathbf{k}_a, \\mathbf{k}_b)|^2.
```
"""
function Iab(a::Patch, b::Patch, V_squared::Function)
    Δε = (a.de + b.de) / 2 # Everage of energy differentials yields ||u||
    if abs(a.e - b.e) < a.de/2 # Check for being between the same energy contours
        return 16 * V_squared(a.k, b.k) * a.djinv * b.djinv / Δε
    else
        return 0.0
    end
end

"""
    electron_impurity(grid::Vector{Patch}, i::Int, j::Int, V_squared)

Return the `(i, j)` element of the collision matrix for electron-impurity scattering on
`grid`, given the scattering potential `V_squared(k_a, k_b)`.
"""
function electron_impurity(grid::Vector{Patch}, i::Int, j::Int, V_squared)
    i == j && return 0.0
    return -Iab(grid[i], grid[j], V_squared) * 2π / grid[i].dV
end

"""
    electron_impurity(grid::Vector{Patch}, i::Int, j::Int, V_squared::Matrix)

Return the `(i, j)` element of the collision matrix for electron-impurity scattering,
assuming a constant scattering strength for each band pair given in `V_squared`. It is
assumed that `grid` is band-ordered and that the length of each band sector is the same.
The ordering of `V_squared` must match the band ordering.
"""
function electron_impurity(grid::Vector{Patch}, i::Int, j::Int, V_squared::Matrix)
    i == j && return 0.0
    ℓ = length(grid) ÷ size(V_squared, 1)
    return -Iab(grid[i], grid[j], (x,y) -> V_squared[(i-1)÷ℓ + 1, (j-1)÷ℓ + 1]) * 2π / grid[i].dV
end

"""
    fill_from_ibz!(L::AbstractMatrix, symmetry_map::BZSymmetryMap)

Given that the rows `symmetry_map.ibz_inds` of `L` have already been populated (e.g. via
[`electron_electron`](@ref) for each `i ∈ symmetry_map.ibz_inds`), fill all remaining rows
using the point-group invariance `L[O*i, O*j] = L[i, j]`.
"""
function fill_from_ibz!(L::AbstractMatrix, symmetry_map::BZSymmetryMap)
    for (i_bz, i_ibz) in symmetry_map.ibz_preimage
        g = symmetry_map.ibz_g_idx[i_bz]
        L[i_bz, :] = L[i_ibz, symmetry_map.g_inv_perms[g]]
    end
    return nothing
end

end
