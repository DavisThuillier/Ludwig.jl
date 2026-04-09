
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

"""
    ee_kernel!(ζ, u, a, b, c, v, εabc, T, ::Val{true})

Exact variant of `ee_kernel!` that replaces the hyperspherical volume approximation with the
exact 5-dimensional volume of the intersection of the energy-conserving hyperplane
`u·x = -δ` with the hypercube `[-1,1]^6`, computed via [`pournin_volume`](@ref).

The coordinate change `x = 2y - 1` (mapping `[-1,1]^6 → [0,1]^6`) transforms the
constraint to `u·y = (Σuᵢ - δ)/2`, and the 5D surface measure scales by `2^5 = 32`.
"""
function ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, v, εabc, T, ::Val{true})
    δ = a.e + b.e - c.e - εabc

    ζ[1] = v[1] * a.jinv[1,1] + v[2] * a.jinv[2,1]
    ζ[2] = v[1] * a.jinv[1,2] + v[2] * a.jinv[2,2]
    ζ[3] = v[1] * b.jinv[1,1] + v[2] * b.jinv[2,1]
    ζ[4] = v[1] * b.jinv[1,2] + v[2] * b.jinv[2,2]
    ζ[5] = - (v[1] * c.jinv[1,1] + v[2] * c.jinv[2,1])
    ζ[6] = - (v[1] * c.jinv[1,2] + v[2] * c.jinv[2,2])

    u[1] = a.de/2.0 - ζ[1]
    u[2] = - ζ[2]
    u[3] = b.de/2.0 - ζ[3]
    u[4] = - ζ[4]
    u[5] = (- c.de/2.0) - ζ[5]
    u[6] = - ζ[6]

    uu = dot(u, u)
    uu == 0.0 && return 0.0

    # Map the constraint u·x = -δ on [-1,1]^6 to u·y = b on [0,1]^6 via x = 2y - 1.
    # The 5D surface measure scales by 2^5 = 32.
    b_pournin = ((u[1] + u[2] + u[3] + u[4] + u[5] + u[6]) - δ) / 2
    vol_5 = 32.0 * pournin_volume(SVector{6,Float64}(u[1], u[2], u[3], u[4], u[5], u[6]), b_pournin)
    vol_5 == 0.0 && return 0.0

    Δε = - δ * dot(ζ, u) / uu
    return vol_5 * a.djinv * b.djinv * c.djinv * (1 - f0(εabc + Δε, T)) / sqrt(uu)
end

ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, v, εabc, T, ::Val{false}) =
    ee_kernel!(ζ, u, a, b, c, v, εabc, T)

function ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, k, εabc, ε::Function, T::Real)
    v::SVector{2,Float64} = ForwardDiff.gradient(ε, k)
    return ee_kernel!(ζ, u, a, b, c, v, εabc, T)
end

function ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, k, εabc, itp::ScaledInterpolation, T::Real)
    v::SVector{2,Float64} = Interpolations.gradient(itp, k[1], k[2])
    return ee_kernel!(ζ, u, a, b, c, v, εabc, T)
end

ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, d::VirtualPatch, T::Real) = ee_kernel!(ζ, u, a, b, c, d.v, d.e, T)
ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, d::VirtualPatch, T::Real, exact::Val) = ee_kernel!(ζ, u, a, b, c, d.v, d.e, T, exact)

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
function electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, bands, T::Real, Weff_squared, rlv, bz; umklapp = true, exact::Bool = false, kwargs...)
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
    exact_val = Val(exact)

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
                Lij += w123 * ee_kernel!(ζ, u, grid[i], grid[j], grid[m], p, T, exact_val) * f0s[j] * (1 - f0s[m])
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
                Lij -= (w123 + w124) * ee_kernel!(ζ, u, grid[i], grid[m], grid[j], p, T, exact_val) * f0s[m] * (1 - f0s[j])
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
function electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, ε::Function, T::Real, Weff_squared, l::NoLattice; exact::Bool = false, kwargs...)
    Lij::Float64 = 0.0
    w123::Float64 = 0.0
    w124::Float64 = 0.0

    ζ  = MVector{6,Float64}(undef)
    u  = MVector{6,Float64}(undef)

    kij = grid[i].k + grid[j].k
    qij = grid[i].k - grid[j].k

    kijm = Vector{Float64}(undef, 2)
    qimj = Vector{Float64}(undef, 2)

    exact_val = Val(exact)

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
            Lij += w123 * ee_kernel!(ζ, u, grid[i], grid[j], grid[m], p, T, exact_val) * f0s[j] * (1 - f0s[m])
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
            Lij -= (w123 + w124) * ee_kernel!(ζ, u, grid[i], grid[m], grid[j], p, T, exact_val) * f0s[m] * (1 - f0s[j])
        end
    end

    return π * Lij / (grid[i].dV * (1 - f0s[i])) / (2π)^6
end

"""
    ep_kernel!(ζ, u, a::Patch, b::Patch, v, ω0, T)

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

"""
    pournin_volume(u, b)

Compute the (n-1)-dimensional volume of `{x ∈ [0,1]^n : u⋅x = b}` using Theorem 2.2
from Pournin (2021). The formula is a finite sum over the vertices of [0,1]^n that lie
on the same side of the hyperplane as the origin:

    Σ (-1)^σ(v) ‖u‖ (b - u⋅v)^{d-1} / ((d-1)! π(u))

where the sum is over vertices v with u⋅v ≤ b, σ(v) = Σvᵢ, π(u) = Πuᵢ (nonzero
components only), and d = #{i : uᵢ ≠ 0}. Dimensions with uᵢ = 0 are free and
contribute a unit factor to the volume.
"""
function pournin_volume(u::SVector{6}, b)
    norm_u_sq = sum(abs2, u)
    norm_u    = sqrt(norm_u_sq)
    threshold = 1e-10 * (norm_u + 1)

    # Compact nonzero components into a stack-allocated buffer (no heap allocation)
    u_compact = MVector{6, Float64}(undef)
    d      = 0
    prod_u = 1.0
    for i in 1:6
        ui = Float64(u[i])
        if abs(ui) > threshold
            d += 1
            @inbounds u_compact[d] = ui / norm_u  # normalize: keeps (b_n - u_n⋅v)^{d-1} O(1)
            prod_u *= ui
        end
    end

    d == 0 && return 0.0

    # Scale-invariance of the formula: replacing a → a/‖a‖ and b → b/‖a‖ leaves
    # the volume unchanged. Normalizing u_compact to unit norm bounds each term
    # (b_n - u_n⋅v)^{d-1} and avoids catastrophic cancellation in the alternating sum.
    # The ‖a‖^d / π(a) factor is preserved by absorbing ‖a‖^d into inv_scale.
    b_norm    = b / norm_u
    inv_scale = norm_u^d / (Float64(factorial(d - 1)) * prod_u)

    # Dispatch on runtime d to a Val{d}-specialised inner function. Each branch
    # compiles to fully unrolled code with all vertex signs, bit-flip indices, and
    # the power exponent baked in as literals — no trailing_zeros, count_ones,
    # loop overhead, or last-iteration branch remain at runtime.
    raw = if d == 1;     _pournin_inner(u_compact, b_norm, Val(1))
          elseif d == 2; _pournin_inner(u_compact, b_norm, Val(2))
          elseif d == 3; _pournin_inner(u_compact, b_norm, Val(3))
          elseif d == 4; _pournin_inner(u_compact, b_norm, Val(4))
          elseif d == 5; _pournin_inner(u_compact, b_norm, Val(5))
          else           _pournin_inner(u_compact, b_norm, Val(6))
          end

    return max(0.0, raw * inv_scale)
end

# Generates fully unrolled Gray-code traversal code for exactly D nonzero components.
# At code-generation time: Gray(k), vertex sign (-1)^popcount(Gray(k)), the bit index
# that flips at each step, and the add/subtract direction are all computed as literals.
# Nothing but float arithmetic and array reads remain in the emitted machine code.
@generated function _pournin_inner(u_compact, b, ::Val{D}) where D
    dm1   = D - 1
    exprs = Expr[]
    push!(exprs, :(uv     = 0.0))
    push!(exprs, :(volume = 0.0))
    for k in 0:2^D - 1
        g_k    = k ⊻ (k >> 1)                          # Gray(k): compile-time constant
        sign_k = iseven(count_ones(g_k)) ? 1.0 : -1.0  # vertex sign: compile-time constant
        vol_ex = dm1 == 0 ? :($sign_k) : :($sign_k * (b - uv)^$dm1)
        push!(exprs, :(if uv ≤ b; volume += $vol_ex; end))
        if k < 2^D - 1
            bit_k = trailing_zeros(k + 1)               # bit that flips next: compile-time
            dir_k = (g_k >> bit_k) & 1                  # 0 = set (add), 1 = clear (sub)
            push!(exprs, dir_k == 0 ?
                :(@inbounds uv += u_compact[$(bit_k + 1)]) :
                :(@inbounds uv -= u_compact[$(bit_k + 1)]))
        end
    end
    push!(exprs, :(return volume))
    return Expr(:block, exprs...)
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
