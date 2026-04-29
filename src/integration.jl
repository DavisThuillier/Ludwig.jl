
"Squared cutoff radius for the hyperspherical approximation to the 5D phase-space volume in [`ee_kernel!`](@ref)."
const ρ::Float64 = 4*6^(1/3)/pi

"Squared cutoff radius for the hyperspherical approximation to the 3D phase-space volume in `electron_phonon`."
const ρ3::Float64 = 4*sqrt(2)/pi

"Volume of the 5-dimensional unit sphere"
const vol::Float64 = (8 * pi^2 / 15)

###
### Internal helpers
###

"""
    fill_kernel_vectors!(ζ, u, a, b, c, v, εabc)

Fill the 6-element kernel vectors `ζ` and `u` in-place and return `δ`.

`ζ` contains the group velocity `v` projected onto the Jacobian columns of each patch:

```math
\\zeta = \\begin{pmatrix} v^\\top J_a^{-1} \\\\ v^\\top J_b^{-1} \\\\ -v^\\top J_c^{-1} \\end{pmatrix}
```

`u` is the energy-coordinate gradient vector:

```math
u = \\begin{pmatrix} \\Delta\\varepsilon_a/2 \\\\ 0 \\\\ \\Delta\\varepsilon_b/2 \\\\ 0 \\\\ -\\Delta\\varepsilon_c/2 \\\\ 0 \\end{pmatrix} - \\zeta
```

`δ = a.e + b.e - c.e - εabc` is the energy mismatch at the patch centers.
"""
function fill_kernel_vectors!(ζ, u, a::Patch, b::Patch, c::Patch, v, εabc)
    δ = a.e + b.e - c.e - εabc
    ζ[1] = v[1] * a.jinv[1,1] + v[2] * a.jinv[2,1]
    ζ[2] = v[1] * a.jinv[1,2] + v[2] * a.jinv[2,2]
    ζ[3] = v[1] * b.jinv[1,1] + v[2] * b.jinv[2,1]
    ζ[4] = v[1] * b.jinv[1,2] + v[2] * b.jinv[2,2]
    ζ[5] = -(v[1] * c.jinv[1,1] + v[2] * c.jinv[2,1])
    ζ[6] = -(v[1] * c.jinv[1,2] + v[2] * c.jinv[2,2])
    u[1] = a.de/2.0 - ζ[1]
    u[2] = -ζ[2]
    u[3] = b.de/2.0 - ζ[3]
    u[4] = -ζ[4]
    u[5] = (-c.de/2.0) - ζ[5]
    u[6] = -ζ[6]
    return δ
end

"""
    pournin_inner(u_compact, b, ::Val{D})

Evaluate the Gray-code alternating sum over the 2^D vertices of `[0,1]^D`.

`u_compact` holds the D nonzero (normalized) components of u; `b` is the normalized
right-hand side. For each Gray-code vertex `v`, accumulates
`(-1)^popcount(v) * (b - u⋅v)^{D-1}` whenever `u⋅v ≤ b`.

All vertex signs, bit-flip indices, and the power exponent are baked in as compile-time
literals, leaving only float arithmetic and array reads in the emitted machine code.

See also [`pournin_volume`](@ref).
"""
@generated function pournin_inner(u_compact, b, ::Val{D}) where D
    dm1   = D - 1
    exprs = Expr[]
    push!(exprs, :(uv     = 0.0))
    push!(exprs, :(volume = 0.0))
    for k in 0:2^D - 1
        g_k    = k ⊻ (k >> 1)
        sign_k = iseven(count_ones(g_k)) ? 1.0 : -1.0
        vol_ex = dm1 == 0 ? :($sign_k) : :($sign_k * (b - uv)^$dm1)
        push!(exprs, :(if uv ≤ b; volume += $vol_ex; end))
        if k < 2^D - 1
            bit_k = trailing_zeros(k + 1)
            dir_k = (g_k >> bit_k) & 1
            push!(exprs, dir_k == 0 ?
                :(@inbounds uv += u_compact[$(bit_k + 1)]) :
                :(@inbounds uv -= u_compact[$(bit_k + 1)]))
        end
    end
    push!(exprs, :(return volume))
    return Expr(:block, exprs...)
end

"""
    pournin_volume(u, b)

Compute the ``(n-1)``-dimensional volume of ``\\{x \\in [0,1]^n : u \\cdot x = b\\}`` using
Theorem 2.2 from Pournin (2021). The formula is a finite sum over the vertices of
``[0,1]^n`` that lie on the same side of the hyperplane as the origin:

```math
\\sum_{v:\\, u \\cdot v \\leq b} (-1)^{\\sigma(v)} \\frac{\\|u\\| (b - u \\cdot v)^{d-1}}{(d-1)!\\, \\pi(u)}
```

where ``\\sigma(v) = \\sum_i v_i``, ``\\pi(u) = \\prod_i u_i`` (nonzero components only),
and ``d = \\#\\{i : u_i \\neq 0\\}``. Dimensions with ``u_i = 0`` are free and contribute a
unit factor to the volume.
"""
function pournin_volume(u::StaticVector{6}, b)
    norm_u_sq = sum(abs2, u)
    norm_u    = sqrt(norm_u_sq)
    threshold = 1e-10 * (norm_u + 1)

    # Collect nonzero components; normalize to keep (b - u⋅v)^{d-1} O(1).
    u_compact = MVector{6, Float64}(undef)
    d      = 0
    prod_u = 1.0
    for i in 1:6
        ui = Float64(u[i])
        if abs(ui) > threshold
            d += 1
            @inbounds u_compact[d] = ui / norm_u
            prod_u *= ui
        end
    end

    d == 0 && return 0.0

    b_norm    = b / norm_u
    inv_scale = norm_u^d / (Float64(factorial(d - 1)) * prod_u)

    # Dispatch to Val{d}-specialised @generated inner loop.
    raw = if d == 1;     pournin_inner(u_compact, b_norm, Val(1))
          elseif d == 2; pournin_inner(u_compact, b_norm, Val(2))
          elseif d == 3; pournin_inner(u_compact, b_norm, Val(3))
          elseif d == 4; pournin_inner(u_compact, b_norm, Val(4))
          elseif d == 5; pournin_inner(u_compact, b_norm, Val(5))
          else           pournin_inner(u_compact, b_norm, Val(6))
          end

    return max(0.0, raw * inv_scale)
end

###
### Scattering kernel
###

"""
    ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, v, εabc, T, exact=true)

Compute the integral

```math
\\mathcal{K}_{abc} \\equiv \\int_a \\int_b \\int_c (1 - f^{(0)}(\\varepsilon_{abc})) \\,
    \\delta(\\varepsilon_a + \\varepsilon_b - \\varepsilon_c - \\varepsilon(\\mathbf{k}_a + \\mathbf{k}_b - \\mathbf{k}_c))
```

with group velocity `v` and energy `εabc` at temperature `T`, where

```math
\\int_i \\equiv \\frac{1}{a^2} \\int_{\\mathbf{k} \\in \\mathcal{P}_i} d^2\\mathbf{k}
```

is an integral over momenta in patch ``\\mathcal{P}_i``.

When `exact = true` (default), the 5-dimensional intersection volume is computed exactly
via [`pournin_volume`](@ref): the energy-conserving hyperplane `u·x = -δ` on `[-1,1]^6`
is mapped to `u·y = (Σuᵢ - δ)/2` on `[0,1]^6` via `x = 2y - 1`, and the 5D surface
measure scales by `2^5 = 32`.

When `exact = false`, the volume is replaced by a hyperspherical approximation
``\\rho^{5/2}_{5} \\|u\\|^{-1}``, where ``\\rho_5`` is the 5D sphere radius implied by
the energy constraint.
"""
function ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, v, εabc, T, exact::Bool = true)
    δ = fill_kernel_vectors!(ζ, u, a, b, c, v, εabc)
    uu = dot(u, u)
    uu == 0.0 && return 0.0

    Δε = -δ * dot(ζ, u) / uu

    if exact
        b_pournin = ((u[1] + u[2] + u[3] + u[4] + u[5] + u[6]) - δ) / 2
        vol_5 = 32.0 * pournin_volume(u, b_pournin)
        vol_5 == 0.0 && return 0.0

        return vol_5 * a.djinv * b.djinv * c.djinv * (1 - f0(εabc + Δε, T)) / sqrt(uu)
    else
        ρ < δ^2 / uu && return 0.0

        r5 = (ρ - δ^2 / uu)^(5/2)
        return vol * a.djinv * b.djinv * c.djinv * r5 * (1 - f0(εabc + Δε, T)) / sqrt(uu)
    end
end

ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, d::VirtualPatch, T::Real) = ee_kernel!(ζ, u, a, b, c, d.v, d.e, T)
ee_kernel!(ζ, u, a::Patch, b::Patch, c::Patch, d::VirtualPatch, T::Real, exact::Bool) = ee_kernel!(ζ, u, a, b, c, d.v, d.e, T, exact)

###
### Generic Scattering Vertex
###

"""
    electron_electron(grid, f0s, i, j, bands, T, Weff_squared, rlv, bz;
                      umklapp=true, exact=true, kwargs...)

Compute the element (`i`, `j`) of the linearized Boltzmann collision operator for
electron-electron scattering.

# Arguments
- `grid::Vector{Patch}`: patches over which the collision operator is defined.
- `f0s::Vector{Float64}`: Fermi-Dirac distribution evaluated at each patch center
  (independent of `i` and `j`, so it is precomputed by the caller).
- `i::Int`, `j::Int`: row and column indices of the matrix element to compute.
- `bands`: collection of dispersions used to construct `grid`; each entry is callable on
  a 2-component momentum.
- `T::Real`: temperature.
- `Weff_squared`: user-defined function `Weff_squared(p1, p2, p3, p4; kwargs...)` of four
  `Patch`/`VirtualPatch` arguments returning the effective spinless quasiparticle
  scattering vertex.
- `rlv::AbstractMatrix`: reciprocal lattice vector matrix.
- `bz`: first Brillouin zone polygon.

# Keyword Arguments
- `umklapp::Bool=true`: when `true`, intermediate momenta are folded back into `bz` via
  the reciprocal lattice.
- `exact::Bool=true`: when `true`, the 5D phase-space volume is computed exactly via
  [`pournin_volume`](@ref); otherwise the hyperspherical approximation in
  [`ee_kernel!`](@ref) is used.
- additional keyword arguments are forwarded verbatim to `Weff_squared`.
"""
function electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, bands, T::Real, Weff_squared, rlv, bz; umklapp = true, exact::Bool = true, kwargs...)
    Lij::Float64 = 0.0
    w123::Float64 = 0.0
    w124::Float64 = 0.0

    ζ  = MVector{6,Float64}(undef)
    u  = MVector{6,Float64}(undef)

    kij = grid[i].k + grid[j].k
    qij = grid[i].k - grid[j].k

    invrlv = inv(rlv)

    for m in eachindex(grid)
        kijm = kij - grid[m].k

        for μ in eachindex(bands)
            p = VirtualPatch(
                bands[μ](kijm),
                umklapp ? map_to_bz(kijm, bz, rlv, invrlv) : kijm,
                band_velocity(bands[μ], kijm),
                μ
            )
            w123 = Weff_squared(grid[i], grid[j], grid[m], p; kwargs...)

            if w123 != 0
                Lij += w123 * ee_kernel!(ζ, u, grid[i], grid[j], grid[m], p, T, exact) * f0s[j] * (1 - f0s[m])
            end
        end

        qimj = qij + grid[m].k

        for μ in eachindex(bands)
            p = VirtualPatch(
                bands[μ](qimj),
                umklapp ? map_to_bz(qimj, bz, rlv, invrlv) : qimj,
                band_velocity(bands[μ], qimj),
                μ
            )

            w123 = Weff_squared(grid[i], grid[m], grid[j], p; kwargs...)
            w124 = Weff_squared(grid[i], grid[m], p, grid[j]; kwargs...)

            if w123 + w124 != 0
                Lij -= (w123 + w124) * ee_kernel!(ζ, u, grid[i], grid[m], grid[j], p, T, exact) * f0s[m] * (1 - f0s[j])
            end
        end
    end

    return π * Lij / (grid[i].dV * (1 - f0s[i]) * (2π)^6)
end

electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, bands, T::Real, Weff_squared, l::Lattice; umklapp = true, kwargs...) = electron_electron(grid, f0s, i, j, bands, T, Weff_squared, reciprocal_lattice_vectors(l), get_bz(l); umklapp, kwargs...)

"""
    electron_electron(grid, f0s, i, j, ε, T, Weff_squared, l::NoLattice;
                      exact=true, kwargs...)

Compute the element (`i`, `j`) of the linearized Boltzmann collision operator for
electron-electron scattering on an isotropic Fermi surface.

The singleton [`NoLattice`](@ref) identifies that the FS is isotropic and umklapp is
ignored; `ε` is taken to be a function of the norm of momentum only.

# Arguments
- `grid::Vector{Patch}`: patches over which the collision operator is defined.
- `f0s::Vector{Float64}`: Fermi-Dirac distribution evaluated at each patch center.
- `i::Int`, `j::Int`: row and column indices of the matrix element to compute.
- `ε::Function`: radial dispersion; called as `ε(norm(k))`.
- `T::Real`: temperature.
- `Weff_squared`: user-defined function of four `Patch`/`VirtualPatch` arguments
  returning the effective spinless quasiparticle scattering vertex.
- `l::NoLattice`: marker selecting the isotropic dispatch.

# Keyword Arguments
- `exact::Bool=true`: when `true`, the 5D phase-space volume is computed exactly via
  [`pournin_volume`](@ref); otherwise the hyperspherical approximation in
  [`ee_kernel!`](@ref) is used.
- additional keyword arguments are forwarded verbatim to `Weff_squared`.
"""
function electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, ε::Function, T::Real, Weff_squared, l::NoLattice; exact::Bool = true, kwargs...)
    Lij::Float64 = 0.0
    w123::Float64 = 0.0
    w124::Float64 = 0.0

    ζ  = MVector{6,Float64}(undef)
    u  = MVector{6,Float64}(undef)

    kij = grid[i].k + grid[j].k
    qij = grid[i].k - grid[j].k

    for m in eachindex(grid)
        kijm = kij - grid[m].k
        nkijm = norm(kijm)

        p = VirtualPatch(
            ε(nkijm),
            kijm,
            nkijm != 0.0 ? ForwardDiff.derivative(ε, nkijm) * kijm / nkijm : zero(SVector{2,Float64}),
            1
        )

        w123 = Weff_squared(grid[i], grid[j], grid[m], p; kwargs...)

        if w123 != 0
            Lij += w123 * ee_kernel!(ζ, u, grid[i], grid[j], grid[m], p, T, exact) * f0s[j] * (1 - f0s[m])
        end

        qimj = qij + grid[m].k
        nqimj = norm(qimj)

        p = VirtualPatch(
            ε(nqimj),
            qimj,
            nqimj != 0.0 ? ForwardDiff.derivative(ε, nqimj) * qimj / nqimj : zero(SVector{2,Float64}),
            1
        )
        w123 = Weff_squared(grid[i], grid[m], grid[j], p; kwargs...)
        w124 = Weff_squared(grid[i], grid[m], p, grid[j]; kwargs...)

        if w123 + w124 != 0
            Lij -= (w123 + w124) * ee_kernel!(ζ, u, grid[i], grid[m], grid[j], p, T, exact) * f0s[m] * (1 - f0s[j])
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
    return 4π * ep_kernel!(ζ, u, a, b, v, ω0, T) * gij2 * (f0j - f0i) * a.djinv * b.djinv / (a.dV * f0i * (1 - f0i) * (2π)^4)
end

electron_phonon(grid::Vector{Patch}, i::Int, j::Int, T::Real, g, ω, l::Lattice; kwargs...) = electron_phonon(grid, i, j, T, g, ω, reciprocal_lattice_vectors(l), get_bz(l); kwargs...)

"""
    Iab(a, b, V_squared)

Compute the scattering amplitude for an electron scattering from patch `a` to patch `b` with `V_squared` given by
```math
    | \\langle \\mathbf{k}_a | \\hat{V} | \\mathbf{k}_b \\rangle |^2 = |V(\\mathbf{k}_a, \\mathbf{k}_b)|^2.
```
"""
function Iab(a::Patch, b::Patch, V_squared::Function)
    Δε = (a.de + b.de) / 2 # Average of energy differentials yields ||u||
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
    return -Iab(grid[i], grid[j], V_squared) * 2π / (grid[i].dV * (2π)^4)
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
    return -Iab(grid[i], grid[j], (x,y) -> V_squared[(i-1)÷ℓ + 1, (j-1)÷ℓ + 1]) * 2π / (grid[i].dV * (2π)^4)
end
