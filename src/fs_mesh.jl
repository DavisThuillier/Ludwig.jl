abstract type AbstractPatch end

"""
    Patch(e::Float64, k::SVector{2,Float64}, v::SVector{2,Float64}, de::Float64, dV::Float64, jinv::Matrix{Float64}, djinv::Float64, band_index::Int)

Construct a `Patch` object defining regions of momentum space over which to integrate. 
# Fields
- `e`: energy
- `k`: momentum 
- `v`: group velocity
- `de`: width of patch in energy
- `dV`: area of patch in momentum space
- `jinv`: Jacobian of transformation from (kx, ky) --> (E, s)
- `djinv`: determinant of above Jacobian
- `band_index`: index of band from which `e` was sampled at `k`
"""
struct Patch <: AbstractPatch
    e::Float64
    k::SVector{2,Float64}
    v::SVector{2,Float64}
    de::Float64
    dV::Float64
    jinv::Matrix{Float64}
    djinv::Float64
    band_index::Int
end

"""
    VirtualPatch(e::Float64, k::SVector{2,Float64}, v::SVector{2,Float64}, band_index::Int)

Construct a `VirtualPatch` object that can be operated on as if it were a `Patch` for the purposes of sampling momentum, energy, and group velocity but which cannot be integrated over. 
# Fields
- `e`: energy
- `k`: momentum 
- `v`: group velocity
- `band_index`: index of band from which `e` was sampled at `k`
"""
struct VirtualPatch <: AbstractPatch
    e::Float64
    k::SVector{2,Float64} 
    v::SVector{2,Float64}
    band_index::Int
end

energy(p::AbstractPatch)   = p.e
momentum(p::AbstractPatch) = p.k
velocity(p::AbstractPatch) = p.v
band(p::AbstractPatch)     = p.band_index
area(p::Patch)             = p.dV

###
### HamiltonianBand
###

"""
    HamiltonianBand(H, n)

Wrap a matrix-valued Hamiltonian function `H(k)` for the `n`-th band.

`H` must return a square Hermitian (or real symmetric) matrix for each momentum
`k::AbstractVector`. The band energy is the `n`-th eigenvalue (in ascending order);
the group velocity is computed by the Hellmann-Feynman theorem using central finite
differences to differentiate `H`.

# Examples
```julia
function H(k)
    t = 1.0
    [0.0+0im  t*(exp(im*k[1]) + exp(im*k[2]));
     t*(exp(-im*k[1]) + exp(-im*k[2]))  0.0+0im]
end
band = HamiltonianBand(H, 1)  # lower band
```

See also [`bz_mesh`](@ref), [`ibz_mesh`](@ref).
"""
struct HamiltonianBand{F}
    H::F
    n::Int
end

(b::HamiltonianBand)(k) = real(eigvals(Hermitian(b.H(k)))[b.n])

"""
    band_velocity(band, k)

Return the group velocity of `band` at momentum `k`.

Dispatches on band type: plain functions use `ForwardDiff.gradient`; a
[`HamiltonianBand`](@ref) uses the Hellmann-Feynman theorem with central finite
differences applied to the Hamiltonian matrix.
"""
band_velocity(ε::Function, k) = gradient(ε, k)

function band_velocity(b::HamiltonianBand, k)
    F   = eigen(Hermitian(b.H(k)))
    ψ   = F.vectors[:, b.n]
    δ   = sqrt(eps(Float64))
    e1  = SVector{2,Float64}(1, 0)
    e2  = SVector{2,Float64}(0, 1)
    dHx = (b.H(k + δ*e1) - b.H(k - δ*e1)) / (2δ)
    dHy = (b.H(k + δ*e2) - b.H(k - δ*e2)) / (2δ)
    return SVector{2,Float64}(real(ψ'*dHx*ψ), real(ψ'*dHy*ψ))
end

"""
    patch_op(p::Patch, M::Matrix)

Apply the active transformation defined by matrix `M` on the relevant fields of patch `p`.
"""
function patch_op(p::Patch, M::Matrix)
    return Patch(
        p.e,
        SVector{2}(M * p.k), 
        SVector{2}(M * p.v),
        p.de,
        p.dV,
        M * p.jinv, 
        p.djinv,
        p.band_index
    )
end

function patch_op(p::VirtualPatch, M::Matrix)
    return VirtualPatch(
        p.energy,
        SVector{2}(M * p.momentum), 
        SVector{2}(M * p.v),
        band_index
    )
end

"""
    Mesh(patches::Vector{Patch}, corners::Vector{SVector{2,Float64}}, corner_inds::Vector{SVector{4,Int}})

Construct a container struct of patches which contains information for plotting functions defined on patch centers.
# Fields
- `patches`: Vector of patches
- `corners`: Vector of points on patch corners for plotting mesh
- `corner_inds`: Vector of vector of indices in the `corners` vector corresponding to the corners of each patch in `patches`
"""
struct Mesh
    patches::Vector{Patch}
    corners::Vector{SVector{2, Float64}} 
    corner_inds::Vector{SVector{4, Int}}
end

patches(m::Mesh) = m.patches
corners(m::Mesh) = m.corners
corner_indices(m::Mesh) = m.corner_inds

"""
    BZSymmetryMap

Store precomputed symmetry information for a BZ mesh, relating each non-IBZ patch to its IBZ
representative via a point-group operation. Built via [`bz_symmetry_map`](@ref).

# Fields
- `ibz_inds`: BZ grid indices of the IBZ patches
- `ibz_preimage`: maps each non-IBZ patch index to its IBZ representative index
- `g_perms`: `g_perms[s][j]` = BZ index of `O_s * grid[j]`, for each group element sector `s`
- `g_inv_perms`: inverse permutations; `g_inv_perms[s][j]` = BZ index of `O_s^{-1} * grid[j]`
- `ibz_g_idx`: maps each non-IBZ patch index to the sector index `s` of its group element
"""
struct BZSymmetryMap
    ibz_inds::Vector{Int}
    ibz_preimage::Dict{Int,Int}
    g_perms::Vector{Vector{Int}}
    g_inv_perms::Vector{Vector{Int}}
    ibz_g_idx::Dict{Int,Int}
end

"""
    bz_symmetry_map(grid::Vector{Patch}, l::Lattice) -> BZSymmetryMap

Build the symmetry map for a BZ mesh generated from lattice `l`. All maps are constructed
by matching momenta in `grid` directly, so the result is independent of the internal
ordering convention used by [`bz_mesh`](@ref).

The returned [`BZSymmetryMap`](@ref) can be passed to [`fill_from_ibz!`](@ref) to fill
the non-IBZ rows of a collision matrix after computing only the IBZ rows.
"""
function bz_symmetry_map(grid::Vector{Patch}, l::Lattice)
    G   = point_group(l)
    ibz = get_ibz(l)
    N   = length(grid)
    G_order = length(G.elements)

    θperm = group_angle_perm(G, ibz)

    # Build momentum → grid index dictionary (robust: does not depend on ordering)
    momentum_to_idx = Dict(round.(p.k, digits = 10) => i for (i, p) in enumerate(grid))

    # Identify IBZ patches by polygon membership
    ibz_inds = filter(i -> in_polygon(grid[i].k, ibz), 1:N)

    # Build permutation vectors for each group element
    g_perms = Vector{Vector{Int}}(undef, G_order)
    for (s, g_idx) in enumerate(θperm)
        O = get_matrix_representation(G.elements[g_idx])
        g_perms[s] = [momentum_to_idx[round.(O * grid[j].k, digits = 10)] for j in 1:N]
    end
    g_inv_perms = [invperm(perm) for perm in g_perms]

    # Find the identity sector (the sector whose permutation is the identity)
    ibz_sector = findfirst(s -> g_perms[s] == collect(1:N), 1:G_order)

    # For each non-identity sector, map BZ patches to their IBZ representatives
    ibz_preimage = Dict{Int,Int}()
    ibz_g_idx    = Dict{Int,Int}()
    for s in 1:G_order
        s == ibz_sector && continue
        for k in ibz_inds
            i_bz = g_perms[s][k]
            ibz_preimage[i_bz] = k
            ibz_g_idx[i_bz]    = s
        end
    end

    return BZSymmetryMap(ibz_inds, ibz_preimage, g_perms, g_inv_perms, ibz_g_idx)
end

###
### General IBZ Meshes
###

"""
    sort_isolines!(bundle::IsolineBundle)

Sort the isolines in `bundle` by centroid distance from the origin (ascending), so that
inner contours have smaller indices than outer contours. Returns `bundle`.
"""
function sort_isolines!(bundle::IsolineBundle)
    sort!(bundle.isolines, by = iso -> norm(sum(iso.points) / length(iso.points)))
    return bundle
end

"""
    split_bouncing_isolines!(bundle, region[; tol])

Split any open isoline in `bundle` that touches an interior IBZ boundary point.

An open isoline "bounces" when a non-endpoint interior point lies on an edge of `region`.
Such a point divides the contour into two geometrically distinct pieces that marching
squares incorrectly joined. Each bounce point becomes an endpoint of the two resulting
isolines, which replace the original in `bundle`. The bundle is re-sorted by centroid
distance after any splits.
"""
function split_bouncing_isolines!(bundle::IsolineBundle, region; tol = 1e-8)
    new_isolines = Isoline[]
    changed = false

    for iso in bundle.isolines
        if iso.isclosed || length(iso.points) < 3
            push!(new_isolines, iso)
            continue
        end

        pts = iso.points
        bounce_inds = Int[]
        for i in 2:length(pts)-1
            edge_index(pts[i], region; tol) != 0 && push!(bounce_inds, i)
        end

        if isempty(bounce_inds)
            push!(new_isolines, iso)
        else
            changed = true
            arcs = get_arclengths(pts)
            split_inds = [1; bounce_inds; length(pts)]
            for k in 1:length(split_inds)-1
                a, b = split_inds[k], split_inds[k+1]
                seg = pts[a:b]
                push!(new_isolines, Isoline(seg, false, arcs[b] - arcs[a]))
            end
        end
    end

    changed || return bundle
    empty!(bundle.isolines)
    append!(bundle.isolines, new_isolines)
    sort_isolines!(bundle)
    return bundle
end

###
### foliated_energies
###

"""
    foliated_energies(X, Y, E, ε, region, Δε, α, T)

Compute the interleaved vector of boundary and center energy levels for meshing.

Builds a symmetric grid of boundary contour energies centered at zero with spacing `Δε`,
spanning `[-α*T, α*T]` (always an even number of levels so that `E = 0` falls on a patch
center). Grid points outside the range of values in `E` restricted to `region` are discarded,
except the first out-of-range point on each side is retained and pinned to the second-lowest
or second-highest distinct energy value within `region`. Using the second extremum rather than
the true extremum ensures the boundary contour level is straddled by at least a full ring of
grid cells, avoiding the degenerate single-point isoline that arises when the extremum is
attained at only one grid point. If the dispersion evaluated at a corner of `region` falls
within the retained range, the levels are shifted so that one boundary contour passes exactly
through that energy.

Returns a vector of length `2*n - 1`, where `n` is the number of retained boundary
contours. Odd-indexed entries are boundary contour energies; even-indexed entries are
patch center energies (midpoints between adjacent boundaries).
"""
function foliated_energies(X, Y, E, ε, region, Δε, α, T)
    n_levels = 2 * max(1, ceil(Int, α * T / Δε)) # always even → E=0 on a patch center
    energies = collect(LinRange(-α * T, α * T, n_levels))

    e_actual_min, e_actual_max = if length(region) > 2
        lo, hi = Inf, -Inf
        for (i, x) in enumerate(X), (j, y) in enumerate(Y)
            isnan(E[i, j]) && continue
            in_polygon([x, y], region) || continue
            e = E[i, j]
            e < lo && (lo = e)
            e > hi && (hi = e)
        end
        if isinf(lo)
            E_valid = E[.!isnan.(E)]
            minimum(E_valid), maximum(E_valid)
        else
            lo, hi
        end
    else
        E_valid = E[.!isnan.(E)]
        minimum(E_valid), maximum(E_valid)
    end

    # Pin the first out-of-range point on the lower side to the actual minimum
    i_low = findlast(<(e_actual_min), energies)
    if i_low !== nothing
        energies[i_low] = e_actual_min
        energies = energies[i_low:end]
    end

    # Pin the first out-of-range point on the upper side to the actual maximum
    i_high = findfirst(>(e_actual_max), energies)
    if i_high !== nothing
        energies[i_high] = e_actual_max
        energies = energies[1:i_high]
    end

    # # Shift energy levels to include a corner energy if the dispersion crosses it
    for k ∈ region
        k == [0.0, 0.0] && continue
        corner_e =  ε(k)
        if energies[begin] < corner_e < energies[end]
            i = argmin(abs.(energies .- corner_e))
            # i == 1 && continue # corner is below the first interior level; skip
            n = length(energies)
            energies[1:i] = LinRange(energies[begin], corner_e, i)
            Δe = (energies[end] - corner_e) / (n - i)
            energies[i+1:end] = LinRange(corner_e + Δe, energies[end], n - i)
        end
    end

    n = length(energies)
    foliated = Vector{Float64}(undef, 2 * n - 1)
    for i in eachindex(foliated)
        if isodd(i)
            foliated[i] = energies[(i - 1) ÷ 2 + 1]
        else
            foliated[i] = (energies[i ÷ 2 + 1] + energies[i ÷ 2]) / 2
        end
    end
    return foliated
end

"""
    aligned_arclength_slice(iso, ref_start, ref_tangent, n; cusp_fractions, angle_threshold)

Resample `iso` at `n` arclength-uniform points, aligned to `ref_start`.

The endpoint of `iso.points` closest to `ref_start` is chosen as the starting
point of the resampled arc. The traversal direction is then checked against
`ref_tangent` (the tangent of the reference contour at its start): if the dot
product of the adjacent arc's initial tangent with `ref_tangent` is negative, the
arc is reversed so that all contours in a sheet are traversed in the same direction.

# Arguments
- `cusp_fractions`: arclength fractions in [0, 1] (relative to the *original* traversal
  direction of `iso`) at which cuts should be snapped. Fractions are mirrored when the
  isoline is reversed. Any cusps intrinsic to this contour are detected and merged
  automatically.
- `angle_threshold`: turning-angle threshold (radians) passed to [`find_cusp_fraction`](@ref).
"""
function aligned_arclength_slice(iso::Isoline, ref_start, ref_tangent, n::Int;
                                  cusp_fractions = Float64[], angle_threshold = π/8)
    pts = iso.points

    reverse_iso = norm(pts[end] - ref_start) < norm(pts[begin] - ref_start)
    reverse_iso = reverse_iso || (length(pts) >= 2 && dot(pts[2] - pts[1], ref_tangent) < 0)
    if reverse_iso
        pts = reverse(pts)
    end
    fractions = reverse_iso ? (1.0 .- cusp_fractions) : copy(cusp_fractions)
    append!(fractions, find_cusp_fraction(pts; angle_threshold))
    L = get_arclengths(pts)[end]
    return arclength_slice(pts, n; pinned_arclengths = fractions .* L)[1]
end

"""
    mesh_sheet(sheet_isolines, ε, band_index, n_arc_s, foliated_energies, corner_offset)

Generate a [`Mesh`](@ref) for a single Fermi surface sheet defined by `sheet_isolines`.

`sheet_isolines[i]` is the isoline at `foliated_energies[i]` for this sheet (no `nothing`
entries — call site must validate). `n_arc_s` is the number of arc-length divisions of the
center contour. All corner indices in the returned `Mesh` are offset by `corner_offset` so
that the caller can concatenate multiple sheet meshes into a single flat `Mesh`.
"""
function mesh_sheet(
    sheet_isolines::Vector{<:Isoline},
    ε,
    band_index::Int,
    n_arc_s::Int,
    foliated_energies::Vector{Float64},
    corner_offset::Int;
    angle_threshold = π/8,
)
    n_levels = (length(foliated_energies) + 1) ÷ 2

    patches    = Matrix{Patch}(undef, n_levels - 1, n_arc_s)
    corners    = Vector{SVector{2, Float64}}(undef, (2 * n_levels - 2) * (n_arc_s + 1))
    corner_ids = Matrix{SVector{4, Int}}(undef, n_levels - 1, n_arc_s)

    cind = 1

    for i in 2:2:2*n_levels-1 # Loop over center energy indices
        arclength_fractions = LinRange(0, 1, 2 * n_arc_s + 1)

        cusp_fractions = mapreduce(
            iso -> find_cusp_fraction(iso.points; angle_threshold),
            vcat,
            (sheet_isolines[i-1], sheet_isolines[i], sheet_isolines[i+1]),
        )

        L_center = sheet_isolines[i].arclength
        k, arclengths = arclength_slice(
            sheet_isolines[i].points, 2 * n_arc_s + 1;
            pinned_arclengths = cusp_fractions .* L_center,
        )
        ref_tangent = k[2] - k[1]

        k_inner = aligned_arclength_slice(
            sheet_isolines[i-1], k[1], ref_tangent, 2 * n_arc_s + 1;
            cusp_fractions, angle_threshold,
        )
        k_outer = aligned_arclength_slice(
            sheet_isolines[i+1], k[1], ref_tangent, 2 * n_arc_s + 1;
            cusp_fractions, angle_threshold,
        )

        # Identify endpoints of contours as corners of first patch
        endpoints = [sheet_isolines[i-1].points[begin], sheet_isolines[i-1].points[end]]
        i3 = argmin(map(x -> norm(x .- k[1]), endpoints))
        ij3 = (i3, i3)
        corners[cind] = endpoints[i3]

        endpoints = [sheet_isolines[i+1].points[begin], sheet_isolines[i+1].points[end]]
        i4 = argmin(map(x -> norm(x .- k[1]), endpoints))
        ij4 = (i4, i4)
        corners[cind+1] = endpoints[i4]

        corners[cind]   = k_inner[1]
        corners[cind+1] = k_outer[1]
        cind += 2

        for j in 2:2:lastindex(k)
            corners[cind]   = k_inner[j+1]
            corners[cind+1] = k_outer[j+1]
            corner_ids[i÷2, j÷2] = SVector{4, Int}(cind-2, cind-1, cind+1, cind) .+ corner_offset
            cind += 2

            # Identify contour points bordering corners
            sp1 = sortperm(get_arclengths(sheet_isolines[i-1].points) / sheet_isolines[i-1].arclength; by = x -> abs(x - arclength_fractions[j-1]))
            ij1 = length(sp1) >= 2 ? sp1[1:2] : [sp1[1], sp1[1]]
            sp2 = sortperm(get_arclengths(sheet_isolines[i+1].points) / sheet_isolines[i+1].arclength; by = x -> abs(x - arclength_fractions[j+1]))
            ij2 = length(sp2) >= 2 ? sp2[1:2] : [sp2[1], sp2[1]]

            # Identify which if the points which border the corner on the contour are closer to patch center
            i3, i1 = map(x -> x[argmin(
                [norm(k[j] - sheet_isolines[i-1].points[x[1]]),
                norm(k[j] - sheet_isolines[i-1].points[x[2]])])],
                (ij3, ij1)
            )
            i4, i2 = map(x -> x[argmin(
                [norm(k[j] - sheet_isolines[i+1].points[x[1]]),
                norm(k[j] - sheet_isolines[i+1].points[x[2]])])],
                (ij4, ij2)
            )

            poly = vcat(corners[cind-4:cind-1], sheet_isolines[i-1].points[min(i1,i3):max(i1,i3)], sheet_isolines[i+1].points[min(i2,i4):max(i2,i4)])

            dV = poly_area(poly, k[j])

            ij3 = ij1; ij4 = ij2

            Δε = foliated_energies[i+1] - foliated_energies[i-1]
            Δs = arclengths[j+1] - arclengths[j-1]

            v  = band_velocity(ε, k[j])
            p1 = k_inner[j]
            p2 = k_outer[j]

            A = Matrix{Float64}(undef, 2, 2)
            J = Matrix{Float64}(undef, 2, 2)

            A[1,1] = (p2[1] - p1[1]) / Δε
            A[1,2] = (p2[2] - p1[2]) / Δε
            A[2,1] = (k[j+1][1] - k[j-1][1]) / Δs
            A[2,2] = (k[j+1][2] - k[j-1][2]) / Δs

            J[1,1] =  2 * v[1] / Δε
            J[1,2] =  2 * v[2] / Δε
            J[2,1] = -2 * A[1,2] / det(A) / Δs
            J[2,2] =  2 * A[1,1] / det(A) / Δs

            patches[i÷2, j÷2] = Patch(
                foliated_energies[i],
                k[j],
                v,
                Δε,
                dV,
                inv(J),
                1 / abs(det(J)),
                band_index
            )
        end
    end

    return Mesh(vec(patches), corners, vec(corner_ids))
end

"""
    get_arclengths(curve)

Get the distance to each point along `curve`.

Treats `curve` as an ordered list of points defining a piecewise linear path.
"""
function get_arclengths(curve)
    s = 0.0
    arclengths = Vector{Float64}(undef, length(curve))
    arclengths[1] = 0.0
    for i in eachindex(curve)
        i == 1 && continue
        s += norm(curve[i] - curve[i-1])
        arclengths[i] = s
    end
    return arclengths
end

"""
    find_cusp_fraction(curve; angle_threshold = π/8)

Return the arclength fractions in [0, 1] of cusp points in `curve`.

A cusp is detected at interior point `i` when the angle between the incoming
tangent `curve[i] - curve[i-1]` and outgoing tangent `curve[i+1] - curve[i]`
exceeds `angle_threshold`.
"""
function find_cusp_fraction(curve; angle_threshold = π/8)
    n = length(curve)
    n < 3 && return Float64[]
    arclengths = get_arclengths(curve)
    L = arclengths[end]
    L < eps() && return Float64[]
    fractions = Float64[]
    for i in 2:n-1
        t1 = curve[i] - curve[i-1]
        t2 = curve[i+1] - curve[i]
        l1, l2 = norm(t1), norm(t2)
        (l1 < eps() || l2 < eps()) && continue
        cos_θ = clamp(dot(t1, t2) / (l1 * l2), -1.0, 1.0)
        if acos(cos_θ) > angle_threshold
            push!(fractions, arclengths[i] / L)
        end
    end
    return fractions
end

"""
    arclength_slice(curve, n::Int; pinned_arclengths)

Cut `curve` into `n` uniformly spaced points.

Treats `curve` as an ordered list of points defining a piecewise linear path, and linear
interpolation is used to find intermediate points. Returns interpolated points and
arclengths along original curve of interpolated points.
For best results, points in `curve` should have approximately uniform spacing and `n`
should be much less than the number of points in `curve`.

# Arguments
- `pinned_arclengths`: optional list of arclength values at which a cut must be placed.
  For each value, the nearest interior corner position (odd 1-based index, excluding the
  two endpoints) is snapped to that arclength. The two adjacent even-indexed (patch-center)
  positions are then recentered halfway between their bounding corners. Values within one
  uniform corner step (`δs = L / n_arc`) of either endpoint are skipped.
"""
function arclength_slice(curve, n::Int; pinned_arclengths = Float64[])
    if length(curve) == 1
        return fill(curve[begin], n), zeros(Float64, n)
    end

    arclengths = get_arclengths(curve)
    τ = collect(LinRange(0.0, arclengths[end], n))

    δs = n > 1 ? 2 * arclengths[end] / (n - 1) : arclengths[end]
    for s_pin in pinned_arclengths
        s = clamp(s_pin, 0.0, arclengths[end])
        (s < δs || s > arclengths[end] - δs) && continue
        best_idx = 0
        best_dist = Inf
        for k in 3:2:n-2
            d = abs(τ[k] - s)
            if d < best_dist
                best_dist = d
                best_idx = k
            end
        end
        if best_idx > 0
            τ[best_idx] = s
            if best_idx > 2
                τ[best_idx - 1] = (τ[best_idx - 2] + τ[best_idx]) / 2
            end
            if best_idx < n - 1
                τ[best_idx + 1] = (τ[best_idx] + τ[best_idx + 2]) / 2
            end
        end
    end
    isempty(pinned_arclengths) || sort!(τ)

    grid = [curve[begin]]

    j = 1
    jmax = length(arclengths)
    for i in eachindex(τ)
        i == 1 && continue
        while j < jmax && arclengths[j] <= τ[i]
            j += 1
        end
        push!(grid, curve[j-1] + (curve[j] - curve[j - 1]) * (τ[i] - arclengths[j-1]) / (arclengths[j] - arclengths[j-1]))
    end

    return grid, collect(τ)
end

"""
    find_marginal_energy(X, Y, E_mat, region, e_target, e_outer, n_iso_target[; iter])

Bisect in the interval `[e_target, e_outer]` to find the energy threshold at which the
number of isolines transitions through `n_iso_target`. `e_target` must satisfy
`n_iso >= n_iso_target`; `e_outer` must not.

Returns the last energy (closest to `e_outer`) where `n_iso >= target_n_iso` still holds.
"""
function find_marginal_energy(X, Y, E_mat, region, e_target, e_outer, n_iso_target; iter = 32)
    a, b = e_target, e_outer
    for _ in 1:iter
        mid = (a + b) / 2
        bundle = contours(X, Y, E_mat, [mid]; mask=region)[1]
        sort_isolines!(bundle)
        if length(bundle.isolines) == n_iso_target
            a = mid
        else
            b = mid
        end
    end
    return a
end

###
### IBZ corner crossing detection
###

# Returns 1-based index of the edge of polygon `region` on which point `p` lies, or 0.
function edge_index(p, region; tol = 1e-8)
    n = length(region)
    for k in 1:n
        a  = region[k]
        b  = region[mod1(k + 1, n)]
        v  = b - a
        w  = p - a
        lv = norm(v)
        lv < eps() && continue
        abs(v[1] * w[2] - v[2] * w[1]) / lv > tol && continue
        t = dot(w, v) / dot(v, v)
        -tol ≤ t ≤ 1 + tol && return k
    end
    return 0
end

"""
    find_boundary_extrema(p1, corner, p2, ε[; n_sample, iter])

Return the locations of local extrema of `ε` along the piecewise-linear path
`p1 → corner → p2` on the boundary of the IBZ.

The path is parameterised by arclength. Each segment is scanned independently with
`n_sample ÷ 2` evenly spaced points; the interval bordering the corner is excluded
from each segment so that the corner itself is never returned. Sign changes in the
finite-difference derivative are refined by bisection.

# Arguments
- `p1::SVector{2,Float64}`: start of path, lying on one IBZ edge.
- `corner::SVector{2,Float64}`: IBZ corner vertex shared by the two edges.
- `p2::SVector{2,Float64}`: end of path, lying on the adjacent IBZ edge.
- `ε`: dispersion function accepting an `AbstractVector` of length 2.
- `n_sample::Int=200`: number of points for the coarse scan.
- `iter::Int=64`: bisection iteration count per bracketed extremum.

# Returns
`Vector{SVector{2,Float64}}` of extremum locations, ordered by arclength.

# Examples
```jldoctest
julia> using StaticArrays
julia> ε(k) = k[1] + k[2];  # linear dispersion, no interior extrema
julia> p1 = SVector(0.0, 0.0); corner = SVector(1.0, 0.0); p2 = SVector(1.0, 1.0);
julia> isempty(find_boundary_extrema(p1, corner, p2, ε))
true
```

See also [`detect_ibz_corner_crossings`](@ref).
"""
function find_boundary_extrema(
    p1::SVector{2,Float64},
    corner::SVector{2,Float64},
    p2::SVector{2,Float64},
    ε;
    n_sample::Int = 200,
    iter::Int = 64,
) :: Vector{SVector{2,Float64}}
    L1 = norm(corner - p1)
    L2 = norm(p2 - corner)
    L  = L1 + L2
    L < eps() && return SVector{2,Float64}[]

    lerp(t) = t ≤ L1 ? p1 + (t / L1) * (corner - p1) :
                        corner + ((t - L1) / L2) * (p2 - corner)

    # Search each segment separately so the path kink at the corner does not produce
    # a spurious sign change in the cross-segment finite-difference derivative.
    # The interval bordering the corner is excluded from each segment: the last
    # interval of p1→corner and the first interval of corner→p2.
    extrema_pts = SVector{2,Float64}[]
    n_seg = max(4, n_sample ÷ 2)
    for (t_lo, t_hi, i_start) in ((0.0, L1, 1), (L1, L, 2))
        t_hi - t_lo < eps() && continue
        ts_seg = LinRange(t_lo, t_hi, n_seg)
        es_seg = [ε(lerp(t)) for t in ts_seg]
        δ = step(ts_seg) / 2
        for i in i_start:(n_seg - 2)
            Δ_left  = es_seg[i + 1] - es_seg[i]
            Δ_right = es_seg[i + 2] - es_seg[i + 1]
            Δ_left * Δ_right < 0 || continue
            t_ext = bisect(t -> ε(lerp(t + δ)) - ε(lerp(t - δ)), ts_seg[i], ts_seg[i + 1]; iter)
            push!(extrema_pts, lerp(t_ext))
        end
    end

    return extrema_pts
end

"""
    detect_ibz_corner_crossings(c, region, ε[; tol])

Detect crossings of open isoline endpoints across corners of the IBZ polygon `region`
as energy varies through the bundles in `c`, and return the boundary energy extrema at
each crossing.

For each adjacent bundle pair `(c[b], c[b+1])` sharing a common sheet count, checks
whether one endpoint of the open isoline slides from one IBZ edge to an adjacent edge.
For each such corner crossing, calls [`find_boundary_extrema`](@ref) on the IBZ boundary
path through the corner. Returns all resulting extremal momentum points, flattened across
all detected crossings.

# Arguments
- `c::Vector{IsolineBundle}`: isoline bundles ordered by energy.
- `region`: IBZ polygon vertices as a counterclockwise `Vector{SVector{2,Float64}}`.
- `ε`: dispersion function accepting an `AbstractVector` of length 2.
- `tol`: collinearity/range tolerance passed to `edge_index` (default `1e-8`).

# Returns
`Vector{SVector{2,Float64}}` of extremal momentum points along IBZ boundary paths,
collected across all detected corner crossings.

# Examples
```julia
# After generating contours over an IBZ mesh:
c = contours(X, Y, E, fe; mask = region)
sort_isolines!.(c)
for q in detect_ibz_corner_crossings(c, region, ε)
    # ε(q) is an energy at which a new contour should be inserted
end
```

See also [`find_boundary_extrema`](@ref).
"""
function detect_ibz_corner_crossings(
    c::Vector{IsolineBundle},
    region,
    ε;
    tol = 1e-8,
) :: Vector{SVector{2,Float64}}
    n_region     = length(region)
    extrema_all  = SVector{2,Float64}[]

    for b in 1:(length(c) - 1)
        n_s = min(length(c[b].isolines), length(c[b + 1].isolines))
        n_s == 0 && continue

        for s in 1:n_s
            iso_a = c[b].isolines[s]
            iso_b = c[b + 1].isolines[s]
            (iso_a.isclosed || iso_b.isclosed) && continue

            pa1 = iso_a.points[begin]
            pa2 = iso_a.points[end]
            pb1 = iso_b.points[begin]
            pb2 = iso_b.points[end]

            # Match endpoints by minimum total distance.
            if norm(pa1 - pb1) + norm(pa2 - pb2) ≤ norm(pa1 - pb2) + norm(pa2 - pb1)
                ea1, ea2 = edge_index(pa1, region; tol), edge_index(pa2, region; tol)
                eb1, eb2 = edge_index(pb1, region; tol), edge_index(pb2, region; tol)
                pa_cands = (pa1, pa2)
                pb_cands = (pb1, pb2)
            else
                ea1, ea2 = edge_index(pa1, region; tol), edge_index(pa2, region; tol)
                eb1, eb2 = edge_index(pb2, region; tol), edge_index(pb1, region; tol)
                pa_cands = (pa1, pa2)
                pb_cands = (pb2, pb1)
            end
            ea_cands = (ea1, ea2)
            eb_cands = (eb1, eb2)

            jumped1 = ea_cands[1] != 0 && eb_cands[1] != 0 && ea_cands[1] != eb_cands[1]
            jumped2 = ea_cands[2] != 0 && eb_cands[2] != 0 && ea_cands[2] != eb_cands[2]
            jumped1 ⊻ jumped2 || continue

            j = jumped1 ? 1 : 2
            old_edge, new_edge = ea_cands[j], eb_cands[j]
            pa_j, pb_j = pa_cands[j], pb_cands[j]

            corner = if new_edge == mod1(old_edge + 1, n_region)
                region[new_edge]
            elseif new_edge == mod1(old_edge - 1, n_region)
                region[old_edge]
            else
                continue
            end

            append!(extrema_all, find_boundary_extrema(pa_j, corner, pb_j, ε))
        end
    end

    return extrema_all
end

"""
    mesh_region(region, ε, band_index::Int, T, Δε, n_arc::Int[, N::Int, α::Real])

Generate a mesh of `region` of momentum space for dispersion `ε` (band `band_index`) at
temperature `T`.

The mesh covers a tube around the Fermi surface between `-αT` and `+αT`. The energy-direction
resolution is controlled by `Δε`: the number of patch rows is `2*ceil(α*T/Δε)`, always even
so that a patch center falls exactly on the Fermi surface. The arc-length resolution along each
energy contour is controlled by `n_arc`: each Fermi surface sheet is divided into `n_arc`
uniform arc-length segments.

Multiple disconnected Fermi surface sheets (e.g. annular topology) are handled automatically —
one sub-mesh is generated per sheet and the results are concatenated. For a closed isoline
(sheet entirely within `region`), `n_arc` is scaled by the ratio of the closed contour's
arc-length to that of the outermost open-arc sheet so that the full-BZ patch density is
consistent across sheets.

The resolution used for marching squares over `region` is set by `N`, the number of sample
points in each direction of the bounding rectangle.
"""
function mesh_region(region, ε, band_index::Int, T, Δε, n_arc::Int, N = 1001, α = 6.0;
                     bbox = nothing, angle_threshold = π/8)
    n_arc = max(3, n_arc)

    # Sample coordinates and energies for marching squares
    ((x_min, x_max), (y_min, y_max)) = get_bounding_box(isnothing(bbox) ? region : bbox)
    X = LinRange(x_min, x_max, N)
    Δx = (x_max - x_min) / (N - 1)
    Ny = round(Int, (y_max - y_min) / Δx)
    Y = LinRange(y_min, y_max, Ny)
    E = Matrix{Float64}(undef, N, Ny)
    for (i, x) in enumerate(X)
        for (j, y) in enumerate(Y)
            E[i,j] = ε([x, y])
        end
    end

    fe = foliated_energies(X, Y, E, ε, region, Δε, α, T)

    c = contours(X, Y, E, fe; mask = region)
    sort_isolines!.(c)
    split_bouncing_isolines!.(c, Ref(region))

    for q in detect_ibz_corner_crossings(c, region, ε)
        e_ext = ε(q)
        any(abs(e_ext - e) < 1e-10 * abs(e_ext) + 1e-14 for e in fe) && continue
        # Find the adjacent boundary energies (odd indices) that bracket e_ext.
        # Replacing the center between them and inserting 2 new entries keeps
        # length(fe) odd, preserving the boundary/center interleaving invariant.
        idx  = searchsortedfirst(fe, e_ext)
        b_hi = isodd(idx) ? idx : idx + 1
        b_lo = b_hi - 2
        (b_lo < 1 || b_hi > length(fe)) && continue
        c_idx    = b_lo + 1
        e_new_lo = (fe[b_lo] + e_ext) / 2
        e_new_hi = (e_ext + fe[b_hi]) / 2
        bndl_ext    = contours(X, Y, E, [e_ext];    mask = region)[1]
        bndl_new_lo = contours(X, Y, E, [e_new_lo]; mask = region)[1]
        bndl_new_hi = contours(X, Y, E, [e_new_hi]; mask = region)[1]
        sort_isolines!(bndl_ext); sort_isolines!(bndl_new_lo); sort_isolines!(bndl_new_hi)
        split_bouncing_isolines!(bndl_ext,    region)
        split_bouncing_isolines!(bndl_new_lo, region)
        split_bouncing_isolines!(bndl_new_hi, region)
        fe[c_idx] = e_new_lo;   c[c_idx] = bndl_new_lo
        insert!(fe, c_idx + 1, e_ext);    insert!(c, c_idx + 1, bndl_ext)
        insert!(fe, c_idx + 2, e_new_hi); insert!(c, c_idx + 2, bndl_new_hi)
    end


    n_isolines = [length(bundle.isolines) for bundle in c]

    all_patches    = Patch[]
    all_corners    = SVector{2,Float64}[]
    all_corner_ids = SVector{4,Int}[]

    # Partition energy levels into contiguous runs with the same isoline count,
    # then process each sheet within each run independently.
    #
    # mesh_sheet requires fe to start and end at boundary energies
    # (odd 1-based indices). If a run starts or ends at an even index (a patch-center
    # energy), extend it with a marginal boundary found by bisection at the transition.
    i = 1
    while i <= length(n_isolines)
        j = i
        while j <= length(n_isolines) && n_isolines[j] == n_isolines[i]
            j += 1
        end

        run   = i:(j-1)
        n_iso = n_isolines[i]
        i     = j

        (n_iso == 0 || run.stop == run.start) && continue

        need_lo = iseven(run.start)
        need_hi = iseven(run.stop)
        shift_lo = run.start != 1 && isodd(run.start)
        shift_hi = run.stop != length(n_isolines) && isodd(run.stop)

        if need_lo || shift_lo
            e_lo_m    = find_marginal_energy(X, Y, E, region,
                            fe[run.start], fe[run.start - 1], n_iso)
            bundle_lo = contours(X, Y, E, [e_lo_m]; mask=region)[1]
            sort_isolines!(bundle_lo)
            split_bouncing_isolines!(bundle_lo, region)
        end
        if need_hi || shift_hi
            e_hi_m    = find_marginal_energy(X, Y, E, region,
                            fe[run.stop], fe[run.stop + 1], n_iso)
            bundle_hi = contours(X, Y, E, [e_hi_m]; mask=region)[1]
            sort_isolines!(bundle_hi)
            split_bouncing_isolines!(bundle_hi, region)
        end

        for s in 1:n_iso
            base_isolines = Isoline[c[idx].isolines[s] for idx in run]
            base_energies = collect(fe[run])

            lo_start = shift_lo ? 2 : 1
            hi_stop  = shift_hi ? length(base_isolines) - 1 : length(base_isolines)
            base_iso = base_isolines[lo_start:hi_stop]
            base_en  = base_energies[lo_start:hi_stop]

            pre_iso = (need_lo || shift_lo) ? [bundle_lo.isolines[s]] : Isoline[]
            pre_en  = (need_lo || shift_lo) ? [e_lo_m]                : Float64[]
            suf_iso = (need_hi || shift_hi) ? [bundle_hi.isolines[s]] : Isoline[]
            suf_en  = (need_hi || shift_hi) ? [e_hi_m]                : Float64[]

            isolines_s          = [pre_iso; base_iso; suf_iso]
            foliated_energies_s = [pre_en;  base_en;  suf_en]

            if need_lo || shift_lo
                foliated_energies_s[2] = (foliated_energies_s[1] + foliated_energies_s[3]) / 2
            end
            if need_hi || shift_hi
                foliated_energies_s[end-1] = (foliated_energies_s[end-2] + foliated_energies_s[end]) / 2
            end

            mesh_s = mesh_sheet(
                isolines_s, ε, band_index, n_arc,
                foliated_energies_s, length(all_corners);
                angle_threshold,
            )
            append!(all_patches,    mesh_s.patches)
            append!(all_corners,    mesh_s.corners)
            append!(all_corner_ids, mesh_s.corner_inds)
        end
    end

    return Mesh(all_patches, all_corners, all_corner_ids)
end

mesh_region(region, ε, T, Δε, n_arc::Int, N = 1001, α = 6.0; bbox = nothing, angle_threshold = π/8) =
    mesh_region(region, ε, 1, T, Δε, n_arc, N, α; bbox, angle_threshold)

"""
    ibz_mesh(l::Lattice, bands::AbstractVector, T, Δε, n_arc::Int[, N::Int, α::Real]; mask)

Generate a mesh of the irreducible Brillouin Zone (IBZ) for lattice `l` given dispersions
in `bands` at temperature `T`. See [`mesh_region`](@ref) for a description of the resolution
parameters `Δε` and `n_arc`.

# Arguments
- `mask`: optional vector of vertices defining a convex polygonal sub-region of the BZ. When
  `length(mask) > 2` the mesh is restricted to `mask` instead of the full IBZ, which is
  useful for finer gridding around a small Fermi surface pocket.
"""
function ibz_mesh(l::Lattice, bands::AbstractVector, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0;
                  mask = SVector{2,Float64}[], angle_threshold = π/8)
    full_patches    = Patch[]
    full_corners    = SVector{2,Float64}[]
    full_corner_ids = SVector{4,Int}[]
    ibz  = get_ibz(l)
    bbox = length(mask) > 2 ? mask : nothing

    for i in eachindex(bands)
        mesh = mesh_region(ibz, bands[i], i, T, Δε, n_arc, N, α; bbox, angle_threshold)
        ℓ = length(full_corners)
        append!(full_patches, mesh.patches)
        append!(full_corner_ids, map(x -> SVector{4,Int}(x .+ ℓ), mesh.corner_inds))
        append!(full_corners, mesh.corners)
    end

    return Mesh(full_patches, full_corners, full_corner_ids)
end

ibz_mesh(l::Lattice, ε::Function, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0;
         mask = SVector{2,Float64}[], angle_threshold = π/8) =
    ibz_mesh(l, [ε], T, Δε, n_arc, N, α; mask, angle_threshold)

"""
    group_angle_perm(G, ibz) -> Vector{Int}

Return the permutation that sorts the elements of point group `G` by the polar angle of
the image of the IBZ centroid under each element's matrix representation.

Concretely, for each group element `g` with 2×2 matrix `O`, the angle
`θ_g = atan(k[2], k[1]) mod 2π` is computed where `k = O * centroid(ibz)`.
The returned index vector `p` satisfies `θ_{p[1]} ≤ θ_{p[2]} ≤ … ≤ θ_{p[|G|]}`.

This ordering assigns each group element to a unique angular sector of the BZ and is the
canonical sector ordering shared by [`bz_mesh`](@ref) and [`bz_symmetry_map`](@ref).
"""
function group_angle_perm(G, ibz)
    centroid = sum(ibz)
    θs = Vector{Float64}(undef, length(G.elements))
    for i in eachindex(G.elements)
        O = get_matrix_representation(G.elements[i])
        k = O * centroid
        θs[i] = mod(atan(k[2], k[1]), 2π)
    end
    return sortperm(θs)
end

"""
    bz_mesh(l::Lattice, bands::AbstractVector, T, Δε, n_arc::Int[, N::Int, α::Real]; mask)

Generate a mesh of the Brillouin Zone (BZ) for lattice `l` given dispersions in `bands` at
temperature `T` by computing the IBZ mesh and replicating it under all point-group operations.
See [`mesh_region`](@ref) for a description of `Δε` and `n_arc`.

# Arguments
- `mask`: optional vector of vertices passed through to [`ibz_mesh`](@ref). When
  `length(mask) > 2` the IBZ mesh is restricted to `mask` instead of the full IBZ before
  being replicated across the BZ.
"""
function bz_mesh(l::Lattice, bands::AbstractVector, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0;
                 mask = SVector{2,Float64}[], angle_threshold = π/8)
    G     = point_group(l)
    ibz   = get_ibz(l)
    θperm = group_angle_perm(G, ibz)

    full_patches     = Patch[]
    full_corners     = SVector{2,Float64}[]
    full_corner_inds = SVector{4,Int}[]

    bbox = length(mask) > 2 ? mask : nothing

    for j in eachindex(bands)
        mesh = mesh_region(ibz, bands[j], j, T, Δε, n_arc, N, α; bbox, angle_threshold)

        for i in θperm
            O = get_matrix_representation(G.elements[i])
            ℓ = length(full_corners)
            append!(full_patches,     map(x -> patch_op(x, O), mesh.patches))
            append!(full_corner_inds, map(x -> SVector{4,Int}(x .+ ℓ), mesh.corner_inds))
            append!(full_corners,     map(x -> O * x, mesh.corners))
        end
    end

    return Mesh(full_patches, full_corners, full_corner_inds)
end

bz_mesh(l::Lattice, ε, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0;
        mask = SVector{2,Float64}[], angle_threshold = π/8) =
    bz_mesh(l, [ε], T, Δε, n_arc, N, α; mask, angle_threshold)

function radial_extrema(ε, kf::AbstractVector{<:Real}, e_window; N_prelim = 100)
    # Use the known Fermi crossings to bracket extrema and bound r_max.
    isempty(kf) && return Float64[], 0.0

    kf_sorted = sort(kf)
    r_last    = kf_sorted[end]

    ###
    ### r_max
    ###
    dε_last = derivative(ε, r_last)
    r_far   = r_last + 4 * e_window / abs(dε_last)
    r_max   = bisect(r -> ε(r) - sign(dε_last) * e_window, r_last, r_far)

    ###
    ### Extrema between consecutive Fermi crossings
    ###
    critical_radii = Float64[]
    for i in 1:(length(kf_sorted) - 1)
        push!(critical_radii, bisect(r -> derivative(ε, r), kf_sorted[i], kf_sorted[i + 1]))
    end

    ###
    ### Extrema before kf[1] (zone-centre pockets)
    ###
    r_grid  = range(eps(), kf_sorted[1], N_prelim)
    dε_vals = [derivative(ε, ri) for ri in r_grid]
    for i in 1:(N_prelim - 1)
        dε_vals[i] * dε_vals[i + 1] < 0 || continue
        push!(critical_radii, bisect(r -> derivative(ε, r), r_grid[i], r_grid[i + 1]))
    end

    return sort!(critical_radii), r_max
end

function isotropic_sheet_mesh(ε, energies_s, n_arc, corner_offset, r_lo, r_hi)
    # Mesh a single monotone radial section at the given boundary energy levels.
    n_levels = length(energies_s)
    n_angles = n_arc + 1
    Δθ       = 2π / n_arc

    corners     = Matrix{SVector{2,Float64}}(undef, n_levels, n_angles)
    k           = Matrix{SVector{2,Float64}}(undef, n_levels - 1, n_arc)
    corner_inds = Matrix{SVector{4,Int}}(undef, n_levels - 1, n_arc)
    radius      = Vector{Float64}(undef, 2 * n_levels - 1)

    foliated = Vector{Float64}(undef, 2 * n_levels - 1)
    for i in eachindex(foliated)
        if isodd(i)
            foliated[i] = energies_s[(i + 1) ÷ 2]
        else
            foliated[i] = (energies_s[i ÷ 2 + 1] + energies_s[i ÷ 2]) / 2
        end
    end

    r_a = min(r_lo, r_hi)
    r_b = max(r_lo, r_hi)

    for i in eachindex(foliated)
        radius[i] = bisect(r -> ε(r) - foliated[i], r_a, r_b)

        if isodd(i)
            corners[(i + 1) ÷ 2, :] = map(
                θ -> SVector{2,Float64}(radius[i] * cos(θ), radius[i] * sin(θ)),
                LinRange(0, 2π, n_angles)
            )
        else
            k[i ÷ 2, :] = map(
                θ -> SVector{2,Float64}(radius[i] * cos(θ), radius[i] * sin(θ)),
                (Δθ/2):Δθ:2π
            )
            for j in 1:n_arc
                ref = corner_offset + (j - 1) * n_levels + (i ÷ 2)
                corner_inds[i ÷ 2, j] = SVector{4,Int}(ref, ref + 1, ref + n_levels + 1, ref + n_levels)
            end
        end
    end

    v = map(x -> derivative(ε, norm(x)) * x / norm(x), k)

    patches = Matrix{Patch}(undef, n_levels - 1, n_arc)
    for i in 1:(n_levels - 1)
        E    = (energies_s[i + 1] + energies_s[i]) / 2
        Δε_i = energies_s[i + 1] - energies_s[i]
        dV   = abs(0.5 * Δθ * (radius[2*i + 1]^2 - radius[2*i - 1]^2))
        for j in 1:n_arc
            θ = atan(k[i, j][2], k[i, j][1])

            Jinv = zeros(Float64, 2, 2)
            Jinv[1,1] =  (Δε_i / 2) * cos(θ) / norm(v[i, j])
            Jinv[2,1] =  (Δε_i / 2) * sin(θ) / norm(v[i, j])
            Jinv[1,2] = -(Δθ / 2) * norm(k[i, j]) * sin(θ)
            Jinv[2,2] =  (Δθ / 2) * norm(k[i, j]) * cos(θ)

            patches[i, j] = Patch(E, k[i, j], v[i, j], Δε_i, dV, Jinv, abs(det(Jinv)), 1)
        end
    end

    return vec(patches), vec(corners), vec(corner_inds)
end

"""
    isotropic_mesh(ε, kf::AbstractVector{<:Real}, T::Real, Δε::Real, n_arc::Int[, α::Real])

Generate a mesh for an isotropic Fermi surface given the radial dispersion `ε` at temperature `T`.

`kf` is a vector of Fermi momenta — the radii at which `ε` is zero. The number of Fermi surface
sheets and the radial extent of the mesh are inferred directly from `kf`: radial extrema are
located by bisection in the intervals between consecutive Fermi crossings, and `r_max` is the
radius beyond `kf[end]` where `|ε| = αT`.

The mesh covers all sheets of the Fermi surface within the energy window [-`αT`, +`αT`]. The
dispersion is partitioned at each extremum into monotone sections, and each section intersecting
the energy window is meshed as a separate annular sheet and combined into a single `Mesh`.

The energy spacing `Δε` controls the resolution in the energy direction: each sheet of clipped
energy range `ΔE` receives `max(4, round(Int, ΔE / Δε) + 1)` levels (rounded up to an even
number). Each energy contour carries `n_arc` patches in the angular direction.

See also [`mesh_region`](@ref), [`bz_mesh`](@ref).
"""
function isotropic_mesh(ε, kf::AbstractVector{<:Real}, T::Real, Δε::Real, n_arc::Int, α::Real = 6.0)
    n_arc    = max(2, n_arc)
    e_window = α * T

    critical_radii, r_max = radial_extrema(ε, kf, e_window)
    r_max == 0.0 && return Mesh(Patch[], SVector{2,Float64}[], SVector{4,Int}[])

    boundaries = [0.0; critical_radii; r_max]

    all_patches    = Patch[]
    all_corners    = SVector{2,Float64}[]
    all_corner_ids = SVector{4,Int}[]
    corner_offset  = 0

    for s in 1:(length(boundaries) - 1)
        r_lo, r_hi = boundaries[s], boundaries[s + 1]
        r_lo ≈ r_hi && continue

        E_lo, E_hi = ε(r_lo), ε(r_hi)
        E_bot = max(min(E_lo, E_hi), -e_window)
        E_top = min(max(E_lo, E_hi),  e_window)
        E_top <= E_bot && continue

        n_levels_s = max(4, round(Int, (E_top - E_bot) / Δε) + 1)
        isodd(n_levels_s) && (n_levels_s += 1)

        energies_s = collect(LinRange(E_bot, E_top, n_levels_s))

        patches_s, corners_s, ids_s = isotropic_sheet_mesh(
            ε, energies_s, n_arc, corner_offset, r_lo, r_hi
        )

        append!(all_patches,    patches_s)
        append!(all_corners,    corners_s)
        append!(all_corner_ids, ids_s)
        corner_offset += n_levels_s * (n_arc + 1)
    end

    return Mesh(all_patches, all_corners, all_corner_ids)
end

isotropic_mesh(ε, kf::Real, T::Real, Δε::Real, n_arc::Int, α::Real = 6.0) = isotropic_mesh(ε, [kf], T, Δε, n_arc, α)
