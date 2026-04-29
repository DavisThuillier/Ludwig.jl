###
### Patch
###

"""
    AbstractPatch

Supertype for momentum-space patches that carry an energy, momentum, group velocity, and
band index.

See [`Patch`](@ref) for an integrable patch and [`VirtualPatch`](@ref) for a sample-only
patch.
"""
abstract type AbstractPatch end

"""
    Patch(e::Float64, k::SVector{2,Float64}, v::SVector{2,Float64},
          de::Float64, dV::Float64, jinv::SMatrix{2,2,Float64,4}, djinv::Float64,
          band_index::Int)

Construct a `Patch` defining a region of momentum space over which to integrate.

# Fields
- `e::Float64`: energy at the patch center.
- `k::SVector{2,Float64}`: momentum at the patch center.
- `v::SVector{2,Float64}`: group velocity at the patch center.
- `de::Float64`: width of the patch in energy.
- `dV::Float64`: area of the patch in momentum space.
- `jinv::SMatrix{2,2,Float64,4}`: Jacobian of the transformation
  ``(k_x, k_y) \\to (\\varepsilon, s)``.
- `djinv::Float64`: determinant of `jinv`.
- `band_index::Int`: index of the band from which `e` was sampled at `k`.
"""
struct Patch <: AbstractPatch
    e::Float64
    k::SVector{2,Float64}
    v::SVector{2,Float64}
    de::Float64
    dV::Float64
    jinv::SMatrix{2,2,Float64,4}
    djinv::Float64
    band_index::Int
end

"""
    VirtualPatch(e::Float64, k::SVector{2,Float64}, v::SVector{2,Float64},
                 band_index::Int)

Construct a `VirtualPatch` for sampling momentum, energy, and group velocity at a point.

A `VirtualPatch` exposes the same accessors as a [`Patch`](@ref) but carries no integration
weights and so cannot be integrated over. Used to represent symmetry-mapped or off-grid
samples inside the collision kernel.

# Fields
- `e::Float64`: energy.
- `k::SVector{2,Float64}`: momentum.
- `v::SVector{2,Float64}`: group velocity.
- `band_index::Int`: index of the band from which `e` was sampled at `k`.
"""
struct VirtualPatch <: AbstractPatch
    e::Float64
    k::SVector{2,Float64}
    v::SVector{2,Float64}
    band_index::Int
end

"""
    energy(p::AbstractPatch)

Return the energy at the centre of patch `p`.

# Examples
```jldoctest
julia> using StaticArrays

julia> p = VirtualPatch(0.5, SVector(0.0, 0.0), SVector(1.0, 0.0), 1);

julia> energy(p)
0.5
```
"""
energy(p::AbstractPatch)   = p.e

"""
    momentum(p::AbstractPatch)

Return the momentum at the centre of patch `p`.

# Examples
```jldoctest
julia> using StaticArrays

julia> p = VirtualPatch(0.0, SVector(0.1, 0.2), SVector(1.0, 0.0), 1);

julia> momentum(p)
2-element SVector{2, Float64} with indices SOneTo(2):
 0.1
 0.2
```
"""
momentum(p::AbstractPatch) = p.k

"""
    velocity(p::AbstractPatch)

Return the group velocity at the centre of patch `p`.

# Examples
```jldoctest
julia> using StaticArrays

julia> p = VirtualPatch(0.0, SVector(0.0, 0.0), SVector(1.0, -0.5), 1);

julia> velocity(p)
2-element SVector{2, Float64} with indices SOneTo(2):
  1.0
 -0.5
```
"""
velocity(p::AbstractPatch) = p.v

"""
    band(p::AbstractPatch)

Return the band index of `p`.

# Examples
```jldoctest
julia> using StaticArrays

julia> p = VirtualPatch(0.0, SVector(0.0, 0.0), SVector(0.0, 0.0), 3);

julia> band(p)
3
```
"""
band(p::AbstractPatch)     = p.band_index

"""
    area(p::Patch)

Return the momentum-space area of patch `p`.
"""
area(p::Patch)             = p.dV

"""
    patch_op(p::Patch, M::AbstractMatrix)

Apply the active transformation defined by matrix `M` on the relevant fields of patch `p`.
"""
function patch_op(p::Patch, M::AbstractMatrix)
    Mₛ = SMatrix{2,2,Float64,4}(M)
    return Patch(
        p.e,
        Mₛ * p.k,
        Mₛ * p.v,
        p.de,
        p.dV,
        Mₛ * p.jinv,
        p.djinv,
        p.band_index
    )
end

function patch_op(p::VirtualPatch, M::AbstractMatrix)
    Mₛ = SMatrix{2,2,Float64,4}(M)
    return VirtualPatch(
        p.e,
        Mₛ * p.k,
        Mₛ * p.v,
        p.band_index
    )
end

###
### Mesh
###

"""
    Mesh(patches::Vector{Patch}, corners::Vector{SVector{2,Float64}},
         corner_indices::Vector{SVector{4,Int}})

Construct a container of [`Patch`](@ref) objects with the corner geometry needed to plot
functions defined on patch centers.

# Fields
- `patches::Vector{Patch}`: the patches making up the mesh.
- `corners::Vector{SVector{2,Float64}}`: points on the patch corners, used for rendering.
- `corner_indices::Vector{SVector{4,Int}}`: for each patch, the four indices into
  `corners` giving its quadrilateral boundary.
"""
struct Mesh
    patches::Vector{Patch}
    corners::Vector{SVector{2, Float64}}
    corner_indices::Vector{SVector{4, Int}}
end

"""
    patches(m::Mesh)

Return the vector of [`Patch`](@ref) objects that make up `m`.
"""
patches(m::Mesh) = m.patches

"""
    corners(m::Mesh)

Return the vector of corner points used to render the patches of `m`.
"""
corners(m::Mesh) = m.corners

"""
    corner_indices(m::Mesh)

Return the vector mapping each patch of `m` to four indices into [`corners(m)`](@ref).
"""
corner_indices(m::Mesh) = m.corner_indices

###
### BZ symmetry
###

"""
    BZSymmetryMap(ibz_inds::Vector{Int}, ibz_preimage::Dict{Int,Int},
                  g_perms::Vector{Vector{Int}}, g_inv_perms::Vector{Vector{Int}},
                  ibz_g_idx::Dict{Int,Int})

Store precomputed symmetry information for a BZ mesh, relating each non-IBZ patch to its IBZ
representative via a point-group operation. Built via [`bz_symmetry_map`](@ref).

# Fields
- `ibz_inds::Vector{Int}`: BZ grid indices of the IBZ patches.
- `ibz_preimage::Dict{Int,Int}`: maps each non-IBZ patch index to its IBZ representative
  index.
- `g_perms::Vector{Vector{Int}}`: `g_perms[s][j]` is the BZ index of `O_s * grid[j]`, for
  each group element sector `s`.
- `g_inv_perms::Vector{Vector{Int}}`: inverse permutations; `g_inv_perms[s][j]` is the BZ
  index of `O_s^{-1} * grid[j]`.
- `ibz_g_idx::Dict{Int,Int}`: maps each non-IBZ patch index to the sector index `s` of its
  group element.
"""
struct BZSymmetryMap
    ibz_inds::Vector{Int}
    ibz_preimage::Dict{Int,Int}
    g_perms::Vector{Vector{Int}}
    g_inv_perms::Vector{Vector{Int}}
    ibz_g_idx::Dict{Int,Int}
end

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
    bz_symmetry_map(grid::Vector{Patch}, l::Lattice) -> BZSymmetryMap

Build the symmetry map for a BZ mesh generated from lattice `l`. All maps are constructed
by matching momenta in `grid` directly, so the result is independent of the internal
ordering convention used by [`bz_mesh`](@ref).

The returned [`BZSymmetryMap`](@ref) can be passed to [`fill_from_ibz!`](@ref) to fill
the non-IBZ rows of a collision matrix after computing only the IBZ rows.
"""
function bz_symmetry_map(grid::Vector{Patch}, l::Lattice)
    G       = point_group(l)
    ibz     = get_ibz(l)
    bz      = get_bz(l)
    N       = length(grid)
    G_order = length(G.elements)

    θperm = group_angle_perm(G, ibz)

    # Identify IBZ patches by polygon membership.
    ibz_inds = filter(i -> in_polygon(grid[i].k, ibz), 1:N)

    # `bz_mesh` builds a full BZ by appending each IBZ patch under every group operation,
    # so length(grid) == G_order × #IBZ-interior patches exactly. Catch the common
    # mistake of passing an `ibz_mesh` here.
    N == G_order * length(ibz_inds) || throw(ArgumentError(
        "bz_symmetry_map requires a full BZ mesh; got $N patches with " *
        "$(length(ibz_inds)) inside the IBZ, but expected $(G_order * length(ibz_inds)) " *
        "(G_order × #IBZ-interior). Did you mean to pass the output of `bz_mesh`?"
    ))

    # Tolerance for matching `O * k` to a grid patch. Sits between float noise
    # from the matrix multiply (~ulp(|k|), so ≤ 1e-13 × diameter(bz)) and mesh
    # spacing (≥ ~1e-3 × diameter(bz) for any realistic resolution).
    tol = 1e-8 * diameter(bz)

    # Sort patches by k[1] so we can find candidates via binary search.
    sort_perm = sortperm(1:N; by = i -> grid[i].k[1])
    sorted_kx = Float64[grid[sort_perm[s]].k[1] for s in 1:N]

    # Locate the grid index whose momentum is within `tol` of `query`.
    find_idx = function(query::SVector{2,Float64})
        lo = searchsortedfirst(sorted_kx, query[1] - tol)
        hi = searchsortedlast(sorted_kx, query[1] + tol)
        best_d = Inf
        best_i = 0
        for s in lo:hi
            i = sort_perm[s]
            d = norm(grid[i].k - query)
            if d < best_d
                best_d = d
                best_i = i
            end
        end
        best_d ≤ tol || throw(ErrorException(
            "bz_symmetry_map: no grid patch within tol = $tol of momentum $query " *
            "(closest distance $best_d). The grid may not be closed under the lattice point group."
        ))
        return best_i
    end

    # Build permutation vectors for each group element.
    g_perms = Vector{Vector{Int}}(undef, G_order)
    for (s, g_idx) in enumerate(θperm)
        O = SMatrix{2,2,Float64,4}(get_matrix_representation(G.elements[g_idx]))
        g_perms[s] = [find_idx(O * grid[j].k) for j in 1:N]
    end
    g_inv_perms = [invperm(perm) for perm in g_perms]

    # Find the identity sector (the sector whose permutation is the identity).
    ibz_sector = findfirst(s -> g_perms[s] == collect(1:N), 1:G_order)

    # For each non-identity sector, map BZ patches to their IBZ representatives.
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

"""
    ibz_matvec!(y, L::AbstractMatrix, x, sym::BZSymmetryMap) -> y

Compute `y = L * x` in-place using only the IBZ rows of `L` (those indexed by
`sym.ibz_inds`). Non-IBZ rows are reconstructed on the fly via the point-group
invariance `L[O*i, O*j] = L[i, j]`, without calling [`fill_from_ibz!`](@ref).

Argument order matches the `LinearAlgebra.mul!(y, A, x)` convention so this routine
composes directly with iterative-solver back ends (`IterativeSolvers.bicgstabl`,
`gmres`, etc.).
"""
function ibz_matvec!(y::AbstractVector, L::AbstractMatrix, x::AbstractVector, sym::BZSymmetryMap)
    for i_ibz in sym.ibz_inds
        y[i_ibz] = dot(view(L, i_ibz, :), x)
    end
    for (i_bz, i_ibz) in sym.ibz_preimage
        g = sym.ibz_g_idx[i_bz]
        y[i_bz] = dot(view(L, i_ibz, :), view(x, sym.g_perms[g]))
    end
    return y
end

"""
    diagonalize_ibz(L::AbstractMatrix, sym::BZSymmetryMap) -> LinearAlgebra.Eigen

Diagonalize the full N×N collision operator `L` without calling [`fill_from_ibz!`](@ref).
Only the IBZ rows of `L` (those indexed by `sym.ibz_inds`) need to be populated.

Internally reconstructs the full matrix column-by-column via [`ibz_matvec!`](@ref) applied
to each standard basis vector, then returns `eigen(L_full)`. `L` is not mutated.

# Example
```julia
L_sym = zeros(N, N)
for i in sym.ibz_inds, j in 1:N
    L_sym[i, j] = electron_electron(grid, f0s, i, j, bands, T, Weff_squared, l)
end
result = diagonalize_ibz(L_sym, sym)
vals, vecs = result.values, result.vectors
```
"""
function diagonalize_ibz(L::AbstractMatrix, sym::BZSymmetryMap)
    N      = size(L, 1)
    L_full = similar(L)
    e_j    = zeros(eltype(L), N)
    col    = Vector{eltype(L)}(undef, N)

    for j in 1:N
        e_j[j] = one(eltype(L))
        ibz_matvec!(col, L, e_j, sym)
        L_full[:, j] .= col
        e_j[j] = zero(eltype(L))
    end

    return eigen(L_full)
end

###
### Energy-level grid
###

"""
    foliate(x::AbstractVector) -> Vector{Float64}

Interleave midpoints between consecutive elements of `x`.

Returns a vector of length `2*length(x) - 1` where odd-indexed entries (1-based) hold the
original values of `x` and even-indexed entries hold midpoints between adjacent elements.

See also [`mesh_region`](@ref).
"""
function foliate(x::AbstractVector)
    n = length(x)
    foliated = Vector{Float64}(undef, 2 * n - 1)
    for i in eachindex(foliated)
        if isodd(i)
            foliated[i] = x[(i - 1) ÷ 2 + 1]
        else
            foliated[i] = (x[i ÷ 2 + 1] + x[i ÷ 2]) / 2
        end
    end
    return foliated
end

"""
    boundary_energies(X, Y, E, ε, region, e_min, e_max, Δε)

Return the vector of energy levels at which to extract Fermi-surface contours inside
`region`.

Levels are placed at spacing `Δε` over the requested window `[e_min, e_max]`, then trimmed
to the actual range of `E` sampled on the `(X, Y)` grid inside `region`. Any corner of
`region` whose energy `ε(k)` falls within the resulting window is pinned exactly onto the
energy grid, so the corner becomes a contour boundary rather than being straddled by
adjacent levels.
"""
function boundary_energies(X, Y, E, ε, region, e_min, e_max, Δε)
    n_levels = max(1, ceil(Int, (e_max - e_min) / Δε))
    if isapprox(e_min, -e_max) && isodd(n_levels)
        n_levels += 1 # force even so E=0 lands on a patch center, not a boundary
    end
    energies = collect(LinRange(e_min, e_max, n_levels))

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

    # Shift energy levels to include a corner energy if the dispersion crosses it
    for k ∈ region
        corner_e = ε(k)
        if energies[begin] < corner_e < energies[end]
            i = argmin(abs.(energies .- corner_e))
            if i == firstindex(energies) || i == lastindex(energies)
                energies[i] = corner_e
            else
                n = length(energies)
                energies[1:i] = LinRange(energies[begin], corner_e, i)
                Δe = (energies[end] - corner_e) / (n - i)
                energies[i+1:end] = LinRange(corner_e + Δe, energies[end], n - i)
            end
        end
    end

    return energies
end

###
### IBZ meshing
###

"""
    mesh_sheet(sheet_isolines, ε, band_index, n_arc_s, foliated_energies, corner_offset; angle_threshold)

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
        k_inner = aligned_arclength_slice(
            sheet_isolines[i-1], sheet_isolines[i], 2 * n_arc_s + 1;
            cusp_fractions, angle_threshold,
        )
        k_outer = aligned_arclength_slice(
            sheet_isolines[i+1], sheet_isolines[i], 2 * n_arc_s + 1;
            cusp_fractions, angle_threshold,
        )

        # Index of the original contour point nearest each corner along
        # sheet_isolines[i±1]. `aligned_arclength_slice` may reverse the
        # contour and pin cusps, so the corners are NOT at the raw arclength
        # fraction of the contour — locate them geometrically instead.
        inner_pts = sheet_isolines[i-1].points
        outer_pts = sheet_isolines[i+1].points
        inner_corner_idx = [argmin(norm(p - kc) for p in inner_pts) for kc in k_inner]
        outer_corner_idx = [argmin(norm(p - kc) for p in outer_pts) for kc in k_outer]

        corners[cind]   = k_inner[1]
        corners[cind+1] = k_outer[1]
        cind += 2

        for j in 2:2:lastindex(k)
            corners[cind]   = k_inner[j+1]
            corners[cind+1] = k_outer[j+1]
            corner_ids[i÷2, j÷2] = SVector{4, Int}(cind-2, cind-1, cind+1, cind) .+ corner_offset
            cind += 2

            # Original-contour indices nearest each corner of this patch.
            # The patch spans arclength [j-1, j+1] on the center contour, so the
            # inner/outer edges run between the corresponding aligned-slice
            # corners on sheet_isolines[i±1].
            i3 = inner_corner_idx[j-1]
            i1 = inner_corner_idx[j+1]
            i4 = outer_corner_idx[j-1]
            i2 = outer_corner_idx[j+1]

            poly = vcat(corners[cind-4:cind-1], sheet_isolines[i-1].points[min(i1,i3):max(i1,i3)], sheet_isolines[i+1].points[min(i2,i4):max(i2,i4)])

            dV = poly_area(poly, k[j])

            Δε = foliated_energies[i+1] - foliated_energies[i-1]
            Δs = arclengths[j+1] - arclengths[j-1]

            v  = band_velocity(ε, k[j])
            p1 = k_inner[j]
            p2 = k_outer[j]

            A = SMatrix{2,2,Float64,4}(
                (p2[1] - p1[1]) / Δε,
                (k[j+1][1] - k[j-1][1]) / Δs,
                (p2[2] - p1[2]) / Δε,
                (k[j+1][2] - k[j-1][2]) / Δs,
            )

            detA = det(A)
            J = SMatrix{2,2,Float64,4}(
                 2 * v[1] / Δε,
                -2 * A[1,2] / detA / Δs,
                 2 * v[2] / Δε,
                 2 * A[1,1] / detA / Δs,
            )

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
    find_marginal_energy(X, Y, E_mat, region, e_target, e_outer, n_iso_target[; iter])

Bisect in the interval `[e_target, e_outer]` to find the energy threshold at which the
number of isolines transitions through `n_iso_target`. `e_target` must satisfy
`n_iso >= n_iso_target`; `e_outer` must not.

Returns the last energy (closest to `e_outer`) where `n_iso == target_n_iso` still holds.
"""
function find_marginal_energy(X, Y, E_mat, region, e_target, e_outer, n_iso_target; iter = 32)
    a, b = e_target, e_outer
    for _ in 1:iter
        mid = (a + b) / 2
        bundle = contours(X, Y, E_mat, [mid]; mask=region)[1]
        if length(bundle.isolines) == n_iso_target
            a = mid
        else
            b = mid
        end
    end
    return a
end

"""
    detect_skipped_corners(c, region[; corner_levels, tol])

Detect crossings of open isoline endpoints across corners of the IBZ polygon `region`
as energy varies through the bundles in `c`.

For each adjacent bundle pair `(c[i], c[i+2])` sharing a common sheet count, checks
whether one endpoint of an open isoline slides from one IBZ edge to an adjacent edge.

# Arguments
- `c::Vector{IsolineBundle}`: isoline bundles ordered by energy.
- `region`: IBZ polygon vertices as a counterclockwise `Vector{SVector{2,Float64}}`.
- `corner_levels`: energies at which a contour is pinned exactly to an IBZ corner; bundle
  pairs containing such a level are skipped because their endpoints terminate at the
  corner by construction rather than crossing past it (default `Float64[]`).
- `tol`: collinearity/range tolerance passed to [`edge_index`](@ref) and used as the
  matching tolerance against `corner_levels` (default `1e-8`).

# Returns
`Vector` of `(i, s, path)` tuples, one per crossing event:
- `i::Int`: bundle index into `c` where the crossing was detected.
- `s::Int`: sheet index within the bundle.
- `path::Vector{SVector{2,Float64}}`: three points `[p1, corner, p2]` where `p1` is the
  isoline endpoint before the crossing, `corner` is the IBZ vertex, and `p2` is the
  endpoint after the crossing.

Multiple sheets at the same energy level each produce a separate entry with the same `i`.
"""
function detect_skipped_corners(
    c::Vector{IsolineBundle},
    region;
    corner_levels::AbstractVector{<:Real} = Float64[],
    tol = 1e-8,
)
    events = Tuple{Int, Int, Vector{SVector{2, Float64}}}[]

    n_reg = length(region)
    length(c) == 0 && return events

    at_corner_level(level) = any(abs(level - cl) < tol for cl in corner_levels)

    for i in 1:2:(length(c) - 2)
        b1 = c[i].isolines
        b2 = c[i + 2].isolines

        n_s = length(b1)
        if n_s != length(b2)
            continue # Different FS topology, handled separately
        end
        n_s == 0 && continue

        permute!(b2, match_contour_segments(b1, b2))

        for s in 1:n_s
            iso_1 = b1[s]
            iso_2 = b2[s]

            p11 = iso_1.points[begin]
            p12 = iso_1.points[end]
            p21 = iso_2.points[begin]
            p22 = iso_2.points[end]

            # Gather sets of edges
            e1 = [edge_index(p11, region; tol), edge_index(p12, region; tol)]
            e2 = [edge_index(p21, region; tol), edge_index(p22, region; tol)]

            Set(e1) == Set(e2) && continue

            # Identify which endpoint jumped and which IBZ corner it crossed
            e_old, e_new, p1, p2 = if e1[1] != e2[1]
                e1[1], e2[1], p11, p21
            else
                e1[2], e2[2], p12, p22
            end

            corner = if mod1(e_old + 1, n_reg) == e_new
                region[e_new]
            elseif mod1(e_new + 1, n_reg) == e_old
                region[e_old]
            else
                region[argmin(norm.(region .- Ref((p1 + p2) / 2)))]
            end

            # A contour at a corner energy whose relevant endpoint is pinned
            # to that corner terminates there by construction — not a missed
            # crossing event. The level alone is insufficient: when the
            # boundary dispersion is non-monotone, a corner energy may be
            # included in the energy grid without the contour at that level
            # actually touching the corner.
            if (at_corner_level(c[i].level)     && norm(p1 - corner) < tol) ||
               (at_corner_level(c[i + 2].level) && norm(p2 - corner) < tol)
                continue
            end

            push!(events, (i, s, [p1, corner, p2]))
        end
    end

    return events
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

julia> p1 = SVector(0.0, 0.0);

julia> corner = SVector(1.0, 0.0);

julia> p2 = SVector(1.0, 1.0);

julia> isempty(find_boundary_extrema(p1, corner, p2, ε))
true
```

See also [`detect_skipped_corners`](@ref).
"""
function find_boundary_extrema(
    p1::SVector{2,Float64},
    corner::SVector{2,Float64},
    p2::SVector{2,Float64},
    ε;
    n_sample::Int = 200,
    iter::Int = 64,
)::Vector{SVector{2,Float64}}
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
    mesh_region(region, ε, band_index, e_min, e_max, Δε, n_arc[, N]; <keyword arguments>)

Generate a [`Mesh`](@ref) of momentum-space `region` for dispersion `ε` (band `band_index`)
covering the energy window `[e_min, e_max]`.

Boundary contours are placed at intervals of `Δε` across `[e_min, e_max]`. The arc-length
resolution along each contour is `n_arc` patches per sheet. Multiple disconnected Fermi
surface sheets are handled automatically — one sub-mesh is generated per sheet and the
results are concatenated.

If corner crossings are detected (an isoline endpoint slides around an IBZ corner between
consecutive energy levels), the crossing region is recursively re-meshed up to `maxdepth`
levels.

A 7-argument overload `mesh_region(region, ε, e_min, e_max, Δε, n_arc[, N]; …)` defaults
`band_index` to `1` for single-band use.

# Arguments
- `region`: convex polygon vertices bounding the momentum-space region, as a
  `Vector{<:AbstractVector}`.
- `ε`: dispersion function called as `ε([kx, ky])`.
- `band_index::Int`: band index recorded on each generated patch.
- `e_min`, `e_max`: lower and upper bounds of the energy window.
- `Δε`: target energy spacing between boundary contours.
- `n_arc::Int`: patches per sheet along each energy contour. Clamped to `max(3, n_arc)`.
- `N::Int = 1001`: marching-squares sample count along the longer axis of the bounding
  rectangle.
- `bbox = nothing`: optional bounding-box polygon; when provided the marching-squares grid
  uses its bounding rectangle instead of that of `region`.
- `angle_threshold = π/8`: turning-angle threshold (radians) for cusp detection.
- `boundaries::AbstractVector = []`: explicit boundary energies; when non-empty, overrides
  the automatically computed grid.
- `maxdepth::Int = 2`: maximum recursion depth for corner re-meshing.

See also [`ibz_mesh`](@ref), [`bz_mesh`](@ref).
"""
function mesh_region(region, ε, band_index::Int, e_min, e_max, Δε, n_arc::Int, N = 1001;
                     bbox = nothing, angle_threshold = π/8, boundaries::AbstractVector = [], depth = 1, maxdepth = 2)
    all_patches    = Patch[]
    all_corners    = SVector{2,Float64}[]
    all_corner_ids = SVector{4,Int}[]
    n_arc = max(3, n_arc)

    # Sample coordinates and energies for marching squares
    ((x_min, x_max), (y_min, y_max)) = get_bounding_box(isnothing(bbox) ? region : bbox)
    Δx = (x_max - x_min) / (N - 1)
    Ny = round(Int, (y_max - y_min) / Δx)
    N  < 2 || Ny < 2 && return Mesh(all_patches, all_corners, all_corner_ids)

    X = LinRange(x_min, x_max, N)
    Y = LinRange(y_min, y_max, Ny)
    E = Matrix{Float64}(undef, N, Ny)
    for (i, x) in enumerate(X)
        for (j, y) in enumerate(Y)
            E[i,j] = ε([x, y])
        end
    end

    be = length(boundaries) > 0 ? sort(unique(boundaries)) : boundary_energies(X, Y, E, ε, region, e_min, e_max, Δε)
    fe = foliate(be)
    c = contours(X, Y, E, fe; mask = region)

    if depth < maxdepth # Recursive step on skipped corner regions.
        # Corner energies that may be pinned by `boundary_energies`; matches the
        # set of levels excluded there so the two stay in sync.
        corner_levels = [ε(k) for k in region if k != [0.0, 0.0]]
        skips = detect_skipped_corners(c, region; corner_levels)

        for (i, s, corner) in skips
            corner_e_min = c[i].level - Δε
            corner_e_max = c[i+2].level + Δε
            corner_Δε = 0.1 * Δε * (corner_e_max - corner_e_min) / (e_max - e_min)
            corner_be = ε.(corner)
            boundary_extrema = ε.(find_boundary_extrema(corner..., ε))

            append!(corner_be, boundary_extrema)
            sort!(corner_be)
            if length(corner_be) > 1
                corner_be[begin] += eps(Float64)
                corner_be[end]   -= eps(Float64)
            end

            δL0 = c[i].isolines[s].arclength / n_arc
            longest_leg = 0.0
            for i ∈ eachindex(corner)
                leg = norm(corner[mod1(i + 1, 3)] - corner[i])
                if leg > longest_leg
                    longest_leg = leg
                end
            end

            corner_n_arc = round(Int, 0.5 * longest_leg / δL0)
            corner_mesh = mesh_region(corner, ε, band_index, corner_e_min, corner_e_max, corner_Δε, corner_n_arc, N; boundaries = corner_be, depth = depth + 1, maxdepth)
            append!(all_patches,    corner_mesh.patches)
            append!(all_corners,    corner_mesh.corners)
            append!(all_corner_ids, corner_mesh.corner_indices)
        end
    end

    n_isolines = [length(bundle.isolines) for bundle in c]

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

        (n_iso == 0 || (run.stop == run.start && isodd(run.start))) && continue

        need_lo = iseven(run.start)
        need_hi = iseven(run.stop)
        shift_lo = run.start != 1 && isodd(run.start)
        shift_hi = run.stop != length(n_isolines) && isodd(run.stop)

        if need_lo || shift_lo
            e_lo_m    = find_marginal_energy(X, Y, E, region,
                            fe[run.start], fe[run.start - 1], n_iso)
            bundle_lo = contours(X, Y, E, [e_lo_m]; mask=region)[1]
        end
        if need_hi || shift_hi
            e_hi_m    = find_marginal_energy(X, Y, E, region,
                            fe[run.stop], fe[run.stop + 1], n_iso)
            bundle_hi = contours(X, Y, E, [e_hi_m]; mask=region)[1]
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
            append!(all_corner_ids, mesh_s.corner_indices)
        end
    end

    return Mesh(all_patches, all_corners, all_corner_ids)
end

mesh_region(region, ε, e_min, e_max, Δε, n_arc::Int, N = 1001;
            bbox = nothing, angle_threshold = π/8, boundaries::AbstractVector = [], maxdepth::Int = 2) =
    mesh_region(region, ε, 1, e_min, e_max, Δε, n_arc, N; bbox, angle_threshold, boundaries, maxdepth)

###
### Public mesh API
###

"""
    ibz_mesh(l, bands, T, Δε, n_arc[, N, α]; <keyword arguments>)

Generate a mesh of the irreducible Brillouin zone (IBZ) for lattice `l` given dispersions
in `bands` at temperature `T`.

The energy window covered is `[-α T, +α T]`. The `Function` overload accepts a single
dispersion directly in place of `bands`.

# Arguments
- `l::Lattice`: lattice whose IBZ is meshed.
- `bands::AbstractVector`: vector of band dispersions, each callable on a 2-component
  momentum.
- `T`: temperature in the same units as the band energies (eV by package convention).
- `Δε`, `n_arc`, `N`: resolution parameters forwarded to [`mesh_region`](@ref); see
  there for details.
- `α::Real = 6.0`: half-width of the energy window in units of `T`.
- `mask = SVector{2,Float64}[]`: optional convex polygon defining a sub-region of the IBZ.
  When `length(mask) > 2` the marching-squares grid uses the bounding rectangle of `mask`
  instead of that of the full IBZ — useful for finer gridding around a small Fermi surface
  pocket. Contour extraction is still masked by the full IBZ.
- `angle_threshold = π/8`, `boundaries::AbstractVector = []`, `maxdepth::Int = 2`:
  forwarded to [`mesh_region`](@ref).

See also [`bz_mesh`](@ref), [`mesh_region`](@ref).
"""
function ibz_mesh(l::Lattice, bands::AbstractVector, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0;
                  mask = SVector{2,Float64}[], angle_threshold = π/8,
                  boundaries::AbstractVector = [], maxdepth::Int = 2)
    full_patches    = Patch[]
    full_corners    = SVector{2,Float64}[]
    full_corner_ids = SVector{4,Int}[]
    ibz  = get_ibz(l)
    bbox = length(mask) > 2 ? mask : nothing

    for i in eachindex(bands)
        mesh = mesh_region(ibz, bands[i], i, -α * T, α * T, Δε, n_arc, N; bbox, angle_threshold, boundaries, maxdepth)
        ℓ = length(full_corners)
        append!(full_patches, mesh.patches)
        append!(full_corner_ids, map(x -> SVector{4,Int}(x .+ ℓ), mesh.corner_indices))
        append!(full_corners, mesh.corners)
    end

    return Mesh(full_patches, full_corners, full_corner_ids)
end

ibz_mesh(l::Lattice, ε::Function, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0;
         mask = SVector{2,Float64}[], angle_threshold = π/8,
         boundaries::AbstractVector = [], maxdepth::Int = 2) =
    ibz_mesh(l, [ε], T, Δε, n_arc, N, α; mask, angle_threshold, boundaries, maxdepth)

"""
    bz_mesh(l, bands, T, Δε, n_arc[, N, α]; <keyword arguments>)

Generate a mesh of the full Brillouin zone (BZ) for lattice `l` given dispersions in
`bands` at temperature `T`, by computing the IBZ mesh and replicating it under every
point-group operation of `l`.

The energy window covered is `[-α T, +α T]`. The output contains
`order(point_group(l)) × N_ibz` patches per band, where `N_ibz` is the patch count
returned by [`ibz_mesh`](@ref) for the same arguments. The `Function` overload accepts a
single dispersion directly in place of `bands`.

# Arguments
- `l::Lattice`: lattice whose BZ is meshed.
- `bands::AbstractVector`: vector of band dispersions, each callable on a 2-component
  momentum.
- `T`: temperature in the same units as the band energies.
- `Δε`, `n_arc`, `N`: resolution parameters forwarded to [`mesh_region`](@ref).
- `α::Real = 6.0`: half-width of the energy window in units of `T`.
- `mask = SVector{2,Float64}[]`: optional convex polygon restricting the marching-squares
  grid to a sub-region of the IBZ before symmetry replication.
- `angle_threshold = π/8`, `boundaries::AbstractVector = []`, `maxdepth::Int = 2`:
  forwarded to [`mesh_region`](@ref).

See also [`ibz_mesh`](@ref), [`bz_symmetry_map`](@ref), [`mesh_region`](@ref).
"""
function bz_mesh(l::Lattice, bands::AbstractVector, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0;
                 mask = SVector{2,Float64}[], angle_threshold = π/8,
                 boundaries::AbstractVector = [], maxdepth::Int = 2)
    G     = point_group(l)
    ibz   = get_ibz(l)
    θperm = group_angle_perm(G, ibz)

    full_patches     = Patch[]
    full_corners     = SVector{2,Float64}[]
    full_corner_inds = SVector{4,Int}[]

    bbox = length(mask) > 2 ? mask : nothing

    for j in eachindex(bands)
        mesh = mesh_region(ibz, bands[j], j, -α * T, α * T, Δε, n_arc, N; bbox, angle_threshold, boundaries, maxdepth)

        for i in θperm
            O = get_matrix_representation(G.elements[i])
            ℓ = length(full_corners)
            append!(full_patches,     map(x -> patch_op(x, O), mesh.patches))
            append!(full_corner_inds, map(x -> SVector{4,Int}(x .+ ℓ), mesh.corner_indices))
            append!(full_corners,     map(x -> O * x, mesh.corners))
        end
    end

    return Mesh(full_patches, full_corners, full_corner_inds)
end

bz_mesh(l::Lattice, ε, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0;
        mask = SVector{2,Float64}[], angle_threshold = π/8,
        boundaries::AbstractVector = [], maxdepth::Int = 2) =
    bz_mesh(l, [ε], T, Δε, n_arc, N, α; mask, angle_threshold, boundaries, maxdepth)

###
### Isotropic mesh
###

"""
    radial_extrema(ε, kf, e_window; N_prelim=100)

Locate radii at which the radial dispersion `ε(r)` has critical points, plus an outer
radius `r_max` enclosing the energy window of width `e_window` beyond the outermost Fermi
crossing.

`kf` is the sorted vector of Fermi crossings of `ε`. Returns `(critical_radii, r_max)`,
where `critical_radii` is the sorted vector of stationary points of `ε` either between
consecutive Fermi crossings or inside `[0, kf[1]]`. Used to bracket monotone radial
sections for [`isotropic_mesh`](@ref).
"""
function radial_extrema(ε, kf::AbstractVector{<:Real}, e_window; N_prelim = 100)
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

"""
    isotropic_sheet_mesh(ε, energies_s, n_arc, corner_offset, r_lo, r_hi)

Mesh a single annular section of an isotropic Fermi surface for the boundary energy
levels in `energies_s` between radii `r_lo` and `r_hi`.

`ε` is the radial dispersion (a function of radius), assumed monotone on the section. The
section is divided into `n_arc` angular cells; `corner_offset` is the index of the first
corner in the parent mesh's corner vector, used to compute global corner indices.

Returns `(corners, k, corner_inds, patches)` for the section.
"""
function isotropic_sheet_mesh(ε, energies_s, n_arc, corner_offset, r_lo, r_hi)
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

            Jinv = SMatrix{2,2,Float64,4}(
                 (Δε_i / 2) * cos(θ) / norm(v[i, j]),
                 (Δε_i / 2) * sin(θ) / norm(v[i, j]),
                -(Δθ / 2) * norm(k[i, j]) * sin(θ),
                 (Δθ / 2) * norm(k[i, j]) * cos(θ),
            )

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
