abstract type AbstractPatch end

"""
    Patch(e::Float64, k::SVector{2,Float64}, v::SVector{2,Float64}, de::Float64, dV::Float64, jinv::Matrix{Float64}, djinv::Float64, band_index::Int)

Construct a `Patch' object defining regions of momentum space over which to integrate. 
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

Construct a `VirtualPatch' object that can be operated on as if it were a `Patch` for the purposes of sampling momentum, energy, and group velocity but which cannot be integrated over. 
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

"""
    patch_op(p::VirtualPatch, M::Matrix)

Apply the active transformation defined by matrix `M` on the relevant fields of virtual patch `p`.
"""
function patch_op(p::VirtualPatch, M::Matrix)
    return VirtualPatch(
        p.energy,
        SVector{2}(M * p.momentum), 
        SVector{2}(M * p.v),
        band_index
    )
end

"""
    Mesh(patches::Vector{Patch}, corners::Vector{SVector{2,Float64}}, corner_inds::Vector{SVector{4,Int}}})

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

Precomputed symmetry information for a BZ mesh, relating each non-IBZ patch to its IBZ
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

    θperm = _group_angle_perm(G, ibz)

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
# Nonuniform Energy Gridding
###

function populate_abscissas!(x, δ)
    Δx = 0.0
    for i ∈ eachindex(x)
        if i == 1
            x[i] = δ
        else
            x[i] = x[i-1] + Δx
        end
        Δx = 4 * δ * cosh(x[i]/2)^2
    end
    return nothing
end

function get_abscissas(n, α, threshold = 1e-10, max_iter = 1000)
    x = Vector{Float64}(undef, n)

    ϵ = α / n
    iter = 1
    step = 0.1 * ϵ^(1/3)
    populate_abscissas!(x, ϵ^(1/3))
    old_sign = sign(x[end] - α)

    while abs(x[end] - α) >= threshold && iter <= max_iter
        new_sign = sign(x[end] - α)
        if new_sign != old_sign
            old_sign = new_sign
            step /= 10.0
        end
        while ϵ - new_sign * step < 0
            step /= 2.0
        end
        ϵ += - new_sign * step
        populate_abscissas!(x, ϵ^(1/3))

        iter += 1
    end

    return vcat(-reverse(x), x)#, ϵ
end

###
# General IBZ Meshes
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

###
### _foliated_energies
###

"""
    _foliated_energies(X, Y, E, ε, region, Δε, α, T)

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
function _foliated_energies(X, Y, E, ε, region, Δε, α, T)
    n_levels = 2 * max(1, ceil(Int, α * T / Δε)) # always even → E=0 on a patch center
    energies = collect(LinRange(-α * T, α * T, n_levels))

    e_actual_min, e_actual_max = if length(region) > 2
        lo, hi = Inf, -Inf
        second_lo, second_hi = Inf, -Inf
        for (i, x) in enumerate(X), (j, y) in enumerate(Y)
            isnan(E[i, j]) && continue
            in_polygon([x, y], region) || continue
            e = E[i, j]
            if e < lo
                second_lo, lo = lo, e
            elseif e > lo && e < second_lo
                second_lo = e
            end
            if e > hi
                second_hi, hi = hi, e
            elseif e < hi && e > second_hi
                second_hi = e
            end
        end
        if isinf(lo)
            E_valid = E[.!isnan.(E)]
            minimum(E_valid), maximum(E_valid)
        else
            isinf(second_lo) ? lo : second_lo,
            isinf(second_hi) ? hi : second_hi
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
            i == 1 && continue # corner is below the first interior level; skip
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
    _aligned_arclength_slice(iso, ref_start, ref_tangent, n)

Resample `iso` at `n` arclength-uniform points, aligned to `ref_start`.

The endpoint of `iso.points` closest to `ref_start` is chosen as the starting
point of the resampled arc. The traversal direction is then checked against
`ref_tangent` (the tangent of the reference contour at its start): if the dot
product of the adjacent arc's initial tangent with `ref_tangent` is negative, the
arc is reversed so that all contours in a sheet are traversed in the same direction.
"""
function _aligned_arclength_slice(iso::Isoline, ref_start, ref_tangent, n::Int)
    pts = iso.points
    if norm(pts[end] - ref_start) < norm(pts[begin] - ref_start)
        pts = reverse(pts)
    end
    if length(pts) >= 2 && dot(pts[2] - pts[1], ref_tangent) < 0
        pts = reverse(pts)
    end
    return arclength_slice(pts, n)[1]
end

"""
    _mesh_sheet(sheet_isolines, ε, band_index, n_arc_s, foliated_energies, corner_offset, region)

Generate a [`Mesh`](@ref) for a single Fermi surface sheet defined by `sheet_isolines`.

`sheet_isolines[i]` is the isoline at `foliated_energies[i]` for this sheet (no `nothing`
entries — call site must validate). `n_arc_s` is the number of arc-length divisions of the
center contour. All corner indices in the returned `Mesh` are offset by `corner_offset` so
that the caller can concatenate multiple sheet meshes into a single flat `Mesh`. `region`
is the masking polygon passed to [`mesh_region`](@ref) and is used for kink detection.
"""
function _mesh_sheet(
    sheet_isolines::Vector{<:Isoline},
    ε,
    band_index::Int,
    n_arc_s::Int,
    foliated_energies::Vector{Float64},
    corner_offset::Int,
    region,
)
    n_levels = (length(foliated_energies) + 1) ÷ 2

    patches    = Matrix{Patch}(undef, n_levels - 1, n_arc_s)
    corners    = Vector{SVector{2, Float64}}(undef, (2 * n_levels - 2) * (n_arc_s + 1))
    corner_ids = Matrix{SVector{4, Int}}(undef, n_levels - 1, n_arc_s)

    cind = 1

    for i in 2:2:2*n_levels-1
        arclength_fractions = LinRange(0, 1, 2 * n_arc_s + 1)

        k, arclengths = arclength_slice(sheet_isolines[i].points, 2 * n_arc_s + 1)
        ref_tangent = k[2] - k[1]

        k_inner = _aligned_arclength_slice(sheet_isolines[i-1], k[1], ref_tangent, 2 * n_arc_s + 1)
        k_outer = _aligned_arclength_slice(sheet_isolines[i+1], k[1], ref_tangent, 2 * n_arc_s + 1)

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

            sp1 = sortperm(get_arclengths(sheet_isolines[i-1].points) / sheet_isolines[i-1].arclength; by = x -> abs(x - arclength_fractions[j-1]))
            ij1 = length(sp1) >= 2 ? sp1[1:2] : [sp1[1], sp1[1]]
            sp2 = sortperm(get_arclengths(sheet_isolines[i+1].points) / sheet_isolines[i+1].arclength; by = x -> abs(x - arclength_fractions[j+1]))
            ij2 = length(sp2) >= 2 ? sp2[1:2] : [sp2[1], sp2[1]]

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

            dV = poly_area(vcat(corners[cind-4:cind-1], sheet_isolines[i-1].points[min(i1,i3):max(i1,i3)], sheet_isolines[i+1].points[min(i2,i4):max(i2,i4)]), k[j])

            ij3 = ij1; ij4 = ij2

            # dV = poly_area(vcat(k_inner[j-1:j+1], k_outer[j-1:j+1]), k[j])

            Δε = foliated_energies[i+1] - foliated_energies[i-1]
            Δs = arclengths[j+1] - arclengths[j-1]

            v  = gradient(ε, k[j])
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
    get_arclengths(curve::AbstractVector)

Get the distance to each point along `curve`. Treats `curve` as an ordered list of points defining a piecewise linear path.
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
    arclength_slice(curve::AbstractVector, n::Int)

Cut `curve` into `n` uniformly spaced points. Treats `curve` as an ordered list of points defining a piecewise linear path, and linear interpolation is used to find intermediate points. Returns interpolated points and arclengths along original curve of interpolated points.

For best results, points in `curve` should have approximately uniform spacing and `n` should be much less than the number of points in `curve`.  
"""
function arclength_slice(curve, n::Int)
    if length(curve) == 1
        return fill(curve[begin], n), zeros(Float64, n)
    end

    arclengths = get_arclengths(curve)
    
    τ = LinRange(0.0, arclengths[end], n)

    grid = [curve[begin]]

    j = 1
    jmax = length(arclengths)
    for i in eachindex(τ)
        i == 1 && continue
        while j < jmax && arclengths[j] <= τ[i]
            j += 1
        end

        # Perform linear interpolation between curve points
        push!(grid, curve[j-1] + (curve[j] - curve[j - 1]) * (τ[i] - arclengths[j-1]) / (arclengths[j] - arclengths[j-1]))
    end

    return grid, collect(τ)   
end

"""
    _find_marginal_energy(X, Y, E_mat, region, e_inner, e_outer, target_n_iso)

Bisect in the interval `[e_inner, e_outer]` to find the energy threshold at which the
number of isolines transitions through `target_n_iso`. `e_inner` must satisfy
`n_iso >= target_n_iso`; `e_outer` must not.

Returns the last energy (closest to `e_outer`) where `n_iso >= target_n_iso` still holds.
"""
function _find_marginal_energy(X, Y, E_mat, region, e_target, e_outer, n_iso_target; iter = 32)
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

"""
    mesh_region(region, ε::Function, band_index::Int, T::Real, Δε::Real, n_arc::Int[, N::Int, α::Real])

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
function mesh_region(region, ε, band_index::Int, T, Δε, n_arc::Int, N = 1001, α = 6.0)
    n_arc = max(3, n_arc)

    # Sample coordinates and energies for marching squares
    ((x_min, x_max), (y_min, y_max)) = get_bounding_box(region)
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

    foliated_energies = _foliated_energies(X, Y, E, ε, region, Δε, α, T)
    n_levels = (length(foliated_energies) + 1) ÷ 2

    c = contours(X, Y, E, foliated_energies; mask = region)
    sort_isolines!.(c)

    n_isolines = [length(bundle.isolines) for bundle in c]

    all_patches    = Patch[]
    all_corners    = SVector{2,Float64}[]
    all_corner_ids = SVector{4,Int}[]

    # Partition energy levels into contiguous runs with the same isoline count,
    # then process each sheet within each run independently.
    #
    # _mesh_sheet requires foliated_energies to start and end at boundary energies
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
            e_lo_m    = _find_marginal_energy(X, Y, E, region,
                            foliated_energies[run.start], foliated_energies[run.start - 1], n_iso)
            bundle_lo = contours(X, Y, E, [e_lo_m]; mask=region)[1]
            sort_isolines!(bundle_lo)
        end
        if need_hi || shift_hi
            e_hi_m    = _find_marginal_energy(X, Y, E, region,
                            foliated_energies[run.stop], foliated_energies[run.stop + 1], n_iso)
            bundle_hi = contours(X, Y, E, [e_hi_m]; mask=region)[1]
            sort_isolines!(bundle_hi)
        end

        for s in 1:n_iso
            base_isolines = Isoline[c[idx].isolines[s] for idx in run]
            base_energies = collect(foliated_energies[run])

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

            mesh_s = _mesh_sheet(
                isolines_s, ε, band_index, n_arc,
                foliated_energies_s, length(all_corners), region
            )
            append!(all_patches,    mesh_s.patches)
            append!(all_corners,    mesh_s.corners)
            append!(all_corner_ids, mesh_s.corner_inds)
        end
    end

    return Mesh(all_patches, all_corners, all_corner_ids)
end

mesh_region(region, ε, T, Δε, n_arc::Int, N = 1001, α = 6.0) = mesh_region(region, ε, 1, T, Δε, n_arc, N, α)

"""
    ibz_mesh(l::Lattice, bands::AbstractVector, T, Δε, n_arc::Int[, N::Int, α::Real])

Generate a mesh of the irreducible Brillouin Zone (IBZ) for lattice `l` given dispersions
in `bands` at temperature `T`. See [`mesh_region`](@ref) for a description of the resolution
parameters `Δε` and `n_arc`.
"""
function ibz_mesh(l::Lattice, bands::AbstractVector, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0)
    full_patches    = Patch[]
    full_corners    = SVector{2,Float64}[]
    full_corner_ids = SVector{4,Int}[]
    ibz = get_ibz(l)

    for i in eachindex(bands)
        mesh = mesh_region(ibz, bands[i], i, T, Δε, n_arc, N, α)
        ℓ = length(full_corners)
        append!(full_patches, mesh.patches)
        append!(full_corner_ids, map(x -> SVector{4,Int}(x .+ ℓ), mesh.corner_inds))
        append!(full_corners, mesh.corners)
    end

    return Mesh(full_patches, full_corners, full_corner_ids)
end

ibz_mesh(l::Lattice, ε::Function, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0) = ibz_mesh(l, [ε], T, Δε, n_arc, N, α)

"""
    _group_angle_perm(G, ibz) -> Vector{Int}

Return the permutation that sorts the elements of point group `G` by the polar angle of
the image of the IBZ centroid under each element's matrix representation.

Concretely, for each group element `g` with 2×2 matrix `O`, the angle
`θ_g = atan(k[2], k[1]) mod 2π` is computed where `k = O * centroid(ibz)`.
The returned index vector `p` satisfies `θ_{p[1]} ≤ θ_{p[2]} ≤ … ≤ θ_{p[|G|]}`.

This ordering assigns each group element to a unique angular sector of the BZ and is the
canonical sector ordering shared by [`bz_mesh`](@ref) and [`bz_symmetry_map`](@ref).
"""
function _group_angle_perm(G, ibz)
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
    bz_mesh(l::Lattice, bands::AbstractVector, T, Δε, n_arc::Int[, N::Int, α::Real])

Generate a mesh of the Brillouin Zone (BZ) for lattice `l` given dispersions in `bands` at
temperature `T` by computing the IBZ mesh and replicating it under all point-group operations.
See [`mesh_region`](@ref) for a description of `Δε` and `n_arc`.
"""
function bz_mesh(l::Lattice, bands::AbstractVector, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0)
    G     = point_group(l)
    ibz   = get_ibz(l)
    θperm = _group_angle_perm(G, ibz)

    full_patches     = Patch[]
    full_corners     = SVector{2,Float64}[]
    full_corner_inds = SVector{4,Int}[]

    for j in eachindex(bands)
        mesh = mesh_region(ibz, bands[j], j, T, Δε, n_arc, N, α)

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

bz_mesh(l::Lattice, ε, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0) = bz_mesh(l, [ε], T, Δε, n_arc, N, α)

"""
    secant_method(f, x0, x1, maxiter[; atol])

Find a root of the function `f` in the interval [`x0`, `x1`] via the secant method. Root finding will iterate until `maxiter` iterations is reached or the root is within the absolute tolerance `atol`.
"""
function secant_method(f, x0, x1, maxiter; atol = eps(Float64))
    x2 = 0.0
    for i in 1:maxiter
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x2
        abs(f(x2)) < atol && break
    end

    return x2
end

"""
    circular_mesh(ε::Function, T::Real, n_levels::Int, n_angles::Int[, α::Real; maxiter, atol])

Generate a mesh for the isotropic Fermi surface given radial dispersion `ε` at temperature `T`. The resultant mesh covers an annular region of the Fermi surface between -`αT` and + `αT` with `n_levels - 1` patches in the energy direction and `n_angles - 1` patches along the energy contours.

The parameters `maxiter` and `atol` determine the maximum number of iterations and absolute tolerance used for inverting the dispersion to find the radial distance corresponding to the energy contours. 
"""
function circular_fs_mesh(ε, T::Real, n_levels::Int, n_angles::Int, α::Real = 6.0; maxiter = 10000, atol = 1e-10)
    n_levels = max(4, n_levels) # Enforce minimum of 3 patches in the energy direction
    if isodd(n_levels)
        n_levels += 1
    end
    n_angles = max(3, n_angles) # Enforce minimum of 2 patches in angular direction
    Δθ = 2π / (n_angles - 1) 
    Δε = 2*α*T / (n_levels - 1) # From uniform spacing for sense of energy scale
    
    # Slice contours to generate patch corners
    corners = Matrix{SVector{2,Float64}}(undef, n_levels, n_angles)
    k = Matrix{SVector{2,Float64}}(undef, n_levels-1, n_angles-1)
    corner_indices = Matrix{SVector{4,Int}}(undef, n_levels-1, n_angles-1)
    radius = Vector{Float64}(undef, 2*n_levels - 1)
    energies = LinRange(-α, α, n_levels) * T

    for i ∈ 1:2*n_levels-1
        if isodd(i)
            E = energies[(i+1)÷2]
        else
            E = (energies[i÷2+1] + energies[i÷2]) / 2.0
        end

        r0 = i == 1 ? 0.0 : radius[i-1]
        r1 = r0 + Δε / derivative(ε, r0)
        isinf(r1) && (r1 = Δε)
        radius[i] = secant_method(r -> ε(r) - E, r0, r1, maxiter; atol)

        if isodd(i)
            corners[i÷2 + 1, :] = map(θ -> SVector{2, Float64}(radius[i]*cos(θ), radius[i]*sin(θ)), LinRange(0, 2π, n_angles))
        else
            k[i÷2, :] = map(θ -> SVector{2, Float64}(radius[i]*cos(θ), radius[i]*sin(θ)), (Δθ/2):Δθ:2π)
            for j in 1:n_angles-1
                ref_index = (j - 1) * n_levels + (i÷2)
                corner_indices[i÷2, j] = [
                    ref_index, 
                    ref_index + 1,
                    ref_index + n_levels + 1,
                    ref_index + n_levels,
                    ]
            end
        end
    end

    v = map(x -> derivative(ε, norm(x)) * x / norm(x), k)

    patches = Matrix{Patch}(undef, n_levels-1, n_angles-1)
    for i in 1:size(patches)[1]
        E = (energies[i+1] + energies[i]) / 2.0
        Δε = energies[i+1] - energies[i] 
        dV = abs(0.5 * Δθ * (radius[2*i + 1]^2 - radius[2*i - 1]^2))
        for j in 1:size(patches)[2]
            θ = atan(k[i,j][2], k[i,j][1])

            Jinv = zeros(Float64, 2, 2)
            Jinv[1,1] = (Δε/2) * cos(θ) / norm(v[i,j])
            Jinv[2,1] = (Δε/2) * sin(θ) / norm(v[i,j])
            Jinv[1,2] = - (Δθ/2) * norm(k[i,j]) * sin(θ)
            Jinv[2,2] = (Δθ/2) * norm(k[i,j]) * cos(θ)

            patches[i, j] = Patch(
                E,   
                k[i,j], 
                v[i,j],
                Δε,
                dV,
                Jinv,
                abs(det(Jinv)),
                1
            )
        end
    end

    return Mesh(vec(patches), vec(corners), vec(corner_indices))
end
