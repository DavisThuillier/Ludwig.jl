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
        p.e,
        SVector{2}(M * p.k), 
        SVector{2}(M * p.v),
        p.band_index
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

function boundary_energies(X, Y, E, ε, region, e_min, e_max, Δε)
    n_levels = max(1, ceil(Int, (e_max - e_min) / Δε)) # always even → E=0 on a patch center 
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
        k == [0.0, 0.0] && continue
        corner_e =  ε(k)
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
        k_inner = aligned_arclength_slice(
            sheet_isolines[i-1], sheet_isolines[i], 2 * n_arc_s + 1;
            cusp_fractions, angle_threshold,
        )
        k_outer = aligned_arclength_slice(
            sheet_isolines[i+1], sheet_isolines[i], 2 * n_arc_s + 1;
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

"""
    detect_skipped_corners(c, region, ε[; tol])

Detect crossings of open isoline endpoints across corners of the IBZ polygon `region`
as energy varies through the bundles in `c`.

For each adjacent bundle pair `(c[i], c[i+2])` sharing a common sheet count, checks
whether one endpoint of an open isoline slides from one IBZ edge to an adjacent edge.

# Arguments
- `c::Vector{IsolineBundle}`: isoline bundles ordered by energy.
- `region`: IBZ polygon vertices as a counterclockwise `Vector{SVector{2,Float64}}`.
- `ε`: dispersion function accepting an `AbstractVector` of length 2.
- `tol`: collinearity/range tolerance passed to `edge_index` (default `1e-8`).

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
    region,
    ε;
    tol = 1e-8,
)
    events = Tuple{Int, Int, Vector{SVector{2, Float64}}}[]

    n_reg = length(region)
    length(c) == 0 && return events

    for i in 1:2:(length(c) - 2)
        b1 = c[i].isolines
        b2 = c[i + 2].isolines

        # Check that neither contour is at a corner energy
        # minimum(abs.(ε.(region) .- c[i].level)) < tol && continue
        # minimum(abs.(ε.(region) .- c[i+2].level)) < tol && continue

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
            e1 = [edge_index(p11, region), edge_index(p12, region)]
            e2 = [edge_index(p21, region), edge_index(p22, region)]

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
julia> p1 = SVector(0.0, 0.0); corner = SVector(1.0, 0.0); p2 = SVector(1.0, 1.0);
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
    mesh_region(region, ε, band_index::Int, e_min, e_max, Δε, n_arc::Int[, N::Int]; kwargs...)

Generate a [`Mesh`](@ref) of momentum-space `region` for dispersion `ε` (band `band_index`)
covering the energy window `[e_min, e_max]`.

Boundary contours are placed at intervals of `Δε` across `[e_min, e_max]`. The arc-length
resolution along each contour is `n_arc` patches per sheet. Multiple disconnected Fermi surface
sheets are handled automatically — one sub-mesh is generated per sheet and the results are
concatenated.

If corner crossings are detected (an isoline endpoint slides around an IBZ corner between
consecutive energy levels), the crossing region is recursively re-meshed up to `maxdepth`
recursions. `N` sets the number of sample points along the longer axis of the bounding rectangle
for the underlying marching-squares contour extraction.

# Arguments
- `bbox`: optional bounding-box polygon; when provided the marching-squares grid uses its
  bounding rectangle instead of that of `region`.
- `angle_threshold`: turning-angle threshold (radians) for cusp detection (default `π/8`).
- `boundaries`: if non-empty, use these explicit boundary energies instead of the
  automatically computed grid.
- `depth`: current recursion depth (default `1`; do not set manually).
- `maxdepth`: maximum recursion depth for corner re-meshing (default `2`).

See also [`ibz_mesh`](@ref), [`bz_mesh`](@ref), [`mesh_sheet`](@ref).
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
    @show be
    fe = foliate(be)
    c = contours(X, Y, E, fe; mask = region)
    # @show map(x -> map(y -> y.arclength, x.isolines), c)

    if depth < maxdepth # Recursive step on skipped corner regions.
        skips = detect_skipped_corners(c, region, ε)
        for (i, s, corner) in skips
            corner_e_min = c[i].level - Δε
            corner_e_max = c[i+2].level + Δε
            corner_Δε = 0.1 * Δε * (corner_e_max - corner_e_min) / (e_max - e_min)
            boundary_energies = ε.(corner)
            @show boundary_energies
            # boundary_energies = [c[i].level, c[i+2].level] # Initially populate with vertex energies
            
            boundary_extrema = ε.(find_boundary_extrema(corner..., ε))
            # for (i, extremum) ∈ enumerate(boundary_extrema)
            #     n_iso = length(contours(X, Y, E, [extremum]; mask = region)[1].isolines)
            #     @show n_iso
            #     e1 = extremum + eps(Float64)
            #     e2 = extremum - eps(Float64)
            #     δn_iso = length.(map(x -> x.isolines, contours(X, Y, E, [e1, e2]; mask = region))) .- n_iso
            #     @show δn_iso
            #     boundary_extrema[i] = [e1, e2][argmin(δn_iso)]
            # end

            append!(boundary_energies, boundary_extrema)
            sort!(boundary_energies)
            if length(boundary_energies) > 1
                boundary_energies[begin] += eps(Float64)
                boundary_energies[end]   -= eps(Float64)
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
            corner_mesh = mesh_region(corner, ε, band_index, corner_e_min, corner_e_max, corner_Δε, corner_n_arc, N; boundaries = boundary_energies, depth = depth + 1, maxdepth)
            @show unique(energy.(corner_mesh.patches))
            append!(all_patches,    corner_mesh.patches)
            append!(all_corners,    corner_mesh.corners)
            append!(all_corner_ids, corner_mesh.corner_inds)
        end
    end

    # split_range_at_crossings(eachindex(c), crossings)

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

        # n_iso == 0 && continue
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
            append!(all_corner_ids, mesh_s.corner_inds)
        end
    end

    return Mesh(all_patches, all_corners, all_corner_ids)
end

mesh_region(region, ε, e_min, e_max, Δε, n_arc::Int, N = 1001;
            bbox = nothing, angle_threshold = π/8, boundaries::AbstractVector = [], maxdepth::Int = 2) =
    mesh_region(region, ε, 1, e_min, e_max, Δε, n_arc, N; bbox, angle_threshold, boundaries, maxdepth)

"""
    ibz_mesh(l::Lattice, bands::AbstractVector, T, Δε, n_arc::Int[, N::Int, α::Real]; mask, angle_threshold, boundaries, maxdepth)

Generate a mesh of the irreducible Brillouin Zone (IBZ) for lattice `l` given dispersions
in `bands` at temperature `T`. See [`mesh_region`](@ref) for a description of the resolution
parameters `Δε` and `n_arc`.

# Arguments
- `mask`: optional vector of vertices defining a convex polygonal sub-region of the IBZ. When
  `length(mask) > 2` the mesh is restricted to `mask` instead of the full IBZ, which is
  useful for finer gridding around a small Fermi surface pocket.
- `angle_threshold`: turning-angle threshold (radians) for cusp detection; passed through to
  [`mesh_region`](@ref) (default `π/8`).
- `boundaries`: explicit boundary energy vector passed through to [`mesh_region`](@ref);
  when non-empty, overrides the automatically computed energy grid (default `[]`).
- `maxdepth`: maximum recursion depth for corner re-meshing passed through to
  [`mesh_region`](@ref) (default `2`).
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
        append!(full_corner_ids, map(x -> SVector{4,Int}(x .+ ℓ), mesh.corner_inds))
        append!(full_corners, mesh.corners)
    end

    return Mesh(full_patches, full_corners, full_corner_ids)
end

ibz_mesh(l::Lattice, ε::Function, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0;
         mask = SVector{2,Float64}[], angle_threshold = π/8,
         boundaries::AbstractVector = [], maxdepth::Int = 2) =
    ibz_mesh(l, [ε], T, Δε, n_arc, N, α; mask, angle_threshold, boundaries, maxdepth)

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
    bz_mesh(l::Lattice, bands::AbstractVector, T, Δε, n_arc::Int[, N::Int, α::Real]; mask, angle_threshold, boundaries, maxdepth)

Generate a mesh of the Brillouin Zone (BZ) for lattice `l` given dispersions in `bands` at
temperature `T` by computing the IBZ mesh and replicating it under all point-group operations.
See [`mesh_region`](@ref) for a description of `Δε` and `n_arc`.

# Arguments
- `mask`: optional vector of vertices passed through to [`mesh_region`](@ref). When
  `length(mask) > 2` the IBZ mesh is restricted to `mask` instead of the full IBZ before
  being replicated across the BZ.
- `angle_threshold`: turning-angle threshold (radians) for cusp detection; passed through to
  [`mesh_region`](@ref) (default `π/8`).
- `boundaries`: explicit boundary energy vector passed through to [`mesh_region`](@ref);
  when non-empty, overrides the automatically computed energy grid (default `[]`).
- `maxdepth`: maximum recursion depth for corner re-meshing passed through to
  [`mesh_region`](@ref) (default `2`).
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
            append!(full_corner_inds, map(x -> SVector{4,Int}(x .+ ℓ), mesh.corner_inds))
            append!(full_corners,     map(x -> O * x, mesh.corners))
        end
    end

    return Mesh(full_patches, full_corners, full_corner_inds)
end

bz_mesh(l::Lattice, ε, T, Δε, n_arc::Int, N::Int = 1001, α::Real = 6.0;
        mask = SVector{2,Float64}[], angle_threshold = π/8,
        boundaries::AbstractVector = [], maxdepth::Int = 2) =
    bz_mesh(l, [ε], T, Δε, n_arc, N, α; mask, angle_threshold, boundaries, maxdepth)

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
