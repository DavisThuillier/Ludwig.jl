"""
    Isoline(points::Vector{SVector{2,Float64}}, isclosed::Bool, arclength::Float64)

Representation of a contour as an ordered set of discrete points.

# Fields
- `points::Vector{SVector{2,Float64}}`: vector of points in the contour.
- `isclosed::Bool`: `true` if the contour returned to its starting point when generated.
- `arclength::Float64`: total length of the contour.
"""
struct Isoline
    points::Vector{SVector{2,Float64}}
    isclosed::Bool
    arclength::Float64
end

"""
    IsolineBundle(isolines::Vector{Isoline}, level::Float64)

Container of contours corresponding to the same level set.

# Fields
- `isolines::Vector{Isoline}`: contours sharing a common level.
- `level::Float64`: constant value along the contours in `isolines`.
"""
struct IsolineBundle
    isolines::Vector{Isoline}
    level::Float64
end

# Top, Bottom, Left, Right binary identification
const T, B, R, L = 0x01, 0x02, 0x04, 0x08

# Lookup table mapping a 4-bit corner-occupancy code (1..14) to the pair of edges crossed
# by the contour. The two saddle cases (intersect codes 5 and 10), in which two contours
# pass through a single cell, are mapped to 0x0 and ignored under the assumption that the
# grid is fine enough that genuine saddles are rare and locally insignificant.
const crossing_lookup = [L|B, B|R, L|R, T|R, 0x0, T|B, L|T, L|T, T|B, 0x0, T|R, L|R, B|R, L|B]

"""
    get_cells(x, y, A::AbstractMatrix [, level = 0.0, mask = []])

Obtain the cells and their crossing classifications for matrix `A` through which the level set corresponding to `level` crosses.

The arguments `x` and `y` define the coordinates of the elements of `A`, allowing for nonuniform gridding.
If `mask` is specified and contains at least three points, it is assumed to be a convex polygon and only cells contained fully within mask will be found.
"""
function get_cells(x, y, A::AbstractMatrix, level::Real = 0.0, mask = [])
    xax, yax = axes(A)

    cells = OrderedDict{Tuple{Int, Int}, UInt8}() # Cells with crossing fully contained within masking polygon
    border_cells = OrderedDict{Tuple{Int, Int}, UInt8}() # Cells with crossing through which the masking polygon passes

    @inbounds for i in first(xax):last(xax)-1
        for j in first(yax):last(yax)-1

            if length(mask) > 2 # Check for intersection with masking polygon
                cell = [[x[i], y[j]], [x[i+1],y[j]], [x[i+1], y[j+1]], [x[i], y[j+1]]]
                inside = in_polygon.(cell, Ref(mask))
                sum(inside) == 0 && continue

                intersect = (A[i, j] > level) ? 0x01 : 0x00
                (A[i + 1, j] > level) && (intersect |= 0x02)
                (A[i + 1, j + 1] > level) && (intersect |= 0x04)
                (A[i, j + 1] > level) && (intersect |= 0x08)

                if !(intersect == 0x00) && !(intersect == 0x0f)
                    if 0 < sum(inside) < 4
                        border_cells[(i,j)] = crossing_lookup[intersect]
                    elseif sum(inside) == 4
                        cells[(i,j)] = crossing_lookup[intersect]
                    end
                end
            else # No mask applied
                intersect = (A[i, j] > level) ? 0x01 : 0x00
                (A[i + 1, j] > level) && (intersect |= 0x02)
                (A[i + 1, j + 1] > level) && (intersect |= 0x04)
                (A[i, j + 1] > level) && (intersect |= 0x08)
                if !(intersect == 0x00) && !(intersect == 0x0f)
                    cells[(i,j)] = crossing_lookup[intersect]
                end
            end
        end
    end

    return cells, border_cells
end

const shift = [(0,1), (0,-1), (1,0), (-1,0)] # Relative position of next cell to check given crossing
const new_edge = (B,T,L,R)

"""
    get_next_cell(edge, index)

Find the next cell through which the contour will pass given the exit `edge` of the current cell and its corresponding `index`.
"""
function get_next_cell(edge, index)
    i = trailing_zeros(edge) + 1
    index = index .+ shift[i]
    return new_edge[i], index
end

"""
    find_contour(x, y, A::AbstractMatrix [, level = 0.0; mask = []])

Get all contours of matrix `A` in terms of the coordinate grids `x` and `y` at `level`.

If `mask` is specified, it is treated as an array of points defining a convex polygonal masking region in terms of the coordinates defined by `x` and `y` to which the contours are constrained.
"""
function find_contour(x, y, A::AbstractMatrix, level::Real = 0.0; mask = [])
    bundle = IsolineBundle(Isoline[], level)
    if level > maximum(A[.!isnan.(A)]) || level < minimum(A[.!isnan.(A)])
        return bundle # No contours to be found
    end

    xax, yax = axes(A) # Valid indices for iterating over

    # Cells are defined by the coordinate of their lower left index:
    is = first(xax):last(xax)-1
    js = first(yax):last(yax)-1

    cells, border_cells = get_cells(x, y, A, level, mask)

    while Base.length(cells) > 0 # Iterate until all cells containing a crossing have been passed through
        segment = Vector{SVector{2,Float64}}(undef, 0) # Initialize empty segment to add to the contour bundle
        start_index, case = first(cells)
        start_edge = 0x01 << trailing_zeros(case)

        end_index = follow_contour!(cells, border_cells, segment, x, y, A, is, js, start_index, start_edge, level; mask)

        # Check if the contour forms a loop
        isclosed = end_index == start_index

        if !isclosed && (length(cells) > 0 || length(border_cells) > 0)
            # Go back to the starting cell and walk the other direction
            edge, index = get_next_cell(start_edge, start_index)
            if haskey(cells, index) || haskey(border_cells, index)
                follow_contour!(cells, border_cells, reverse!(segment), x, y, A, is, js, index, edge, level; mask)
            end
        end

        seg_length = 0.0
        for i in eachindex(segment)
            i == length(segment) && continue
            seg_length += norm(segment[i + 1] - segment[i])
        end
        isclosed && (seg_length += norm(last(segment) - first(segment)))

        remove_duplicates!(segment)

        push!(bundle.isolines, Isoline(segment, isclosed, seg_length))
    end

    return bundle
end

"""
    remove_duplicates!(segment)

Remove consecutive points of `segment` that coincide within `rtol = 1e-10` and return the
mutated `segment`.
"""
function remove_duplicates!(segment)
    to_delete = Int[]
    for i ∈ eachindex(segment)
        i == 1 && continue
        if isapprox(segment[i], segment[i-1]; rtol = 1e-10)
            push!(to_delete, i)
        end
    end

    return deleteat!(segment, to_delete)
end

"""
    follow_contour!(cells, border_cells, contour, x, y, A, is, js,
                    start_index, start_edge, level[; mask = []])

March through `cells` and `border_cells` containing crossings at `level`, appending each
crossing to `contour` (mutated in place) and removing traversed cells.

`start_edge` and `start_index` specify the cell in `cells` and `border_cells` at which to
start the march. If `mask` is specified, constrain the contour to the convex polygon
defined by the points in `mask`. Returns the index of the last cell visited.
"""
function follow_contour!(cells, border_cells, contour, x, y, A, is, js, start_index, start_edge, level; mask = [])
    index = start_index
    edge = start_edge

    push!(contour, get_crossing(x, y, A, index, edge, level))
    while true

        if index ∈ keys(cells)
            edge = pop!(cells, index) ⊻ edge
            push!(contour, get_crossing(x, y, A, index, edge, level))
            edge, index = get_next_cell(edge, index)

            (!(index[1] ∈ is) || !(index[2] ∈ js) || index == start_index) && break
            continue
        end

        if index ∈ keys(border_cells)
            while true
                edge = pop!(border_cells, index) ⊻ edge
                p = get_crossing(x, y, A, index, edge, level)

                intersection_found = false
                for i in eachindex(mask)
                    ip = mod(i, length(mask)) + 1
                    q, t = param_intersection(mask[i], mask[ip] - mask[i], contour[end], p - contour[end])
                    if 0 <= t[2] <= 1
                        push!(contour, q)
                        intersection_found = !intersection_found
                        break
                    end
                end
                intersection_found && break

                push!(contour, p)
                edge, index = get_next_cell(edge, index)

                !(index ∈ keys(border_cells)) && break
            end

            break
        end

        break # next cell has already been removed or was never a crossing cell
    end

    return index
end

"""
    get_crossing(x, y, A, index, edge, level)

Linearly interpolate values in cell at `index` of `A` along `edge` to obtain crossing at value `level` to find crossing in terms of coordinate grids `x` and `y`.
"""
function get_crossing(x, y, A, index, edge, level)
    i, j = index
    if edge == 0x08 # Left
        xcoord = x[i]
        ycoord = y[j] + (level - A[i,j]) * (y[j + 1] - y[j]) / (A[i, j + 1] - A[i,j])
    elseif edge == 0x01 # Top
        xcoord = x[i] + (level - A[i, j + 1]) * (x[i + 1] - x[i]) / (A[i + 1, j + 1] - A[i, j + 1])
        ycoord = y[j + 1]
    elseif edge == 0x04 # Right
        xcoord = x[i + 1]
        ycoord = y[j] + (level - A[i + 1, j]) * (y[j + 1] - y[j]) / (A[i + 1, j + 1] - A[i + 1, j])
    elseif edge == 0x02 # Bottom
        xcoord = x[i] + (level - A[i,j]) * (x[i + 1] - x[i]) / (A[i + 1, j] - A[i,j])
        ycoord = y[j]
    end

    return SVector(xcoord, ycoord)
end

"""
    contours(x, y, A, levels [; mask = []])

Generate contour bundles for each level in `levels` of the matrix `A` on coordinate grids `x` and `y`.

See also [`find_contour`](@ref).
"""
function contours(x, y, A, levels; mask = [])
    bundles = Vector{IsolineBundle}(undef, Base.length(levels))
    for i in eachindex(levels)
        bundles[i] = find_contour(x, y, A, levels[i]; mask)
    end

    return bundles
end

"""
    get_bounding_box(points)

Find bounding values for each coordinate of 2D `points`.

Returns `((x_min, x_max), (y_min, y_max))` over the supplied points.

# Examples
```jldoctest
julia> using StaticArrays

julia> get_bounding_box([SVector(-1.0, 0.0), SVector(2.0, 3.0), SVector(0.0, -2.0)])
((-1.0, 2.0), (-2.0, 3.0))
```
"""
function get_bounding_box(points)
    min_x = minimum(first.(points))
    max_x = maximum(first.(points))

    min_y = minimum(last.(points))
    max_y = maximum(last.(points))

    return ((min_x, max_x), (min_y, max_y))
end

"""
    contour_intersection(a, v, iso::Isoline[, i = 1])

Find intersection of contour `iso` with the line x = `a` + t `v` for real parameter t.

All sign changes of the cross product (iso.points[i] - a) × v are located; the crossing
with minimum |t| is returned. For closed isolines there are typically two crossings; for
open arcs there may also be two crossings when the arc is U-shaped and the gradient line
cuts both arms. In all cases only the nearest crossing is returned.

If `i` is specified, the search starts at point `i` of `iso.points`.
"""
function contour_intersection(a, v, iso::Isoline, i = 1)
    imax    = lastindex(iso.points)
    oldsign = 0
    area    = 0.0
    best_p  = iso.points[begin]
    best_t  = Inf
    best_ij = (firstindex(iso.points), firstindex(iso.points))

    while i <= imax
        w       = iso.points[i] - a
        area    = w[1] * v[2] - w[2] * v[1]
        newsign = sign(area)
        if oldsign != 0 && newsign * oldsign < 0
            i_cross = i - 1
            p = intersection(a, v, iso.points[i_cross], iso.points[i] - iso.points[i_cross])
            t = dot(p - a, v) / dot(v, v)
            if abs(t) < abs(best_t)
                best_t  = t
                best_p  = p
                best_ij = (i_cross, i)
            end
        end
        oldsign = newsign
        i += 1
    end

    if isinf(best_t) # no crossing found — fall back to nearest endpoint
        area_end = let w = iso.points[imax] - a; w[1] * v[2] - w[2] * v[1] end
        area_beg = let w = iso.points[begin] - a; w[1] * v[2] - w[2] * v[1] end
        θmax = atan(abs(area_end), abs(dot(v, iso.points[imax] - a)))
        θ1   = atan(abs(area_beg), abs(dot(v, iso.points[begin] - a)))
        if abs(θmax) < abs(θ1)
            return iso.points[imax], (imax, imax)
        else
            return iso.points[begin], (firstindex(iso.points), firstindex(iso.points))
        end
    end

    return best_p, best_ij
end

"""
    arc_points(iso::Isoline, ia::Int, ib::Int)

Return the points of `iso` between indices `ia` and `ib`, taking the shorter arc.

For open isolines this is a plain slice. For closed isolines, if the direct slice
`min(ia,ib):max(ia,ib)` covers more than half the loop, the wrap-around arc (from
the larger index to the end and from the beginning to the smaller index) is returned
instead.
"""
function arc_points(iso::Isoline, ia::Int, ib::Int)
    a, b = min(ia, ib), max(ia, ib)
    pts  = iso.points
    imax = lastindex(pts)
    if !iso.isclosed || b - a <= imax ÷ 2
        return pts[a:b]
    else
        return vcat(pts[b:end], pts[begin:a])
    end
end

###
### Curve geometry
###

"""
    centroid(iso::Isoline)

Return the centroid of the points of `iso`.

# Examples
```jldoctest
julia> using StaticArrays

julia> iso = Isoline([SVector(0.0, 0.0), SVector(2.0, 0.0), SVector(1.0, 3.0)], false, 0.0);

julia> Ludwig.centroid(iso)
2-element SVector{2, Float64} with indices SOneTo(2):
 1.0
 1.0
```
"""
centroid(iso::Isoline) = sum(iso.points) / length(iso.points)

"""
    get_arclengths(curve)

Get the distance to each point along `curve`.

Treats `curve` as an ordered list of points defining a piecewise linear path.

# Examples
```jldoctest
julia> using StaticArrays

julia> Ludwig.get_arclengths([SVector(0.0, 0.0), SVector(3.0, 0.0), SVector(3.0, 4.0)])
3-element Vector{Float64}:
 0.0
 3.0
 7.0
```
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
    aligned_arclength_slice(iso, ref_iso, n; cusp_fractions, angle_threshold)

Resample `iso` at `n` arclength-uniform points, oriented to match `ref_iso`.

The traversal direction of `iso` is chosen so that its starting tangent best agrees
with the starting tangent of `ref_iso`. Both the forward tangent at `iso.points[begin]`
and the reversed tangent at `iso.points[end]` are compared against the tangent at
`ref_iso.points[begin]`; the orientation with the larger dot product is used.

# Arguments
- `cusp_fractions`: arclength fractions in [0, 1] (relative to the *original* traversal
  direction of `iso`) at which cuts should be snapped. Fractions are mirrored when the
  isoline is reversed. Any cusps intrinsic to this contour are detected and merged
  automatically.
- `angle_threshold`: turning-angle threshold (radians) passed to [`find_cusp_fraction`](@ref).
"""
function aligned_arclength_slice(iso::Isoline, ref_iso::Isoline, n::Int;
                                  cusp_fractions = Float64[], angle_threshold = π/8)
    pts = iso.points
    ref_pts = ref_iso.points

    fractions = copy(cusp_fractions)

    if length(pts) > 1
        ref_tangent = ref_pts[2] - ref_pts[1]
        t_fwd = pts[2] - pts[1]
        t_rev = pts[end-1] - pts[end]
        reverse_iso = dot(t_rev, ref_tangent) > dot(t_fwd, ref_tangent)

        if reverse_iso
            pts = reverse(pts)
            fractions = 1.0 .- cusp_fractions
        end
    end
    append!(fractions, find_cusp_fraction(pts; angle_threshold))
    L = get_arclengths(pts)[end]
    return arclength_slice(pts, n; pinned_arclengths = fractions .* L)[1]
end

"""
    match_contour_segments(b1, b2)

Return the permutation of `b2` isolines that best matches the isolines of `b1` by centroid
proximity. Returns `nothing` if the two collections have different lengths.
"""
function match_contour_segments(b1::Vector{Isoline}, b2::Vector{Isoline})
    length(b1) != length(b2) && return nothing
    c1 = centroid.(b1)
    c2 = centroid.(b2)

    perm = Int[] # Permutation of b2 isolines that matches with b1 isolines
    for i ∈ eachindex(c2)
        order = sortperm(norm.(Ref(c2[i]) .- c1)) # Order centroid c2[i] by distance to all centroids, c1
        push!(perm, order[findfirst(x -> !(x ∈ perm), order)])
    end

    return perm
end

match_contour_segments(b1::IsolineBundle, b2::IsolineBundle) = match_contour_segments(b1.isolines, b2.isolines)
