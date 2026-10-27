###
### Marching Simplices
###

"""
    signchange(value_a, value_b, value_c)

Return whether the three shifted corner values straddle the level (some negative, some nonnegative).
"""
@inline signchange(value_a, value_b, value_c) =
    (min(value_a, value_b, value_c) < 0.0) & (max(value_a, value_b, value_c) ≥ 0.0)



"""
    clip_segment_to_polytope(endpoint_1, endpoint_2, polytope)

Clip the segment `endpoint_1`–`endpoint_2` to the convex `polytope` (Liang–Barsky). Return
`(clipped_1, clipped_2, keep)`: the clipped endpoints and whether any part of the segment lies inside.
"""
@inline function clip_segment_to_polytope(endpoint_1::SVector{2,Float64}, endpoint_2::SVector{2,Float64},
                                          polytope::ConvexPolytope{2})
    direction = endpoint_2 - endpoint_1
    t_enter = 0.0
    t_exit = 1.0
    @inbounds for face in polytope.faces
        # inside ⇔ normal·x ≤ offset
        value = dot(face.normal, endpoint_1) - face.offset
        rate = dot(face.normal, direction)
        if rate == 0.0
            value > 0.0 && return (endpoint_1, endpoint_2, false)   # parallel to face and outside
        elseif rate > 0.0
            t = -value / rate                        # leaving the half-space
            t < t_exit && (t_exit = t)
        else
            t = -value / rate                        # entering the half-space
            t > t_enter && (t_enter = t)
        end
        t_enter > t_exit && return (endpoint_1, endpoint_2, false)
    end
    return (endpoint_1 + t_enter * direction, endpoint_1 + t_exit * direction, true)
end

# Clip the convex polygon `input` (vertex list) by one half-space into `output` (Sutherland–Hodgman).
function clip_against_halfspace!(output::Vector{SVector{3,Float64}},
                                 input::Vector{SVector{3,Float64}}, face::HalfSpace{3})
    empty!(output)
    vertex_count = length(input)
    vertex_count == 0 && return
    @inbounds for i in 1:vertex_count
        current = input[i]
        next_vertex = input[i == vertex_count ? 1 : i + 1]
        value_current = dot(face.normal, current) - face.offset
        value_next = dot(face.normal, next_vertex) - face.offset
        current_inside = value_current ≤ 0.0
        next_inside = value_next ≤ 0.0
        if current_inside
            push!(output, current)
            if !next_inside
                crossing_fraction = value_current / (value_current - value_next)
                push!(output, current + crossing_fraction * (next_vertex - current))
            end
        elseif next_inside
            crossing_fraction = value_current / (value_current - value_next)
            push!(output, current + crossing_fraction * (next_vertex - current))
        end
    end
    return
end

"""
    clip_triangle_to_polytope!(vertices, simplices, corner_1, corner_2, corner_3, positive_corner, polytope, buffer_A, buffer_B)

Clip the surface triangle to the convex `polytope` and append the (fan-triangulated) result to
`vertices`/`simplices`, wound toward `positive_corner`. `buffer_A`/`buffer_B` are reusable scratch.
"""
function clip_triangle_to_polytope!(vertices, simplices,
                                    corner_1::SVector{3,Float64}, corner_2::SVector{3,Float64},
                                    corner_3::SVector{3,Float64}, positive_corner::SVector{3,Float64},
                                    polytope::ConvexPolytope{3},
                                    buffer_A::Vector{SVector{3,Float64}},
                                    buffer_B::Vector{SVector{3,Float64}})
    empty!(buffer_A)
    push!(buffer_A, corner_1); push!(buffer_A, corner_2); push!(buffer_A, corner_3)
    source = buffer_A
    target = buffer_B
    for face in polytope.faces
        clip_against_halfspace!(target, source, face)
        isempty(target) && return nothing            # fully clipped away
        source, target = target, source
    end
    length(source) < 3 && return nothing             # degenerate sliver
    @inbounds for i in 2:length(source)-1            # fan-triangulate around source[1]
        push_triangle!(vertices, simplices, source[1], source[i], source[i + 1], positive_corner)
    end
    return nothing
end

###
### Leaf marching
###

"""
    edge_crossing(point_a, value_a, point_b, value_b)

Return the point on edge `point_a`–`point_b` where the shifted value linearly crosses zero.
"""
@inline edge_crossing(point_a, value_a, point_b, value_b) =
    point_a + (value_a / (value_a - value_b)) * (point_b - point_a)

"""
    emit_segment!(verts, simps, point_a, value_a, point_b, value_b, point_c, value_c)

Append the contour segment crossing a leaf triangle to `verts` and `simps`, if one exists.
"""
function emit_segment!(verts, simps,
                       point_a::SVector{2,Float64}, value_a::Float64,
                       point_b::SVector{2,Float64}, value_b::Float64,
                       point_c::SVector{2,Float64}, value_c::Float64)
    # Exactly 0 counts as nonnegative so the contour snaps to that vertex rather than
    # emitting a degenerate zero-length segment.
    a_nonnegative = value_a ≥ 0.0
    b_nonnegative = value_b ≥ 0.0
    c_nonnegative = value_c ≥ 0.0
    (a_nonnegative == b_nonnegative == c_nonnegative) && return nothing   # all on one side: no crossing

    # Exactly two of the three edges flip sign; collect their crossings.
    crossing_1 = point_a
    crossing_2 = point_a
    crossings_found = 0
    if a_nonnegative != b_nonnegative
        crossing = edge_crossing(point_a, value_a, point_b, value_b)
        crossings_found == 0 ? (crossing_1 = crossing) : (crossing_2 = crossing); crossings_found += 1
    end
    if b_nonnegative != c_nonnegative
        crossing = edge_crossing(point_b, value_b, point_c, value_c)
        crossings_found == 0 ? (crossing_1 = crossing) : (crossing_2 = crossing); crossings_found += 1
    end
    if c_nonnegative != a_nonnegative
        crossing = edge_crossing(point_c, value_c, point_a, value_a)
        crossings_found == 0 ? (crossing_1 = crossing) : (crossing_2 = crossing); crossings_found += 1
    end

    if crossings_found == 2
        push!(verts, crossing_1)
        push!(verts, crossing_2)
        vertex_count = length(verts)
        push!(simps, (vertex_count - 1, vertex_count))
    end
    return nothing
end

"""
    emit_segment!(verts, simps, point_a, value_a, point_b, value_b, point_c, value_c, polytope)

Append the contour segment crossing a leaf triangle, clipped to the convex `polytope`, if one remains.
"""
function emit_segment!(verts, simps,
                       point_a::SVector{2,Float64}, value_a::Float64,
                       point_b::SVector{2,Float64}, value_b::Float64,
                       point_c::SVector{2,Float64}, value_c::Float64,
                       polytope::ConvexPolytope{2})
    a_nonnegative = value_a ≥ 0.0
    b_nonnegative = value_b ≥ 0.0
    c_nonnegative = value_c ≥ 0.0
    (a_nonnegative == b_nonnegative == c_nonnegative) && return nothing   # all on one side: no crossing

    crossing_1 = point_a
    crossing_2 = point_a
    crossings_found = 0
    if a_nonnegative != b_nonnegative
        crossing = edge_crossing(point_a, value_a, point_b, value_b)
        crossings_found == 0 ? (crossing_1 = crossing) : (crossing_2 = crossing); crossings_found += 1
    end
    if b_nonnegative != c_nonnegative
        crossing = edge_crossing(point_b, value_b, point_c, value_c)
        crossings_found == 0 ? (crossing_1 = crossing) : (crossing_2 = crossing); crossings_found += 1
    end
    if c_nonnegative != a_nonnegative
        crossing = edge_crossing(point_c, value_c, point_a, value_a)
        crossings_found == 0 ? (crossing_1 = crossing) : (crossing_2 = crossing); crossings_found += 1
    end

    if crossings_found == 2
        clipped_1, clipped_2, keep = clip_segment_to_polytope(crossing_1, crossing_2, polytope)
        if keep
            push!(verts, clipped_1)
            push!(verts, clipped_2)
            vertex_count = length(verts)
            push!(simps, (vertex_count - 1, vertex_count))
        end
    end
    return nothing
end

###
### Adaptive refinement
###

"""
    refine_triangle!(verts, simps, f, level, point_a, value_a, point_b, value_b, point_c, value_c, depth, maxdepth, polytope)

Quadrisect a straddling triangle (resampling `f` at its edge midpoints) until the size floor, then march it.
A convex `polytope` (or `nothing`) restricts the contour: triangles fully outside it are culled and the
contour is snapped to its boundary.
"""
function refine_triangle!(verts, simps, f, level,
                          point_a::SVector{2,Float64}, value_a::Float64,
                          point_b::SVector{2,Float64}, value_b::Float64,
                          point_c::SVector{2,Float64}, value_c::Float64,
                          depth::Int, maxdepth::Int, polytope::Nothing)
    if depth < maxdepth && signchange(value_a, value_b, value_c)
        point_ab = 0.5 * (point_a + point_b)
        point_bc = 0.5 * (point_b + point_c)
        point_ca = 0.5 * (point_c + point_a)
        value_ab = f(point_ab) - level
        value_bc = f(point_bc) - level
        value_ca = f(point_ca) - level
        child_depth = depth + 1
        refine_triangle!(verts, simps, f, level, point_a,  value_a,  point_ab, value_ab, point_ca, value_ca, child_depth, maxdepth, polytope)
        refine_triangle!(verts, simps, f, level, point_ab, value_ab, point_b,  value_b,  point_bc, value_bc, child_depth, maxdepth, polytope)
        refine_triangle!(verts, simps, f, level, point_ca, value_ca, point_bc, value_bc, point_c,  value_c,  child_depth, maxdepth, polytope)
        refine_triangle!(verts, simps, f, level, point_ab, value_ab, point_bc, value_bc, point_ca, value_ca, child_depth, maxdepth, polytope)
    else
        emit_segment!(verts, simps, point_a, value_a, point_b, value_b, point_c, value_c)
    end
    return nothing
end

function refine_triangle!(verts, simps, f, level,
                          point_a::SVector{2,Float64}, value_a::Float64,
                          point_b::SVector{2,Float64}, value_b::Float64,
                          point_c::SVector{2,Float64}, value_c::Float64,
                          depth::Int, maxdepth::Int, polytope::ConvexPolytope{2})
    number_inside = in_convex_polytope(point_a, polytope) +
                    in_convex_polytope(point_b, polytope) +
                    in_convex_polytope(point_c, polytope)
    number_inside == 0 && return nothing   # fully outside: cull
    if number_inside == 3                  # fully interior: descendants need no polytope checks
        return refine_triangle!(verts, simps, f, level, point_a, value_a, point_b, value_b, point_c, value_c,
                                depth, maxdepth, nothing)
    end
    # straddling the boundary: keep probing children, snap leaf contour to the polytope
    if depth < maxdepth && signchange(value_a, value_b, value_c)
        point_ab = 0.5 * (point_a + point_b)
        point_bc = 0.5 * (point_b + point_c)
        point_ca = 0.5 * (point_c + point_a)
        value_ab = f(point_ab) - level
        value_bc = f(point_bc) - level
        value_ca = f(point_ca) - level
        child_depth = depth + 1
        refine_triangle!(verts, simps, f, level, point_a,  value_a,  point_ab, value_ab, point_ca, value_ca, child_depth, maxdepth, polytope)
        refine_triangle!(verts, simps, f, level, point_ab, value_ab, point_b,  value_b,  point_bc, value_bc, child_depth, maxdepth, polytope)
        refine_triangle!(verts, simps, f, level, point_ca, value_ca, point_bc, value_bc, point_c,  value_c,  child_depth, maxdepth, polytope)
        refine_triangle!(verts, simps, f, level, point_ab, value_ab, point_bc, value_bc, point_ca, value_ca, child_depth, maxdepth, polytope)
    else
        emit_segment!(verts, simps, point_a, value_a, point_b, value_b, point_c, value_c, polytope)
    end
    return nothing
end

"""
    seed_grid!(verts, simps, f, level, x0, y0, dx, dy, N, maxdepth, samples, polytope)

Sample `f` on the `N×N` grid into `samples` and adaptively refine the two triangles of each cell, optionally
restricting the contour to the convex `polytope`.
"""
function seed_grid!(verts, simps, f, level, x0, y0, dx, dy, N, maxdepth, samples, polytope)
    @inbounds for j in 1:N, i in 1:N
        point = SVector(x0 + (i - 1) * dx, y0 + (j - 1) * dy)
        samples[i, j] = f(point) - level
    end
    @inbounds for j in 1:N-1, i in 1:N-1
        x_lower = x0 + (i - 1) * dx
        x_upper = x0 + i * dx
        y_lower = y0 + (j - 1) * dy
        y_upper = y0 + j * dy
        point_00 = SVector(x_lower, y_lower)
        point_10 = SVector(x_upper, y_lower)
        point_01 = SVector(x_lower, y_upper)
        point_11 = SVector(x_upper, y_upper)
        value_00 = samples[i, j]
        value_10 = samples[i+1, j]
        value_01 = samples[i, j+1]
        value_11 = samples[i+1, j+1]
        refine_triangle!(verts, simps, f, level, point_00, value_00, point_10, value_10, point_11, value_11, 0, maxdepth, polytope)
        refine_triangle!(verts, simps, f, level, point_00, value_00, point_11, value_11, point_01, value_01, 0, maxdepth, polytope)
    end
    return nothing
end

"""
    level_set(f, level, xlims, ylims, initial_depth; refinement_depth=initial_depth, samples=Matrix{Float64}(undef,N,N), weld=false, mask=nothing)

Return the `level` contour of `f` over the box `xlims × ylims` as a `SimplexMesh{2}`. The coarse seed grid
has `N = 2^initial_depth + 1` points per axis (i.e. `initial_depth` subdivisions of the box); each
straddling cell is then adaptively quadrisected until the total subdivision depth from the bounding box
reaches `refinement_depth`, so the finest cell spans `extent / 2^refinement_depth` regardless of
`initial_depth`. `refinement_depth` must be `≥ initial_depth`.

`mask`, a `ConvexPolytope{2}`, restricts the contour to that polygon: the contour outside it is dropped
and the contour reaching it is snapped to its boundary.
"""
function level_set(f, level::Real,
                   xlims::Tuple{<:Real,<:Real}, ylims::Tuple{<:Real,<:Real}, initial_depth::Integer;
                   refinement_depth::Integer=initial_depth,
                   samples::AbstractMatrix{Float64}=Matrix{Float64}(undef, 2^max(initial_depth, 0) + 1, 2^max(initial_depth, 0) + 1),
                   weld::Bool=false,
                   mask::Union{Nothing,ConvexPolytope}=nothing)
    initial_depth ≥ 0 || throw(ArgumentError("initial_depth must be nonnegative, got initial_depth = $initial_depth"))
    refinement_depth ≥ initial_depth || throw(ArgumentError("refinement_depth (=$refinement_depth) must be ≥ initial_depth (=$initial_depth)"))
    N = 2^initial_depth + 1
    maxdepth = refinement_depth - initial_depth
    level_value = Float64(level)
    x0, x1 = Float64(xlims[1]), Float64(xlims[2])
    y0, y1 = Float64(ylims[1]), Float64(ylims[2])
    dx = (x1 - x0) / (N - 1)
    dy = (y1 - y0) / (N - 1)

    polytope = mask

    verts = SVector{2,Float64}[]
    simps = NTuple{2,Int}[]
    sizehint!(verts, 4N)
    sizehint!(simps, 2N)

    seed_grid!(verts, simps, f, level_value, x0, y0, dx, dy, N, maxdepth, samples, polytope)

    mesh = SimplexMesh{2}(verts, simps)
    if weld
        # The bin width tracks the finest cell extent — extent / 2^refinement_depth, independent of
        # initial_depth — so distinct contour vertices stay in distinct bins at any refinement depth;
        # the floor keeps the tolerance above floating-point noise in vertex coordinates (~eps · extent).
        cell_fine = min(x1 - x0, y1 - y0) / 2.0^refinement_depth
        atol = max(1e-9 * max(x1 - x0, y1 - y0), 1e-3 * cell_fine)
        return merge_vertices(mesh; atol)
    end
    return mesh
end

###
### Marching Tetrahedra (3D)
###

@inline signchange(value_1, value_2, value_3, value_4) =
    (min(value_1, value_2, value_3, value_4) < 0.0) & (max(value_1, value_2, value_3, value_4) ≥ 0.0)

"""
    push_triangle!(verts, simps, v0, v1, v2, positive_corner)

Append a surface triangle to `verts`/`simps`, wound so its normal points toward `positive_corner`.
"""
@inline function push_triangle!(verts, simps,
                                v0::SVector{3,Float64}, v1::SVector{3,Float64}, v2::SVector{3,Float64},
                                positive_corner::SVector{3,Float64})
    normal = cross(v1 - v0, v2 - v0)
    if dot(normal, positive_corner - v0) < 0.0
        v1, v2 = v2, v1
    end
    push!(verts, v0); push!(verts, v1); push!(verts, v2)
    vertex_count = length(verts)
    push!(simps, (vertex_count - 2, vertex_count - 1, vertex_count))
    return nothing
end

"""
    emit_triangles!(verts, simps, point_1, value_1, point_2, value_2, point_3, value_3, point_4, value_4)

Append the iso-surface (one or two triangles) crossing a leaf tetrahedron to `verts` and `simps`, if any.
"""
function emit_triangles!(verts, simps,
                         point_1::SVector{3,Float64}, value_1::Float64,
                         point_2::SVector{3,Float64}, value_2::Float64,
                         point_3::SVector{3,Float64}, value_3::Float64,
                         point_4::SVector{3,Float64}, value_4::Float64)
    corner_points = (point_1, point_2, point_3, point_4)
    corner_values = (value_1, value_2, value_3, value_4)
    corner_nonnegative = (value_1 ≥ 0.0, value_2 ≥ 0.0, value_3 ≥ 0.0, value_4 ≥ 0.0)
    negative_count = (!corner_nonnegative[1]) + (!corner_nonnegative[2]) +
                     (!corner_nonnegative[3]) + (!corner_nonnegative[4])
    (negative_count == 0 || negative_count == 4) && return nothing   # all on one side: no surface

    if negative_count == 1 || negative_count == 3
        # 1-vs-3: one lone corner, surface is one triangle on the 3 edges incident to it.
        lone_is_positive = negative_count == 3
        lone = corner_nonnegative[1] == lone_is_positive ? 1 :
               corner_nonnegative[2] == lone_is_positive ? 2 :
               corner_nonnegative[3] == lone_is_positive ? 3 : 4
        other_1 = 0; other_2 = 0; other_3 = 0
        for i in 1:4
            i == lone && continue
            other_1 == 0 ? (other_1 = i) : other_2 == 0 ? (other_2 = i) : (other_3 = i)
        end
        crossing_1 = edge_crossing(corner_points[lone], corner_values[lone],
                                   corner_points[other_1], corner_values[other_1])
        crossing_2 = edge_crossing(corner_points[lone], corner_values[lone],
                                   corner_points[other_2], corner_values[other_2])
        crossing_3 = edge_crossing(corner_points[lone], corner_values[lone],
                                   corner_points[other_3], corner_values[other_3])
        positive_corner = lone_is_positive ? corner_points[lone] : corner_points[other_1]
        push_triangle!(verts, simps, crossing_1, crossing_2, crossing_3, positive_corner)
    else
        # 2-vs-2: quad cut on the 4 edges joining a negative corner to a positive one → 2 triangles.
        negative_1 = 0; negative_2 = 0; positive_1 = 0; positive_2 = 0
        for i in 1:4
            if corner_nonnegative[i]
                positive_1 == 0 ? (positive_1 = i) : (positive_2 = i)
            else
                negative_1 == 0 ? (negative_1 = i) : (negative_2 = i)
            end
        end
        crossing_neg1_pos1 = edge_crossing(corner_points[negative_1], corner_values[negative_1],
                                           corner_points[positive_1], corner_values[positive_1])
        crossing_neg1_pos2 = edge_crossing(corner_points[negative_1], corner_values[negative_1],
                                           corner_points[positive_2], corner_values[positive_2])
        crossing_neg2_pos2 = edge_crossing(corner_points[negative_2], corner_values[negative_2],
                                           corner_points[positive_2], corner_values[positive_2])
        crossing_neg2_pos1 = edge_crossing(corner_points[negative_2], corner_values[negative_2],
                                           corner_points[positive_1], corner_values[positive_1])
        # cyclic quad order neg1_pos1 → neg1_pos2 → neg2_pos2 → neg2_pos1 (consecutive points share a tet corner)
        push_triangle!(verts, simps, crossing_neg1_pos1, crossing_neg1_pos2, crossing_neg2_pos2, corner_points[positive_1])
        push_triangle!(verts, simps, crossing_neg1_pos1, crossing_neg2_pos2, crossing_neg2_pos1, corner_points[positive_1])
    end
    return nothing
end

"""
    emit_triangles!(verts, simps, point_1, value_1, point_2, value_2, point_3, value_3, point_4, value_4, polytope, buffer_A, buffer_B)

Append the iso-surface crossing a leaf tetrahedron, clipped (snapped) to the convex `polytope`, if any.
`buffer_A`/`buffer_B` are reusable scratch for the clip.
"""
function emit_triangles!(verts, simps,
                         point_1::SVector{3,Float64}, value_1::Float64,
                         point_2::SVector{3,Float64}, value_2::Float64,
                         point_3::SVector{3,Float64}, value_3::Float64,
                         point_4::SVector{3,Float64}, value_4::Float64,
                         polytope::ConvexPolytope{3},
                         buffer_A::Vector{SVector{3,Float64}},
                         buffer_B::Vector{SVector{3,Float64}})
    corner_points = (point_1, point_2, point_3, point_4)
    corner_values = (value_1, value_2, value_3, value_4)
    corner_nonnegative = (value_1 ≥ 0.0, value_2 ≥ 0.0, value_3 ≥ 0.0, value_4 ≥ 0.0)
    negative_count = (!corner_nonnegative[1]) + (!corner_nonnegative[2]) +
                     (!corner_nonnegative[3]) + (!corner_nonnegative[4])
    (negative_count == 0 || negative_count == 4) && return nothing   # all on one side: no surface

    if negative_count == 1 || negative_count == 3
        # 1-vs-3: one lone corner, surface is one triangle on the 3 edges incident to it.
        lone_is_positive = negative_count == 3
        lone = corner_nonnegative[1] == lone_is_positive ? 1 :
               corner_nonnegative[2] == lone_is_positive ? 2 :
               corner_nonnegative[3] == lone_is_positive ? 3 : 4
        other_1 = 0; other_2 = 0; other_3 = 0
        for i in 1:4
            i == lone && continue
            other_1 == 0 ? (other_1 = i) : other_2 == 0 ? (other_2 = i) : (other_3 = i)
        end
        crossing_1 = edge_crossing(corner_points[lone], corner_values[lone],
                                   corner_points[other_1], corner_values[other_1])
        crossing_2 = edge_crossing(corner_points[lone], corner_values[lone],
                                   corner_points[other_2], corner_values[other_2])
        crossing_3 = edge_crossing(corner_points[lone], corner_values[lone],
                                   corner_points[other_3], corner_values[other_3])
        positive_corner = lone_is_positive ? corner_points[lone] : corner_points[other_1]
        clip_triangle_to_polytope!(verts, simps, crossing_1, crossing_2, crossing_3, positive_corner,
                                   polytope, buffer_A, buffer_B)
    else
        # 2-vs-2: quad cut on the 4 edges joining a negative corner to a positive one → 2 triangles.
        negative_1 = 0; negative_2 = 0; positive_1 = 0; positive_2 = 0
        for i in 1:4
            if corner_nonnegative[i]
                positive_1 == 0 ? (positive_1 = i) : (positive_2 = i)
            else
                negative_1 == 0 ? (negative_1 = i) : (negative_2 = i)
            end
        end
        crossing_neg1_pos1 = edge_crossing(corner_points[negative_1], corner_values[negative_1],
                                           corner_points[positive_1], corner_values[positive_1])
        crossing_neg1_pos2 = edge_crossing(corner_points[negative_1], corner_values[negative_1],
                                           corner_points[positive_2], corner_values[positive_2])
        crossing_neg2_pos2 = edge_crossing(corner_points[negative_2], corner_values[negative_2],
                                           corner_points[positive_2], corner_values[positive_2])
        crossing_neg2_pos1 = edge_crossing(corner_points[negative_2], corner_values[negative_2],
                                           corner_points[positive_1], corner_values[positive_1])
        # cyclic quad order neg1_pos1 → neg1_pos2 → neg2_pos2 → neg2_pos1 (consecutive points share a tet corner)
        clip_triangle_to_polytope!(verts, simps, crossing_neg1_pos1, crossing_neg1_pos2, crossing_neg2_pos2,
                                   corner_points[positive_1], polytope, buffer_A, buffer_B)
        clip_triangle_to_polytope!(verts, simps, crossing_neg1_pos1, crossing_neg2_pos2, crossing_neg2_pos1,
                                   corner_points[positive_1], polytope, buffer_A, buffer_B)
    end
    return nothing
end

"""
    refine_tetrahedron!(verts, simps, f, level, point_1, value_1, point_2, value_2, point_3, value_3, point_4, value_4, depth, maxdepth, polytope, [buffer_A, buffer_B])

Regularly refine a straddling tetrahedron into 8 sub-tets (resampling `f` at its 6 edge midpoints) until the
size floor, then march it. A convex `polytope` (or `nothing`) restricts the iso-surface: tets fully outside
it are culled and the surface is snapped to its boundary; `buffer_A`/`buffer_B` are scratch for the clip.
"""
function refine_tetrahedron!(verts, simps, f, level,
                             point_1::SVector{3,Float64}, value_1::Float64,
                             point_2::SVector{3,Float64}, value_2::Float64,
                             point_3::SVector{3,Float64}, value_3::Float64,
                             point_4::SVector{3,Float64}, value_4::Float64,
                             depth::Int, maxdepth::Int, polytope::Nothing)
    if depth < maxdepth && signchange(value_1, value_2, value_3, value_4)
        midpoint_12 = 0.5*(point_1+point_2); midpoint_13 = 0.5*(point_1+point_3); midpoint_14 = 0.5*(point_1+point_4)
        midpoint_23 = 0.5*(point_2+point_3); midpoint_24 = 0.5*(point_2+point_4); midpoint_34 = 0.5*(point_3+point_4)
        value_12 = f(midpoint_12)-level; value_13 = f(midpoint_13)-level; value_14 = f(midpoint_14)-level
        value_23 = f(midpoint_23)-level; value_24 = f(midpoint_24)-level; value_34 = f(midpoint_34)-level
        child_depth = depth + 1
        # 4 corner tets
        refine_tetrahedron!(verts,simps,f,level, point_1,value_1, midpoint_12,value_12, midpoint_13,value_13, midpoint_14,value_14, child_depth, maxdepth, polytope)
        refine_tetrahedron!(verts,simps,f,level, point_2,value_2, midpoint_12,value_12, midpoint_23,value_23, midpoint_24,value_24, child_depth, maxdepth, polytope)
        refine_tetrahedron!(verts,simps,f,level, point_3,value_3, midpoint_13,value_13, midpoint_23,value_23, midpoint_34,value_34, child_depth, maxdepth, polytope)
        refine_tetrahedron!(verts,simps,f,level, point_4,value_4, midpoint_14,value_14, midpoint_24,value_24, midpoint_34,value_34, child_depth, maxdepth, polytope)
        # 4 interior tets, fanned around the SHORTEST octahedron diagonal (Bey: bounds slivers)
        squared_diagonal_12_34 = sum(abs2, midpoint_12 - midpoint_34)
        squared_diagonal_13_24 = sum(abs2, midpoint_13 - midpoint_24)
        squared_diagonal_14_23 = sum(abs2, midpoint_14 - midpoint_23)
        if squared_diagonal_12_34 ≤ squared_diagonal_13_24 && squared_diagonal_12_34 ≤ squared_diagonal_14_23
            refine_tetrahedron!(verts,simps,f,level, midpoint_12,value_12, midpoint_34,value_34, midpoint_13,value_13, midpoint_23,value_23, child_depth, maxdepth, polytope)
            refine_tetrahedron!(verts,simps,f,level, midpoint_12,value_12, midpoint_34,value_34, midpoint_23,value_23, midpoint_24,value_24, child_depth, maxdepth, polytope)
            refine_tetrahedron!(verts,simps,f,level, midpoint_12,value_12, midpoint_34,value_34, midpoint_24,value_24, midpoint_14,value_14, child_depth, maxdepth, polytope)
            refine_tetrahedron!(verts,simps,f,level, midpoint_12,value_12, midpoint_34,value_34, midpoint_14,value_14, midpoint_13,value_13, child_depth, maxdepth, polytope)
        elseif squared_diagonal_13_24 ≤ squared_diagonal_14_23
            refine_tetrahedron!(verts,simps,f,level, midpoint_13,value_13, midpoint_24,value_24, midpoint_12,value_12, midpoint_23,value_23, child_depth, maxdepth, polytope)
            refine_tetrahedron!(verts,simps,f,level, midpoint_13,value_13, midpoint_24,value_24, midpoint_23,value_23, midpoint_34,value_34, child_depth, maxdepth, polytope)
            refine_tetrahedron!(verts,simps,f,level, midpoint_13,value_13, midpoint_24,value_24, midpoint_34,value_34, midpoint_14,value_14, child_depth, maxdepth, polytope)
            refine_tetrahedron!(verts,simps,f,level, midpoint_13,value_13, midpoint_24,value_24, midpoint_14,value_14, midpoint_12,value_12, child_depth, maxdepth, polytope)
        else
            refine_tetrahedron!(verts,simps,f,level, midpoint_14,value_14, midpoint_23,value_23, midpoint_12,value_12, midpoint_24,value_24, child_depth, maxdepth, polytope)
            refine_tetrahedron!(verts,simps,f,level, midpoint_14,value_14, midpoint_23,value_23, midpoint_24,value_24, midpoint_34,value_34, child_depth, maxdepth, polytope)
            refine_tetrahedron!(verts,simps,f,level, midpoint_14,value_14, midpoint_23,value_23, midpoint_34,value_34, midpoint_13,value_13, child_depth, maxdepth, polytope)
            refine_tetrahedron!(verts,simps,f,level, midpoint_14,value_14, midpoint_23,value_23, midpoint_13,value_13, midpoint_12,value_12, child_depth, maxdepth, polytope)
        end
    else
        emit_triangles!(verts, simps, point_1,value_1, point_2,value_2, point_3,value_3, point_4,value_4)
    end
    return nothing
end

function refine_tetrahedron!(verts, simps, f, level,
                             point_1::SVector{3,Float64}, value_1::Float64,
                             point_2::SVector{3,Float64}, value_2::Float64,
                             point_3::SVector{3,Float64}, value_3::Float64,
                             point_4::SVector{3,Float64}, value_4::Float64,
                             depth::Int, maxdepth::Int, polytope::ConvexPolytope{3},
                             buffer_A::Vector{SVector{3,Float64}},
                             buffer_B::Vector{SVector{3,Float64}})
    number_inside = in_convex_polytope(point_1, polytope) + in_convex_polytope(point_2, polytope) +
                    in_convex_polytope(point_3, polytope) + in_convex_polytope(point_4, polytope)
    number_inside == 0 && return nothing   # fully outside: cull
    if number_inside == 4                  # fully interior: descendants need no polytope checks
        return refine_tetrahedron!(verts, simps, f, level, point_1,value_1, point_2,value_2, point_3,value_3, point_4,value_4, depth, maxdepth, nothing)
    end
    # straddling the boundary: keep probing children, snap leaf surface to the polytope
    if depth < maxdepth && signchange(value_1, value_2, value_3, value_4)
        midpoint_12 = 0.5*(point_1+point_2); midpoint_13 = 0.5*(point_1+point_3); midpoint_14 = 0.5*(point_1+point_4)
        midpoint_23 = 0.5*(point_2+point_3); midpoint_24 = 0.5*(point_2+point_4); midpoint_34 = 0.5*(point_3+point_4)
        value_12 = f(midpoint_12)-level; value_13 = f(midpoint_13)-level; value_14 = f(midpoint_14)-level
        value_23 = f(midpoint_23)-level; value_24 = f(midpoint_24)-level; value_34 = f(midpoint_34)-level
        child_depth = depth + 1
        # 4 corner tets
        refine_tetrahedron!(verts,simps,f,level, point_1,value_1, midpoint_12,value_12, midpoint_13,value_13, midpoint_14,value_14, child_depth, maxdepth, polytope, buffer_A, buffer_B)
        refine_tetrahedron!(verts,simps,f,level, point_2,value_2, midpoint_12,value_12, midpoint_23,value_23, midpoint_24,value_24, child_depth, maxdepth, polytope, buffer_A, buffer_B)
        refine_tetrahedron!(verts,simps,f,level, point_3,value_3, midpoint_13,value_13, midpoint_23,value_23, midpoint_34,value_34, child_depth, maxdepth, polytope, buffer_A, buffer_B)
        refine_tetrahedron!(verts,simps,f,level, point_4,value_4, midpoint_14,value_14, midpoint_24,value_24, midpoint_34,value_34, child_depth, maxdepth, polytope, buffer_A, buffer_B)
        # 4 interior tets, fanned around the SHORTEST octahedron diagonal (Bey: bounds slivers)
        squared_diagonal_12_34 = sum(abs2, midpoint_12 - midpoint_34)
        squared_diagonal_13_24 = sum(abs2, midpoint_13 - midpoint_24)
        squared_diagonal_14_23 = sum(abs2, midpoint_14 - midpoint_23)
        if squared_diagonal_12_34 ≤ squared_diagonal_13_24 && squared_diagonal_12_34 ≤ squared_diagonal_14_23
            refine_tetrahedron!(verts,simps,f,level, midpoint_12,value_12, midpoint_34,value_34, midpoint_13,value_13, midpoint_23,value_23, child_depth, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, midpoint_12,value_12, midpoint_34,value_34, midpoint_23,value_23, midpoint_24,value_24, child_depth, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, midpoint_12,value_12, midpoint_34,value_34, midpoint_24,value_24, midpoint_14,value_14, child_depth, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, midpoint_12,value_12, midpoint_34,value_34, midpoint_14,value_14, midpoint_13,value_13, child_depth, maxdepth, polytope, buffer_A, buffer_B)
        elseif squared_diagonal_13_24 ≤ squared_diagonal_14_23
            refine_tetrahedron!(verts,simps,f,level, midpoint_13,value_13, midpoint_24,value_24, midpoint_12,value_12, midpoint_23,value_23, child_depth, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, midpoint_13,value_13, midpoint_24,value_24, midpoint_23,value_23, midpoint_34,value_34, child_depth, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, midpoint_13,value_13, midpoint_24,value_24, midpoint_34,value_34, midpoint_14,value_14, child_depth, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, midpoint_13,value_13, midpoint_24,value_24, midpoint_14,value_14, midpoint_12,value_12, child_depth, maxdepth, polytope, buffer_A, buffer_B)
        else
            refine_tetrahedron!(verts,simps,f,level, midpoint_14,value_14, midpoint_23,value_23, midpoint_12,value_12, midpoint_24,value_24, child_depth, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, midpoint_14,value_14, midpoint_23,value_23, midpoint_24,value_24, midpoint_34,value_34, child_depth, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, midpoint_14,value_14, midpoint_23,value_23, midpoint_34,value_34, midpoint_13,value_13, child_depth, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, midpoint_14,value_14, midpoint_23,value_23, midpoint_13,value_13, midpoint_12,value_12, child_depth, maxdepth, polytope, buffer_A, buffer_B)
        end
    else
        emit_triangles!(verts, simps, point_1,value_1, point_2,value_2, point_3,value_3, point_4,value_4, polytope, buffer_A, buffer_B)
    end
    return nothing
end

"""
    seed_grid!(verts, simps, f, level, x0, y0, z0, dx, dy, dz, N, maxdepth, samples, polytope, buffer_A, buffer_B)

Sample `f` on the `N×N×N` grid into `samples` and adaptively refine the six Kuhn tetrahedra of each cell,
optionally restricting the iso-surface to the convex `polytope` (`buffer_A`/`buffer_B` are clip scratch).
"""
function seed_grid!(verts, simps, f, level, x0, y0, z0, dx, dy, dz, N, maxdepth, samples,
                    polytope, buffer_A, buffer_B)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        point = SVector(x0 + (i-1)*dx, y0 + (j-1)*dy, z0 + (k-1)*dz)
        samples[i, j, k] = f(point) - level
    end
    @inbounds for k in 1:N-1, j in 1:N-1, i in 1:N-1
        x_lower = x0 + (i-1)*dx; x_upper = x0 + i*dx
        y_lower = y0 + (j-1)*dy; y_upper = y0 + j*dy
        z_lower = z0 + (k-1)*dz; z_upper = z0 + k*dz
        corner_0 = SVector(x_lower,y_lower,z_lower); value_0 = samples[i,   j,   k]      # corner (0,0,0)
        corner_1 = SVector(x_upper,y_lower,z_lower); value_1 = samples[i+1, j,   k]      #        (1,0,0)
        corner_2 = SVector(x_lower,y_upper,z_lower); value_2 = samples[i,   j+1, k]      #        (0,1,0)
        corner_3 = SVector(x_upper,y_upper,z_lower); value_3 = samples[i+1, j+1, k]      #        (1,1,0)
        corner_4 = SVector(x_lower,y_lower,z_upper); value_4 = samples[i,   j,   k+1]    #        (0,0,1)
        corner_5 = SVector(x_upper,y_lower,z_upper); value_5 = samples[i+1, j,   k+1]    #        (1,0,1)
        corner_6 = SVector(x_lower,y_upper,z_upper); value_6 = samples[i,   j+1, k+1]    #        (0,1,1)
        corner_7 = SVector(x_upper,y_upper,z_upper); value_7 = samples[i+1, j+1, k+1]    #        (1,1,1)
        # 6 Kuhn tets sharing the main diagonal 0–7 (identical per cell ⇒ face-conforming)
        if polytope === nothing
            refine_tetrahedron!(verts,simps,f,level, corner_0,value_0, corner_1,value_1, corner_3,value_3, corner_7,value_7, 0, maxdepth, nothing)
            refine_tetrahedron!(verts,simps,f,level, corner_0,value_0, corner_1,value_1, corner_5,value_5, corner_7,value_7, 0, maxdepth, nothing)
            refine_tetrahedron!(verts,simps,f,level, corner_0,value_0, corner_2,value_2, corner_3,value_3, corner_7,value_7, 0, maxdepth, nothing)
            refine_tetrahedron!(verts,simps,f,level, corner_0,value_0, corner_2,value_2, corner_6,value_6, corner_7,value_7, 0, maxdepth, nothing)
            refine_tetrahedron!(verts,simps,f,level, corner_0,value_0, corner_4,value_4, corner_5,value_5, corner_7,value_7, 0, maxdepth, nothing)
            refine_tetrahedron!(verts,simps,f,level, corner_0,value_0, corner_4,value_4, corner_6,value_6, corner_7,value_7, 0, maxdepth, nothing)
        else
            refine_tetrahedron!(verts,simps,f,level, corner_0,value_0, corner_1,value_1, corner_3,value_3, corner_7,value_7, 0, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, corner_0,value_0, corner_1,value_1, corner_5,value_5, corner_7,value_7, 0, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, corner_0,value_0, corner_2,value_2, corner_3,value_3, corner_7,value_7, 0, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, corner_0,value_0, corner_2,value_2, corner_6,value_6, corner_7,value_7, 0, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, corner_0,value_0, corner_4,value_4, corner_5,value_5, corner_7,value_7, 0, maxdepth, polytope, buffer_A, buffer_B)
            refine_tetrahedron!(verts,simps,f,level, corner_0,value_0, corner_4,value_4, corner_6,value_6, corner_7,value_7, 0, maxdepth, polytope, buffer_A, buffer_B)
        end
    end
    return nothing
end

"""
    level_set(f, level, xlims, ylims, zlims, initial_depth; refinement_depth=initial_depth, samples=Array{Float64,3}(undef,N,N,N), weld=false, mask=nothing)

Return the `level` iso-surface of `f` over the box `xlims × ylims × zlims` as a `SimplexMesh{3}`. The
coarse seed grid has `N = 2^initial_depth + 1` points per axis (i.e. `initial_depth` subdivisions of the
box); each straddling cell is then adaptively refined into 8 sub-tetrahedra until the total subdivision
depth from the bounding box reaches `refinement_depth`, so the finest cell spans `extent / 2^refinement_depth`
regardless of `initial_depth`. `refinement_depth` must be `≥ initial_depth`.

`mask`, a `ConvexPolytope{3}`, restricts the iso-surface to that polytope: the surface outside it is
dropped and the surface reaching it is snapped to its boundary.
"""
function level_set(f, level::Real,
                   xlims::Tuple{<:Real,<:Real}, ylims::Tuple{<:Real,<:Real}, zlims::Tuple{<:Real,<:Real},
                   initial_depth::Integer;
                   refinement_depth::Integer=initial_depth,
                   samples::AbstractArray{Float64,3}=Array{Float64,3}(undef, 2^max(initial_depth, 0) + 1, 2^max(initial_depth, 0) + 1, 2^max(initial_depth, 0) + 1),
                   weld::Bool=false,
                   mask::Union{Nothing,ConvexPolytope}=nothing
                   )
    initial_depth ≥ 0 || throw(ArgumentError("initial_depth must be nonnegative, got initial_depth = $initial_depth"))
    refinement_depth ≥ initial_depth || throw(ArgumentError("refinement_depth (=$refinement_depth) must be ≥ initial_depth (=$initial_depth)"))
    N = 2^initial_depth + 1
    maxdepth = refinement_depth - initial_depth
    level_value = Float64(level)
    x0, x1 = Float64(xlims[1]), Float64(xlims[2])
    y0, y1 = Float64(ylims[1]), Float64(ylims[2])
    z0, z1 = Float64(zlims[1]), Float64(zlims[2])
    dx = (x1 - x0) / (N - 1)
    dy = (y1 - y0) / (N - 1)
    dz = (z1 - z0) / (N - 1)

    polytope = mask
    buffer_A = SVector{3,Float64}[]
    buffer_B = SVector{3,Float64}[]
    if polytope !== nothing
        capacity = 3 + length(polytope.faces)   # a clipped triangle gains ≤ 1 vertex per face
        sizehint!(buffer_A, capacity)
        sizehint!(buffer_B, capacity)
    end

    verts = SVector{3,Float64}[]
    simps = NTuple{3,Int}[]
    sizehint!(verts, 12N)
    sizehint!(simps, 4N)

    seed_grid!(verts, simps, f, level_value, x0, y0, z0, dx, dy, dz, N, maxdepth, samples,
               polytope, buffer_A, buffer_B)  # function barrier

    mesh = SimplexMesh{3}(verts, simps)
    if weld
        # The bin width tracks the finest cell extent — extent / 2^refinement_depth, independent of
        # initial_depth — so distinct contour vertices stay in distinct bins at any refinement depth;
        # the floor keeps the tolerance above floating-point noise in vertex coordinates (~eps · extent).
        cell_fine = min(x1 - x0, y1 - y0, z1 - z0) / 2.0^refinement_depth
        atol = max(1e-9 * max(x1 - x0, y1 - y0, z1 - z0), 1e-3 * cell_fine)
        return merge_vertices(mesh; atol)
    end
    return mesh
end

###
### Vertex welding
###

"""
    merge_vertices(mesh::SimplexMesh{D}; atol)

Return a copy of `mesh` with vertices that coincide within `atol` merged and simplices reindexed.
Simplices whose vertices collapse onto one another under the merge become degenerate and are dropped,
so every retained simplex has strictly positive measure.
"""
function merge_vertices(mesh::SimplexMesh{D}; atol::Real) where {D}
    inverse_bin_width = 1.0 / Float64(atol)
    bins = Dict{NTuple{D,Int},Int}()
    merged_vertices = SVector{D,Float64}[]
    vertex_remap = Vector{Int}(undef, length(mesh.vertices))
    for (i, vertex) in enumerate(mesh.vertices)
        key = ntuple(d -> round(Int, vertex[d] * inverse_bin_width), D)
        merged_index = get(bins, key, 0)
        if merged_index == 0
            push!(merged_vertices, vertex)
            merged_index = length(merged_vertices)
            bins[key] = merged_index
        end
        vertex_remap[i] = merged_index
    end
    merged_simplices = NTuple{D,Int}[]
    for simplex in mesh.simplices
        remapped = ntuple(k -> vertex_remap[simplex[k]], D)
        allunique(remapped) && push!(merged_simplices, remapped)
    end
    return SimplexMesh{D}(merged_vertices, merged_simplices)
end
