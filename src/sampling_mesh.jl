###
### Closest point on a simplex
###

# Barycentric weights on triangle `(vertex_a, vertex_b, vertex_c)` of the point on it closest to `query`,
# by Voronoi region test (Ericson, Real-Time Collision Detection, §5.1.5 — robust for the obtuse/sliver
# triangles the marcher emits). Weights are nonnegative and sum to one.
function closest_point_weights(query::SVector{3,Float64}, vertex_a::SVector{3,Float64},
                               vertex_b::SVector{3,Float64}, vertex_c::SVector{3,Float64})
    edge_ab = vertex_b - vertex_a
    edge_ac = vertex_c - vertex_a
    a_to_query = query - vertex_a
    ab_dot_a_to_query = dot(edge_ab, a_to_query)
    ac_dot_a_to_query = dot(edge_ac, a_to_query)
    # vertex region A
    (ab_dot_a_to_query ≤ 0.0 && ac_dot_a_to_query ≤ 0.0) && return (1.0, 0.0, 0.0)

    b_to_query = query - vertex_b
    ab_dot_b_to_query = dot(edge_ab, b_to_query)
    ac_dot_b_to_query = dot(edge_ac, b_to_query)
    # vertex region B
    (ab_dot_b_to_query ≥ 0.0 && ac_dot_b_to_query ≤ ab_dot_b_to_query) && return (0.0, 1.0, 0.0)

    # nonpositive ⇔ query lies outside edge ab
    barycentric_numerator_c = ab_dot_a_to_query * ac_dot_b_to_query - ab_dot_b_to_query * ac_dot_a_to_query
    if barycentric_numerator_c ≤ 0.0 && ab_dot_a_to_query ≥ 0.0 && ab_dot_b_to_query ≤ 0.0
        # edge region AB
        t_along_ab = ab_dot_a_to_query / (ab_dot_a_to_query - ab_dot_b_to_query)
        return (1.0 - t_along_ab, t_along_ab, 0.0)
    end

    c_to_query = query - vertex_c
    ab_dot_c_to_query = dot(edge_ab, c_to_query)
    ac_dot_c_to_query = dot(edge_ac, c_to_query)
    # vertex region C
    (ac_dot_c_to_query ≥ 0.0 && ab_dot_c_to_query ≤ ac_dot_c_to_query) && return (0.0, 0.0, 1.0)

    # nonpositive ⇔ query lies outside edge ac
    barycentric_numerator_b = ab_dot_c_to_query * ac_dot_a_to_query - ab_dot_a_to_query * ac_dot_c_to_query
    if barycentric_numerator_b ≤ 0.0 && ac_dot_a_to_query ≥ 0.0 && ac_dot_c_to_query ≤ 0.0
        # edge region AC
        t_along_ac = ac_dot_a_to_query / (ac_dot_a_to_query - ac_dot_c_to_query)
        return (1.0 - t_along_ac, 0.0, t_along_ac)
    end

    # nonpositive ⇔ query lies outside edge bc
    barycentric_numerator_a = ab_dot_b_to_query * ac_dot_c_to_query - ab_dot_c_to_query * ac_dot_b_to_query
    if barycentric_numerator_a ≤ 0.0 && (ac_dot_b_to_query - ab_dot_b_to_query) ≥ 0.0 &&
                                        (ab_dot_c_to_query - ac_dot_c_to_query) ≥ 0.0
        # edge region BC
        t_along_bc = (ac_dot_b_to_query - ab_dot_b_to_query) /
                     ((ac_dot_b_to_query - ab_dot_b_to_query) + (ab_dot_c_to_query - ac_dot_c_to_query))
        return (0.0, 1.0 - t_along_bc, t_along_bc)
    end

    # face interior
    inverse_total_area = 1.0 / (barycentric_numerator_a + barycentric_numerator_b + barycentric_numerator_c)
    weight_b = barycentric_numerator_b * inverse_total_area
    weight_c = barycentric_numerator_c * inverse_total_area
    return (1.0 - weight_b - weight_c, weight_b, weight_c)
end

# Weights and distance of the closest point of simplex `i` to `query`.
function simplex_closest(mesh::SimplexMesh{3}, i::Int, query::SVector{3,Float64})
    vertex_a, vertex_b, vertex_c = build_simplex(i, mesh)
    weights = closest_point_weights(query, vertex_a, vertex_b, vertex_c)
    closest = weights[1] * vertex_a + weights[2] * vertex_b + weights[3] * vertex_c
    return weights, norm(query - closest)
end

function simplex_closest(mesh::SimplexMesh{2}, i::Int, query::SVector{2,Float64})
    vertex_a, vertex_b = build_simplex(i, mesh)
    edge_ab = vertex_b - vertex_a
    t_along_edge = clamp(dot(query - vertex_a, edge_ab) / dot(edge_ab, edge_ab), 0.0, 1.0)
    return (1.0 - t_along_edge, t_along_edge), norm(query - (vertex_a + t_along_edge * edge_ab))
end

###
### Fine → coarse surface map
###

"""
    SurfaceLocator{D}

Spatial index over a coarse `SimplexMesh{D}` for locating the simplex nearest a query point. Coarse
simplices are bucketed on a uniform grid sized to the mesh's mean simplex extent, so a query near the
surface resolves in `O(1)` cells.
"""
struct SurfaceLocator{D}
    mesh::SimplexMesh{D}
    origin::SVector{D,Float64}
    cell::Float64
    inv_cell::Float64
    dims::NTuple{D,Int}
    buckets::Dict{NTuple{D,Int},Vector{Int}}
end

function characteristic_length(mesh::SimplexMesh{D}) where {D}
    total = 0.0
    for i in eachindex(mesh.simplices)
        simplex_vertices = build_simplex(i, mesh)
        longest_edge = 0.0
        for j in 1:D, k in j+1:D
            longest_edge = max(longest_edge, norm(simplex_vertices[k] - simplex_vertices[j]))
        end
        total += longest_edge
    end
    return total / length(mesh.simplices)
end

"""
    surface_locator(mesh::SimplexMesh)

Build a [`SurfaceLocator`](@ref) over the coarse `mesh`, ready to answer [`locate`](@ref) queries.
"""
function surface_locator(mesh::SimplexMesh{D}) where {D}
    buckets = Dict{NTuple{D,Int},Vector{Int}}()
    isempty(mesh.simplices) &&
        return SurfaceLocator{D}(mesh, zero(SVector{D,Float64}), 1.0, 1.0, ntuple(_ -> 1, D), buckets)

    box_min = reduce((u, v) -> min.(u, v), mesh.vertices)
    box_max = reduce((u, v) -> max.(u, v), mesh.vertices)
    extent = box_max - box_min
    cell = max(characteristic_length(mesh), 1e-12 * maximum(extent), floatmin(Float64))
    inv_cell = 1.0 / cell
    dims = ntuple(d -> max(1, ceil(Int, extent[d] * inv_cell) + 1), D)

    for i in eachindex(mesh.simplices)
        simplex_vertices = build_simplex(i, mesh)
        simplex_min = reduce((u, v) -> min.(u, v), simplex_vertices)
        simplex_max = reduce((u, v) -> max.(u, v), simplex_vertices)
        cell_min = ntuple(d -> clamp(floor(Int, (simplex_min[d] - box_min[d]) * inv_cell), 0, dims[d] - 1), D)
        cell_max = ntuple(d -> clamp(floor(Int, (simplex_max[d] - box_min[d]) * inv_cell), 0, dims[d] - 1), D)
        for cell_index in CartesianIndices(ntuple(d -> cell_min[d]:cell_max[d], D))
            push!(get!(buckets, Tuple(cell_index), Int[]), i)
        end
    end
    return SurfaceLocator{D}(mesh, box_min, cell, inv_cell, dims, buckets)
end

# Scan the Chebyshev-radius-`shell_radius` shell of cells around `base_cell` for a simplex closer to
# `query` than `best`. A plain function so `shell_radius` is never captured in a closure and boxed:
# passing it as an argument keeps the per-query shell sweep allocation-free.
function scan_shell(locator::SurfaceLocator{D}, query::SVector{D,Float64}, base_cell::NTuple{D,Int},
                    shell_radius::Int, best::Tuple{Int,NTuple{D,Float64},Float64}) where {D}
    best_simplex, best_weights, best_distance = best
    for offset in CartesianIndices(ntuple(_ -> -shell_radius:shell_radius, Val(D)))
        maximum(abs, Tuple(offset)) == shell_radius || continue   # only the shell at this Chebyshev radius
        cell = ntuple(d -> base_cell[d] + offset[d], Val(D))
        cell_simplices = get(locator.buckets, cell, nothing)
        cell_simplices === nothing && continue
        for i in cell_simplices
            weights, distance = simplex_closest(locator.mesh, i, query)
            distance < best_distance && ((best_simplex, best_weights, best_distance) = (i, weights, distance))
        end
    end
    return (best_simplex, best_weights, best_distance)
end

"""
    locate(locator::SurfaceLocator{D}, query) -> (simplex, weights, distance)

Coarse `simplex` nearest to `query`, the barycentric `weights` (one per simplex vertex, nonnegative,
summing to one) of its closest point, and that `distance`. A large `distance` flags a fine point with no
faithful coarse counterpart — e.g. across an IBZ mask seam where the two clipped surfaces disagree by
`O(h)`. Returns `(0, zeros, Inf)` when the coarse mesh is empty.
"""
function locate(locator::SurfaceLocator{D}, query::SVector{D,Float64}) where {D}
    best = (0, ntuple(_ -> 0.0, Val(D)), Inf)
    isempty(locator.mesh.simplices) && return best

    base_cell = ntuple(d -> clamp(floor(Int, (query[d] - locator.origin[d]) * locator.inv_cell),
                                  0, locator.dims[d] - 1), Val(D))
    shell_radius = 0
    while shell_radius ≤ maximum(locator.dims)
        best = scan_shell(locator, query, base_cell, shell_radius, best)
        # cells beyond this shell are at least shell_radius·cell from the query, so once best is that
        # close no closer one remains.
        best[1] != 0 && shell_radius * locator.cell ≥ best[3] && break
        shell_radius += 1
    end
    return best
end

"""
    interpolate(locator::SurfaceLocator, values::AbstractMatrix, query)

Barycentrically interpolate the coarse per-vertex `values` (one column per field) at the fine point
`query`, via the coarse simplex `query` maps to. Returns a `NaN` vector when `query` has no coarse
counterpart.
"""
function interpolate(locator::SurfaceLocator{D}, values::AbstractMatrix, query::SVector{D,Float64}) where {D}
    simplex_index, weights, _ = locate(locator, query)
    simplex_index == 0 && return fill(NaN, size(values, 2))
    interpolated = zeros(size(values, 2))
    for (vertex, weight) in zip(locator.mesh.simplices[simplex_index], weights)
        @views interpolated .+= weight .* values[vertex, :]
    end
    return interpolated
end

"""
    scatter_to_vertices(locator::SurfaceLocator, points, masses) -> Vector

Attribute each `masses[i]` at `points[i]` onto the coarse mesh vertices, split by the barycentric weights
of the coarse simplex the point maps to. Conserves total mass exactly except for points with no coarse
counterpart (`locate` simplex `0`), which are dropped — the residual to check against an IBZ mask seam.
"""
function scatter_to_vertices(locator::SurfaceLocator{D}, points, masses) where {D}
    accumulated = zeros(length(locator.mesh.vertices))
    for (point, mass) in zip(points, masses)
        simplex_index, weights, _ = locate(locator, point)
        simplex_index == 0 && continue
        for (vertex, weight) in zip(locator.mesh.simplices[simplex_index], weights)
            accumulated[vertex] += weight * mass
        end
    end
    return accumulated
end

###
### Sampling-mesh construction
###

"""
    sampling_mesh(f, level, xlims, ylims, zlims, initial_depth; sampling_depth, mask)

A coarse, welded `level_set` mesh for a single energy `level`, built at the fixed `sampling_depth` so its
centroids form a depth-stable candidate-node set. `level_set` is deterministic, so a fixed
`(level, sampling_depth)` yields the same mesh regardless of any fine integration mesh's depth. Shares
`initial_depth` with the integration mesh so both detect the same pockets at the common seed grid.
`sampling_depth` is the total subdivision depth from the bounding box (the `refinement_depth` convention)
and must be `≥ initial_depth`.
"""
function sampling_mesh(f, level::Real, xlims::Tuple{<:Real,<:Real}, ylims::Tuple{<:Real,<:Real}, zlims::Tuple{<:Real,<:Real},
                       initial_depth::Integer;
                       sampling_depth::Integer,
                       mask::Union{Nothing,ConvexPolytope} = nothing)
    sampling_depth ≥ initial_depth ||
        throw(ArgumentError("sampling_depth (=$sampling_depth) must be ≥ initial_depth (=$initial_depth)"))
    return level_set(f, level, xlims, ylims, zlims, initial_depth; refinement_depth = sampling_depth, mask, weld = true)
end
