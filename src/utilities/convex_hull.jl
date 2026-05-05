###
### Quickhull
###

mutable struct QuickhullFacet{D, T}
    verts::NTuple{D, Int}
    normal::SVector{D, T}
    offset::T
    neighbors::Vector{Int}
    outside::Vector{Int}
end

"""
    quickhull(points::Vector{SVector{D, T}}) -> (ridges, facets)

Compute the convex hull of `points` and return its 1-skeleton and
(D-1)-dimensional facets.

Implements the Quickhull algorithm: select a nondegenerate (D+1)-simplex,
partition the remaining points into outside sets, and iteratively cone the
furthest outside point of a facet to its horizon ridges, until every facet has
an empty outside set.

# Returns
- `ridges::Vector{Tuple{Int, Int}}`: each (D-2)-face of the hull, encoded as
  a sorted `Tuple{Int, Int}` of vertex indices into `points`. In 3D ridges are
  the edges shared between adjacent triangular facets. In 2D ridges are
  individual hull vertices, encoded as `(v, v)` (both entries equal).
- `facets::Vector{Facet{D, T}}`: each simplicial (D-1)-facet with its vertex
  indices and outward unit normal/offset (see [`Facet`](@ref)).

The input is assumed to be in general position (no D+1 points lying on a
common (D-2)-flat); roundoff handling for degenerate inputs is not
implemented.

# References
Barber, C. B., Dobkin, D. P., and Huhdanpaa, H. (1996). "The Quickhull
Algorithm for Convex Hulls." *ACM Transactions on Mathematical Software*
22(4):469–483.
"""
function quickhull(points::Vector{SVector{D, T}}) where {D, T}
    n = length(points)
    n ≥ D + 1 || error("Quickhull requires at least D+1 = $(D+1) points")

    simplex = initial_simplex(points)
    interior = sum(points[i] for i in simplex) / (D + 1)

    facets = Dict{Int, QuickhullFacet{D, T}}()
    next_id = Ref(0)
    fresh_id() = (next_id[] += 1; next_id[])

    initial_ids = Int[]
    for skip in 1:D+1
        verts = ntuple(k -> simplex[k < skip ? k : k + 1], D)
        normal, offset = oriented_hyperplane(SVector{D, T}[points[i] for i in verts],
                                             interior)
        id = fresh_id()
        push!(initial_ids, id)
        facets[id] = QuickhullFacet{D, T}(verts, normal, offset, Int[], Int[])
    end
    for id in initial_ids
        f = facets[id]
        f.neighbors = [first(j for j in initial_ids
                             if j != id && !(v in facets[j].verts))
                       for v in f.verts]
    end

    in_simplex = Set(simplex)
    for p in 1:n
        p in in_simplex && continue
        for id in initial_ids
            f = facets[id]
            if f.normal ⋅ points[p] - f.offset > 0
                push!(f.outside, p)
                break
            end
        end
    end

    queue = [id for id in initial_ids if !isempty(facets[id].outside)]
    while !isempty(queue)
        fid = pop!(queue)
        haskey(facets, fid) || continue
        f = facets[fid]
        isempty(f.outside) && continue

        pidx = f.outside[argmax([f.normal ⋅ points[q] - f.offset for q in f.outside])]

        visible = Set{Int}([fid])
        stack = [fid]
        while !isempty(stack)
            for nb in facets[pop!(stack)].neighbors
                nb in visible && continue
                fn = facets[nb]
                if fn.normal ⋅ points[pidx] - fn.offset > 0
                    push!(visible, nb)
                    push!(stack, nb)
                end
            end
        end

        horizons = Tuple{NTuple{D - 1, Int}, Int, Int}[]
        outside_pool = Int[]
        for vid in visible
            fv = facets[vid]
            for (i, nb) in enumerate(fv.neighbors)
                if !(nb in visible)
                    ridge = ntuple(k -> fv.verts[k < i ? k : k + 1], D - 1)
                    push!(horizons, (ridge, vid, nb))
                end
            end
            for q in fv.outside
                q != pidx && push!(outside_pool, q)
            end
        end

        new_ids = Int[]
        ridge_map = Dict{NTuple{D - 1, Int}, Tuple{Int, Int}}()
        for (ridge, vis_id, old_nb) in horizons
            verts = (ridge..., pidx)
            normal, offset = oriented_hyperplane(SVector{D, T}[points[i] for i in verts],
                                                 interior)
            id = fresh_id()
            push!(new_ids, id)
            nbs = Vector{Int}(undef, D)
            nbs[D] = old_nb
            for i in 1:D-1
                sub = ntuple(k -> k == D - 1 ? pidx :
                                  (k < i ? ridge[k] : ridge[k + 1]), D - 1)
                key = NTuple{D - 1, Int}(sort!(collect(sub)))
                if haskey(ridge_map, key)
                    other_id, other_pos = ridge_map[key]
                    nbs[i] = other_id
                    facets[other_id].neighbors[other_pos] = id
                else
                    ridge_map[key] = (id, i)
                    nbs[i] = 0
                end
            end
            facets[id] = QuickhullFacet{D, T}(verts, normal, offset, nbs, Int[])

            of = facets[old_nb]
            for k in 1:D
                if of.neighbors[k] == vis_id
                    of.neighbors[k] = id
                    break
                end
            end
        end

        for q in outside_pool
            for nid in new_ids
                fn = facets[nid]
                if fn.normal ⋅ points[q] - fn.offset > 0
                    push!(fn.outside, q)
                    break
                end
            end
        end

        for vid in visible
            delete!(facets, vid)
        end
        for nid in new_ids
            isempty(facets[nid].outside) || push!(queue, nid)
        end
    end

    out_facets = Facet{D, T}[]
    ridges = Set{Tuple{Int, Int}}()
    for f in values(facets)
        push!(out_facets, Facet{D, T}(collect(f.verts), f.normal, f.offset))
        if D == 2
            for v in f.verts
                push!(ridges, (v, v))
            end
        else
            for i in 1:D, j in (i + 1):D
                a, b = minmax(f.verts[i], f.verts[j])
                push!(ridges, (a, b))
            end
        end
    end
    return sort!(collect(ridges)), out_facets
end

###
### Helpers
###

function initial_simplex(pts::Vector{SVector{D, T}}) where {D, T}
    indices = [argmin([p[1] for p in pts])]
    for _ in 1:D
        best_d, best_i = T(-Inf), 0
        for i in eachindex(pts)
            i in indices && continue
            d = affine_distance(pts, indices, pts[i])
            if d > best_d
                best_d, best_i = d, i
            end
        end
        best_i == 0 && error("Quickhull: input is not full-dimensional")
        push!(indices, best_i)
    end
    return indices
end

function affine_distance(pts::Vector{SVector{D, T}}, indices,
                         p::SVector{D, T}) where {D, T}
    base = pts[first(indices)]
    basis = SVector{D, T}[]
    for i in Iterators.drop(indices, 1)
        v = pts[i] - base
        for b in basis
            v -= (v ⋅ b) * b
        end
        nv = norm(v)
        nv > 0 && push!(basis, v / nv)
    end
    r = p - base
    for b in basis
        r -= (r ⋅ b) * b
    end
    return norm(r)
end

function oriented_hyperplane(pts::Vector{SVector{D, T}},
                             interior::SVector{D, T}) where {D, T}
    base = pts[1]
    M = Matrix{T}(undef, D, D - 1)
    for j in 1:D-1, i in 1:D
        M[i, j] = pts[j + 1][i] - base[i]
    end
    F = qr(M)
    n = SVector{D, T}((F.Q * Matrix{T}(I, D, D))[:, D])
    n /= norm(n)
    offset = n ⋅ base
    if n ⋅ interior > offset
        n = -n
        offset = -offset
    end
    return n, offset
end
