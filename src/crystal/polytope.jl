abstract type AbstractConvexPolytope{D, T} end

vertices(p::AbstractConvexPolytope) = p.vertices

"""
    Facet{D, T}
"""
struct Facet{D, T}
    vertices::Vector{Int} # Indices of vertices in parent polytope
    normal::SVector{D, T} # outward unit normal
    offset::T             # n · k = offset (for k on the face)
end

normal(f::Facet) = f.normal
offset(f::Facet) = f.offset

function contains(p::AbstractConvexPolytope{D}, k::SVector{D}) where {D}
    for facet in p.facets
        facet.normal ⋅ k > facet.offset && return false
    end
    return true
end

"""
    volume(p::AbstractConvexPolytope{D, T}) -> T

Return the (`D`-dimensional) volume of a convex polytope. Uses the divergence
theorem with `F = x/D`: for each facet, ``\\int_{\\text{facet}} (n\\cdot x)\\,dA``
is summed, then divided by `D`. In 3D each facet is fan-triangulated about its
first vertex; the polytope's facets are assumed to have vertices in cyclic
order (as produced by [`polytope_from_halfspaces`](@ref)).
"""
function volume(p::AbstractConvexPolytope{2, T}) where {T}
    A = zero(T)
    for f in p.facets
        v1 = p.vertices[f.vertices[1]]
        v2 = p.vertices[f.vertices[2]]
        A += dot(f.normal, v1 + v2) * norm(v2 - v1)
    end
    return A / 4
end

function volume(p::AbstractConvexPolytope{3, T}) where {T}
    V = zero(T)
    for f in p.facets
        verts = f.vertices
        v0 = p.vertices[verts[1]]
        for i in 2:length(verts) - 1
            v1 = p.vertices[verts[i]]
            v2 = p.vertices[verts[i + 1]]
            V += dot(v0, cross(v1 - v0, v2 - v0))
        end
    end
    return V / 6
end

# Cyclic-order facet vertex indices counterclockwise around the facet centroid as
# viewed from the +n side (outward), so fan triangulation (idx[1], idx[i], idx[i+1])
# tiles the planar face. 3D only — `cross` is defined for 3-vectors.
function sort_facet_cyclic(verts, idx, n::SVector{3})
    pts = [verts[i] for i in idx]
    c = sum(pts) / length(pts)
    e = abs(n[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
    u = normalize(e - dot(e, n) * n)
    w = cross(n, u)
    angles = [atan(dot(p - c, w), dot(p - c, u)) for p in pts]
    return idx[sortperm(angles)]
end

"""
    polytope_from_halfspaces(normals, offsets, scale)

Build a bounded convex polytope from the half-space intersection
``\\{k : n_i \\cdot k \\leq c_i\\ \\forall i\\}``. `normals` must be unit vectors.
`scale` is a length² (e.g. the squared distance from the origin to the farthest
facet) used to set ε tolerances for vertex deduplication and incidence tests.

Returns a `NamedTuple` with fields `vertices`, `ridges`, `facets`, `normals`,
`offsets` matching the layout of [`BrillouinZone`](@ref). Half-space planes
whose intersection with the polytope is lower-dimensional (fewer than `D`
incident vertices) are dropped from the returned `normals`/`offsets`/`facets`,
and the input `normals`/`offsets` are passed through unchanged otherwise.
Supports `D = 2` and `D = 3`.
"""
function polytope_from_halfspaces(normals::Vector{SVector{D, T}},
                                  offsets::Vector{T},
                                  scale::T) where {D, T}
    D in (2, 3) || throw(ArgumentError(
        "polytope_from_halfspaces is only implemented for D = 2 or D = 3"
    ))
    length(normals) == length(offsets) || throw(DimensionMismatch(
        "normals and offsets must have the same length"
    ))

    nf = length(normals)
    ε = sqrt(eps(T))
    L = sqrt(scale)

    # A vertex is the intersection of D facet planes (n_i·k = c_i) that lies
    # inside every remaining half-space. Multiple D-subsets can hit the same
    # vertex when more than D facets meet there, so we deduplicate as we go.
    vertices = SVector{D, T}[]
    vertex_facets = Vector{Vector{Int}}()
    M = Matrix{T}(undef, D, D)
    b = Vector{T}(undef, D)
    for inds in Iterators.product(ntuple(_ -> 1:nf, D)...)
        all(inds[i] < inds[i + 1] for i in 1:D - 1) || continue

        for d in 1:D
            M[d, :] .= normals[inds[d]]
            b[d] = offsets[inds[d]]
        end
        abs(det(M)) ≥ ε || continue

        v = SVector{D, T}(M \ b)

        all(m in inds || dot(normals[m], v) ≤ offsets[m] + ε * L
            for m in 1:nf) || continue

        existing = findfirst(u -> sum(abs2, u - v) < ε^2 * scale, vertices)
        if existing === nothing
            push!(vertices, v)
            push!(vertex_facets, collect(inds))
        else
            for i in inds
                i in vertex_facets[existing] || push!(vertex_facets[existing], i)
            end
        end
    end

    # Pick up incidences missed when several D-tuples that define the same vertex
    # all degenerated (e.g. a 4-valent vertex whose three triples each had one
    # near-singular system).
    for (vi, v) in enumerate(vertices)
        for m in 1:nf
            m in vertex_facets[vi] && continue
            abs(dot(normals[m], v) - offsets[m]) ≤ ε * L &&
                push!(vertex_facets[vi], m)
        end
        sort!(vertex_facets[vi])
    end

    # Drop bisector planes whose intersection with the polytope is
    # lower-dimensional (a true (D-1)-facet must support at least D vertices).
    keep = trues(nf)
    facet_verts = [Int[] for _ in 1:nf]
    for (vi, vf) in enumerate(vertex_facets), i in vf
        push!(facet_verts[i], vi)
    end
    for i in 1:nf
        keep[i] = length(facet_verts[i]) ≥ D
    end
    new_facet = cumsum(keep)
    for vf in vertex_facets
        filter!(i -> keep[i], vf)
        map!(i -> new_facet[i], vf, vf)
    end

    kept_normals = SVector{D, T}[]
    kept_offsets = T[]
    facets = Facet{D, T}[]
    for i in 1:nf
        keep[i] || continue
        push!(kept_normals, normals[i])
        push!(kept_offsets, offsets[i])
        verts = D == 3 ? sort_facet_cyclic(vertices, facet_verts[i], normals[i]) :
                         facet_verts[i]
        push!(facets, Facet{D, T}(verts, normals[i], offsets[i]))
    end

    ridges = if D == 3
        # Two vertices form an edge iff they share at least D-1 facets.
        rs = Tuple{Int, Int}[]
        for i in eachindex(vertices), j in (i + 1):lastindex(vertices)
            length(intersect(vertex_facets[i], vertex_facets[j])) ≥ D - 1 &&
                push!(rs, (i, j))
        end
        sort!(rs)
    else
        Tuple{Int, Int}[]
    end

    return (; vertices, ridges, facets, normals=kept_normals, offsets=kept_offsets)
end
