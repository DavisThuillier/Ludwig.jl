Base.@kwdef struct BrillouinZone{D, T<:AbstractFloat} <: AbstractConvexPolytope{D, T}
    vertices::Vector{SVector{D, T}}
    ridges::Vector{Tuple{Int, Int}} # 3D only
    facets::Vector{Facet{D,T}} # Edges in 2D, faces in 3D
    normals::Vector{SVector{D,T}}
    offsets::Vector{T}
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
    BrillouinZone(rlv::SMatrix{D, D, T})

Construct the first Brillouin zone (Wigner-Seitz cell of the reciprocal lattice)
from the matrix `rlv` whose columns are the primitive reciprocal lattice vectors
``\\mathbf{b}_1, \\ldots, \\mathbf{b}_D``.

The cell is the intersection of half-spaces
``\\{\\mathbf{k} : \\mathbf{k} \\cdot \\mathbf{G} \\leq |\\mathbf{G}|^2/2\\}``
for each candidate ``\\mathbf{G} = \\sum_i n_i \\mathbf{b}_i`` with
``n_i \\in \\{-2, \\ldots, 2\\}``, which is sufficient for any 2D or 3D Bravais
lattice. A reciprocal vector ``\\mathbf{G}`` defines a facet iff
``\\mathbf{G} \\cdot \\mathbf{G}' \\leq |\\mathbf{G}'|^2`` for every other
``\\mathbf{G}'``; vertices are then the intersections of `D` facet planes that
satisfy every remaining half-space.

Supports `D = 2` and `D = 3`.
"""
function BrillouinZone(rlv::SMatrix{D, D, T, L}) where {D, T, L}
    D in (2, 3) || throw(ArgumentError(
        "BrillouinZone construction is only implemented for D = 2 or D = 3."
    ))

    Gs = SVector{D, T}[]
    for n in Iterators.product(ntuple(_ -> -2:2, D)...)
        all(iszero, n) && continue
        push!(Gs, rlv * SVector{D, Int}(n))
    end

    g2 = maximum(G -> dot(G, G), Gs)
    ε = sqrt(eps(T))

    # G defines a BZ facet iff origin is the closest reciprocal point to G/2,
    # equivalently G·G' ≤ |G'|² for every other reciprocal vector G'.
    facet_Gs = SVector{D, T}[]
    for G in Gs
        all(G == Gp || dot(G, Gp) ≤ dot(Gp, Gp) + ε * g2 for Gp in Gs) &&
            push!(facet_Gs, G)
    end
    nf = length(facet_Gs)

    # A vertex is the intersection of D facet planes (G_i·k = |G_i|²/2) that lies
    # inside every remaining half-space. Multiple D-subsets can hit the same vertex
    # when more than D facets meet there, so we deduplicate as we go.
    vertices = SVector{D, T}[]
    vertex_facets = Vector{Vector{Int}}()
    M = Matrix{T}(undef, D, D)
    b = Vector{T}(undef, D)
    for inds in Iterators.product(ntuple(_ -> 1:nf, D)...)
        all(inds[i] < inds[i + 1] for i in 1:D - 1) || continue

        for d in 1:D
            G = facet_Gs[inds[d]]
            M[d, :] .= G
            b[d] = dot(G, G) / 2
        end
        abs(det(M)) ≥ ε * sqrt(g2)^D || continue

        v = SVector{D, T}(M \ b)

        all(m in inds ||
            dot(facet_Gs[m], v) ≤ dot(facet_Gs[m], facet_Gs[m]) / 2 + ε * g2
            for m in 1:nf) || continue

        existing = findfirst(u -> sum(abs2, u - v) < ε^2 * g2, vertices)
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
            abs(dot(facet_Gs[m], v) - dot(facet_Gs[m], facet_Gs[m]) / 2) ≤ ε * g2 &&
                push!(vertex_facets[vi], m)
        end
        sort!(vertex_facets[vi])
    end

    # Drop bisector planes whose intersection with the BZ is lower-dimensional
    # (a true (D-1)-facet must support at least D vertices).
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

    normals = SVector{D, T}[]
    offsets = T[]
    facets = Facet{D, T}[]
    for i in 1:nf
        keep[i] || continue
        G = facet_Gs[i]
        nG = norm(G)
        n = G / nG
        c = nG / 2
        push!(normals, n)
        push!(offsets, c)
        verts = D == 3 ? sort_facet_cyclic(vertices, facet_verts[i], n) :
                         facet_verts[i]
        push!(facets, Facet{D, T}(verts, n, c))
    end

    ridges = if D == 3
        # Two vertices form an edge of the BZ iff they share at least D-1 facets.
        rs = Tuple{Int, Int}[]
        for i in eachindex(vertices), j in (i + 1):lastindex(vertices)
            length(intersect(vertex_facets[i], vertex_facets[j])) ≥ D - 1 &&
                push!(rs, (i, j))
        end
        sort!(rs)
    else
        Tuple{Int, Int}[]
    end

    return BrillouinZone{D, T}(; vertices, ridges, facets, normals, offsets)
end
