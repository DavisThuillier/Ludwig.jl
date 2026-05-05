Base.@kwdef struct BrillouinZone{D, T<:AbstractFloat} <: AbstractConvexPolytope{D, T}
    vertices::Vector{SVector{D, T}}
    ridges::Vector{Tuple{Int, Int}} # 3D only
    facets::Vector{Facet{D,T}} # Edges in 2D, faces in 3D
    normals::Vector{SVector{D,T}}
    offsets::Vector{T}
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
        "BrillouinZone construction is only implemented for D = 2 or D = 3"
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
    normals = SVector{D, T}[]
    offsets = T[]
    for G in Gs
        all(G == Gp || dot(G, Gp) ≤ dot(Gp, Gp) + ε * g2 for Gp in Gs) || continue
        nG = norm(G)
        push!(normals, G / nG)
        push!(offsets, nG / 2)
    end

    poly = polytope_from_halfspaces(normals, offsets, g2)

    return BrillouinZone{D, T}(;
        vertices = poly.vertices,
        ridges   = poly.ridges,
        facets   = poly.facets,
        normals  = poly.normals,
        offsets  = poly.offsets,
    )
end
