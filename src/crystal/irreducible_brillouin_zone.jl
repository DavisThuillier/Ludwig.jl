Base.@kwdef struct IrreducibleBrillouinZone{D, T<:AbstractFloat} <: AbstractConvexPolytope{D, T}
    vertices::Vector{SVector{D, T}}
    ridges::Vector{Tuple{Int, Int}} # 3D only
    facets::Vector{Facet{D,T}} # Edges in 2D, faces in 3D
    normals::Vector{SVector{D,T}}
    offsets::Vector{T}
end

"""
    IrreducibleBrillouinZone(lattice::Lattice, pg::PointGroup)

Construct the irreducible Brillouin zone (IBZ) of `lattice` under the action of
the crystallographic point group `pg`. The result is the convex polytope

``\\mathrm{IBZ} = \\mathrm{BZ} \\cap \\bigcap_{g \\neq e} \\{k : (g\\!\\cdot\\!p - p)\\cdot k \\leq 0\\}``

— the Voronoi cell of a generic interior point `p` taken over the orbit
``\\{g\\!\\cdot\\!p : g \\in G\\}``, intersected with the Brillouin zone. Each
non-identity ``g`` is orthogonal so ``|g\\!\\cdot\\!p| = |p|`` and the bisecting
plane passes through ``\\Gamma``; the IBZ therefore has ``\\Gamma`` as a vertex
and volume ``\\mathrm{vol}(\\mathrm{BZ}) / |G|``.

Supports `D = 2` and `D = 3`.
"""
function IrreducibleBrillouinZone(lattice::Lattice{D, T, L},
                                  pg::PointGroup{D, T, L}) where {D, T, L}
    bz = lattice.brillouin_zone

    # Generic interior point with trivial stabilizer in any crystallographic
    # point group: an irrational-ratio direction (log10 of small integers) is
    # incommensurate with all 1-, 2-, 3-, 4-, 6-fold rotations and standard
    # mirrors. Scaled small enough to lie strictly inside the BZ.
    coeffs = D == 3 ? SVector{3, T}(0.30102, 0.47712, 0.60206) :
                      SVector{2, T}(0.30102, 0.47712)
    d = lattice.reciprocal_vectors * coeffs
    p = T(0.1) * minimum(bz.offsets) / norm(d) * d

    iden = SMatrix{D, D, T, L}(I)
    tol = sqrt(eps(T))
    for g in pg.operations
        isapprox(g, iden; atol=tol) && continue
        norm(g * p - p) > tol * norm(p) || error(
            "generic-point heuristic failed: stabilizer of p in pg is non-trivial"
        )
    end

    # Half-space list: BZ facets ∪ orbit bisectors through Γ.
    normals = SVector{D, T}[f.normal for f in bz.facets]
    offsets = T[f.offset for f in bz.facets]
    for g in pg.operations
        isapprox(g, iden; atol=tol) && continue
        w = g * p - p
        push!(normals, w / norm(w))
        push!(offsets, zero(T))
    end

    scale = 4 * maximum(bz.offsets)^2
    poly = polytope_from_halfspaces(normals, offsets, scale)

    return IrreducibleBrillouinZone{D, T}(;
        vertices = poly.vertices,
        ridges   = poly.ridges,
        facets   = poly.facets,
        normals  = poly.normals,
        offsets  = poly.offsets,
    )
end
