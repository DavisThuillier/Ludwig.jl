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

function volume(p::AbstractConvexPolytope{D}) where {D} end
