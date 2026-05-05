Base.@kwdef struct IrreducibleBrillouinZone{D, T}
    vertices::Vector{SVector{D, T}}
    ridges::Vector{Tuple{Int, Int}} # 3D only
    facets::Vector{Facet{D,T}} # Edges in 2D, Faces in 3D
    normals::Vector{SVector{D,T}}
    offsets::Vector{T}
end

# function IrreducibleBrillouinZone(lattice::Lattice{D,T,L}, point_group::PointGroup{D,T,L})
    # return 
# end
