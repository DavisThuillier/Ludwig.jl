
struct Lattice{D}
    primitives::SMatrix{D,D,Float64}
    reciprocal_lattice_vectors::SMatrix{D,D,Float64}
end

primitives(lattice::Lattice) = lattice.primitives
reciprocal_lattice_vectors(lattice::Lattice) = lattice.reciprocal_lattice_vectors

function Lattice(primitives::AbstractMatrix)
    D = LinearAlgebra.checksquare(primitives) # Throw error if not square

    lattice_vectors  = SMatrix{D,D}(primitives)
    reciprocal_lattice_vectors= 2π * transpose(inv(primitives))
    return Lattice{D}(lattice_vectors, reciprocal_lattice_vectors)
end

struct Crystal{D}
    lattice::Lattice{D}
    point_group::PointGroup{D}
end

Crystal(primitives::SMatrix{D}, point_group::PointGroup{D}) where {D} = Crystal(Lattice(primitives), point_group)

lattice(crystal::Crystal) = crystal.lattice
primitives(crystal::Crystal) = crystal.lattice.primitives
reciprocal_lattice_vectors(crystal::Crystal) = crystal.lattice.reciprocal_lattice_vectors
point_group(crystal::Crystal) = crystal.point_group

###
### Brillouin zone construction
###

# Perpendicular bisector between the origin and the reciprocal lattice point with integer indices `neighbor`.
function bisector(rlv::SMatrix{D}, neighbor::SVector{D,Int}) where {D}
    G = rlv * SVector{D,Float64}(neighbor)
    magnitude = norm(G)
    return HalfSpace{D}(G / magnitude, magnitude / 2)
end

index_box(shell::Int, ::Val{D}) where {D} =
    (SVector{D,Int}(Tuple(c)) for c ∈ CartesianIndices(ntuple(_ -> -shell:shell, Val(D))))


function bz_halfspaces(rlv::SMatrix{D}) where {D}
    shell = 2 # Initialize search radius to +-2 reciprocal lattice vectors
    halfspaces = [bisector(rlv, neighbor) for neighbor in index_box(shell, Val(D)) if !iszero(neighbor)]
    smallest_singular_value = minimum(svdvals(rlv)) # Lower bound on distance to neighbors
    circumradius = maximum(norm, vertices(ConvexPolytope{D}(halfspaces))) # Upper bound on final cell size 
    while smallest_singular_value * (shell + 1) < 2 * circumradius # Absolute upper bound for rejecting neighbors
        shell += 1
        for neighbor in index_box(shell, Val(D))
            maximum(abs, neighbor) == shell || continue   # only the newly added outer layer
            norm(rlv * SVector{D,Float64}(neighbor)) < 2 * circumradius || continue
            push!(halfspaces, bisector(rlv, neighbor))
        end
        circumradius = maximum(norm, vertices(ConvexPolytope{D}(halfspaces)))
    end
    return halfspaces
end

"""
    brillouin_zone(rlv) -> ConvexPolytope{D}

The first Brillouin zone (Wigner–Seitz cell of the reciprocal lattice) as a convex polytope, built
from the perpendicular bisectors between the origin and its neighbouring reciprocal lattice points.
"""
function brillouin_zone(rlv::SMatrix{D}) where {D}
    return prune_redundant_faces(ConvexPolytope{D}(bz_halfspaces(rlv)))
end

"""
    generic_seed(rlv) -> SVector{D,Float64}

A generic direction with mutually irrational coordinates, so no non-identity point-group operation
fixes it.
"""
function generic_seed(rlv::SMatrix{D}) where {D}
    components = (Float64(π) - 3, Float64(ℯ) - 2, sqrt(2.0) - 1)   # ≈ (0.142, 0.718, 0.414)
    return rlv * SVector{D,Float64}(ntuple(i -> components[i], D))
end

"""
    irreducible_brillouin_zone(rlv, point_group) -> ConvexPolytope{D}

Generate the intersection of the fundamental domain of `point_group` and the first Brillouin Zone.
"""
function irreducible_brillouin_zone(rlv::SMatrix{D}, point_group::PointGroup{D}) where {D}
    halfspaces = bz_halfspaces(rlv)
    seed = generic_seed(rlv)
    for g in operations(point_group)
        normal = g * seed - seed
        magnitude = norm(normal)
        magnitude ≤ 1e-9 && continue               # identity / stabilizer: no constraint
        push!(halfspaces, HalfSpace{D}(normal / magnitude, 0.0))
    end
    return prune_redundant_faces(ConvexPolytope{D}(halfspaces))
end

irreducible_brillouin_zone(crystal::Crystal) =
    irreducible_brillouin_zone(reciprocal_lattice_vectors(crystal), point_group(crystal))
brillouin_zone(crystal::Crystal) = brillouin_zone(reciprocal_lattice_vectors(crystal))
