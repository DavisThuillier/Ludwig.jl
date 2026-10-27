
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

# Perpendicular-bisector half-spaces between the origin and each nearby reciprocal lattice point;
# the Wigner–Seitz cell is their intersection. `shell` bounds the integer index box: 2 covers
# standard Bravais lattices, a highly oblique basis may need more.
function bz_halfspaces(rlv::SMatrix{D}, shell::Int) where {D}
    halfspaces = HalfSpace{D}[]
    neighbors = vec([SVector{D,Int}(Tuple(c)) for c ∈ CartesianIndices(ntuple(_ -> -shell:shell, Val(D)))])
    for neighbor in neighbors
        iszero(neighbor) && continue
        G = rlv * SVector{D,Float64}(neighbor)
        magnitude = norm(G)
        push!(halfspaces, HalfSpace{D}(G / magnitude, magnitude / 2))
    end
    return halfspaces
end

"""
    brillouin_zone(rlv; shell=2) -> ConvexPolytope{D}

The first Brillouin zone (Wigner–Seitz cell of the reciprocal lattice) as a convex polytope, built
from the perpendicular bisectors between the origin and its neighbouring reciprocal lattice points.
"""
function brillouin_zone(rlv::SMatrix{D}; shell::Int = 2) where {D}
    return prune_redundant_faces(ConvexPolytope{D}(bz_halfspaces(rlv, shell)))
end

# A generic direction with mutually irrational coordinates, so no non-identity point-group operation
# fixes it. Used as the seed whose orbit cuts the BZ down to a fundamental domain.
function generic_seed(rlv::SMatrix{D}) where {D}
    components = (Float64(π) - 3, Float64(ℯ) - 2, sqrt(2.0) - 1)   # ≈ (0.142, 0.718, 0.414)
    return rlv * SVector{D,Float64}(ntuple(i -> components[i], D))
end

"""
    irreducible_brillouin_zone(rlv, point_group; shell=2) -> ConvexPolytope{D}

Generate the intersection of the fundamental domain of `point_group` and the first Brillouin Zone.
"""
function irreducible_brillouin_zone(rlv::SMatrix{D}, point_group::PointGroup{D}; shell::Int = 2) where {D}
    halfspaces = bz_halfspaces(rlv, shell)
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
