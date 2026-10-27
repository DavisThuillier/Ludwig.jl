###
### Reusable crystal-structure test fixtures
###
#
# Shared by the point-group / BZ / IBZ tests, the representation-theory tests and the
# end-to-end quadrature and collision tests. Build a fixture by calling the helper, e.g.
# `cubic_crystal()`. The 3D helpers return a fully-closed `Crystal{3}` (lattice + point
# group); the point group is constructed from Cartesian generator matrices only (the
# constructor validates that each generator preserves the lattice and closes the group).
#
# Every fixture is a function, so nothing here is a top-level `const` whose redefinition
# would warn. Consumers guard the include all the same,
#
#     @isdefined(cubic_crystal) || include("crystal_fixtures.jl")
#
# because `Pkg.test` runs with `--warn-overwrite=yes`, under which a repeated include warns
# once per method it defines.

using Ludwig
using Ludwig: PointGroup, ConvexPolytope
using StaticArrays
using LinearAlgebra

###
### Cubic, 3D (m-3m, order 48)
###

# Cartesian generators of the full cubic point group m-3m (Oh):
#   - C4z: 90 degree rotation about z,
#   - C3 : 120 degree rotation about the body diagonal, (x,y,z) -> (z,x,y),
#   - inversion.
# Their closure is the order-48 group. Generators are passed as-is; the PointGroup
# constructor closes the group.
cubic_generators() = [
    SMatrix{3,3,Float64}( 0, 1, 0,  -1, 0, 0,   0, 0, 1),   # C4 about z
    SMatrix{3,3,Float64}( 0, 0, 1,   1, 0, 0,   0, 1, 0),   # C3 about (1,1,1)
    SMatrix{3,3,Float64}(-1, 0, 0,   0,-1, 0,   0, 0,-1),   # inversion
]

cubic_point_group(primitives::SMatrix{3,3,Float64}) = PointGroup(cubic_generators(), primitives)

"""
    cubic_crystal(; a = 1.0, kind = :sc) -> Crystal{3}

A cubic `Crystal{3}` carrying the full m-3m point group (order 48). `kind` selects the
conventional cubic Bravais lattice whose conventional cell edge is `a`:
`:sc` (simple cubic), `:fcc`, or `:bcc`. The point group is built from Cartesian
generators only.
"""
function cubic_crystal(; a::Float64 = 1.0, kind::Symbol = :sc)
    primitives = cubic_primitives(a, kind)
    return Crystal(primitives, cubic_point_group(primitives))
end

function cubic_primitives(a::Float64, kind::Symbol)
    if kind === :sc
        return SMatrix{3,3,Float64}(a, 0, 0,  0, a, 0,  0, 0, a)
    elseif kind === :fcc
        return (a / 2) * SMatrix{3,3,Float64}(0, 1, 1,  1, 0, 1,  1, 1, 0)
    elseif kind === :bcc
        return (a / 2) * SMatrix{3,3,Float64}(-1, 1, 1,  1, -1, 1,  1, 1, -1)
    else
        throw(ArgumentError("unknown cubic kind $kind (expected :sc, :fcc, or :bcc)"))
    end
end

###
### Hexagonal, 3D (6/mmm, order 24)
###

"""
    hexagonal_crystal(; a = 1.0, c = 1.6) -> Crystal{3}

A hexagonal `Crystal{3}` carrying the full 6/mmm point group (order 24), with in-plane
lattice constant `a` and out-of-plane constant `c`.
"""
function hexagonal_crystal(; a::Float64 = 1.0, c::Float64 = 1.6)
    primitives = SMatrix{3,3,Float64}(
        a, 0.0, 0.0,
        -a / 2, a * sqrt(3) / 2, 0.0,
        0.0, 0.0, c,
    )
    theta = pi / 3
    generators = [
        SMatrix{3,3,Float64}(cos(theta), sin(theta), 0,  -sin(theta), cos(theta), 0,  0, 0, 1),  # C6 about z
        SMatrix{3,3,Float64}(1, 0, 0,  0, -1, 0,  0, 0, -1),                                       # C2 about x
        SMatrix{3,3,Float64}(-1, 0, 0,  0, -1, 0,  0, 0, -1),                                       # inversion
    ]
    return Crystal(primitives, PointGroup(generators, primitives))
end

###
### Planar lattices and operations, 2D
###
#
# Primitives and Cartesian operations of the planar crystallographic groups. `c4_planar`
# alone closes to C4 (order 4) and with `mirror_x_planar` to 4mm (order 8), both on the
# square lattice; `c6_planar` with `mirror_x_planar` closes to 6mm (order 12) on the planar
# hexagonal lattice. That planar hexagonal group is *not* the one `hexagonal_crystal()`
# carries: the latter is the 3D group 6/mmm, of order 24.

square_primitives() = SMatrix{2,2,Float64}(1, 0, 0, 1)
planar_hexagonal_primitives() = SMatrix{2,2,Float64}(1.0, 0.0, -0.5, sqrt(3) / 2)

c4_planar() = SMatrix{2,2,Float64}(0, 1, -1, 0)                                           # quarter turn
c6_planar() = SMatrix{2,2,Float64}(cos(pi / 3), sin(pi / 3), -sin(pi / 3), cos(pi / 3))   # sixth turn
mirror_x_planar() = SMatrix{2,2,Float64}(1, 0, 0, -1)                                     # reflection across x

###
### Test-only convex-polytope volume (H-rep + enumerated vertices)
###
#
# Not part of the domain source: a triangulation-based exact volume used to check the
# unfolding invariant volume(IBZ) == volume(BZ) / order(point_group). The polytope must
# be bounded (the BZ and IBZ always are).

function polytope_volume(poly::ConvexPolytope{3})
    verts = Ludwig.vertices(poly)
    apex = sum(verts) / length(verts)
    scale = maximum(abs(f.offset) for f in poly.faces; init = 1.0)
    tolerance = 1e-7 * max(scale, 1.0)
    total = 0.0
    for face in poly.faces
        on_face = filter(v -> abs(dot(face.normal, v) - face.offset) <= tolerance, verts)
        length(on_face) < 3 && continue
        face_centroid = sum(on_face) / length(on_face)
        reference = abs(face.normal[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
        in_plane_u = normalize(cross(face.normal, reference))
        in_plane_w = cross(face.normal, in_plane_u)
        angle(v) = atan(dot(v - face_centroid, in_plane_w), dot(v - face_centroid, in_plane_u))
        sort!(on_face; by = angle)
        for i in 2:length(on_face)-1
            edge_a = on_face[1] - apex
            edge_b = on_face[i] - apex
            edge_c = on_face[i+1] - apex
            total += abs(dot(edge_a, cross(edge_b, edge_c))) / 6
        end
    end
    return total
end

###
### Full-BZ grid scaffolding
###

# Grid-index permutation realizing operation `a` on a canonically laid out full-BZ grid:
# positional block `g` carries operation `g`, so `a` permutes blocks by the Cayley table and
# fixes each node's representative. Test scaffolding for building point-group-invariant
# operators by orbit averaging.
grid_permutation(multiplication_table::Matrix{Int}, a::Int, N_ibz::Int) =
    [(multiplication_table[a, (i - 1) ÷ N_ibz + 1] - 1) * N_ibz + (i - 1) % N_ibz + 1
     for i in 1:size(multiplication_table, 1) * N_ibz]
