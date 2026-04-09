module Ludwig

include("constants.jl")

using LinearAlgebra
using StaticArrays
using DataStructures
using ForwardDiff
using Interpolations
using IterativeSolvers

# Both ForwardDiff and Interpolations export `gradient`; bind unqualified usage to ForwardDiff.
# Interpolations.gradient is used explicitly by qualified name in Integration.jl.
import ForwardDiff: gradient, derivative

include("utilities.jl")
include("GeometryUtilities.jl")
include("Groups.jl")
include("Lattices.jl")
include("MarchingSquares.jl")
include("FSMesh.jl")
include("Integration.jl")

###
### Exports
###

# Utilities
export f0, symmetrize

# GeometryUtilities
export in_polygon, diameter, signed_area, intersection, param_intersection
export perpendicular_bisector_intersection, poly_area

# Groups
export Group, PermutationGroup, PermutationGroupElement, GroupElement
export get_cyclic_group, get_dihedral_group, get_symmetric_group, get_table, inverse
export get_matrix_representation, order, is_identity

# Lattices
export AbstractLattice, Lattice, NoLattice, primitives, reciprocal_lattice_vectors
export point_group, lattice_type, get_bz, get_ibz, map_to_bz

# MarchingSquares
export Isoline, IsolineBundle, contour_intersection, get_bounding_box

# FSMesh
export Patch, VirtualPatch, AbstractPatch, energy, momentum, velocity
export Mesh, patches, corners, corner_indices
export mesh_region, ibz_mesh, bz_mesh, circular_fs_mesh
export BZSymmetryMap, bz_symmetry_map

# Integration
export electron_electron, electron_impurity, electron_phonon
export fill_from_ibz!, ibz_matvec!, diagonalize_ibz

include("properties.jl")

end # module Ludwig
