module Ludwig

include("constants.jl")

using LinearAlgebra
using StaticArrays
using DataStructures
using ForwardDiff
using Interpolations

# Both ForwardDiff and Interpolations export `gradient`; bind unqualified usage to ForwardDiff.
# Interpolations.gradient is used explicitly by qualified name in Integration.jl.
import ForwardDiff: gradient, derivative

include("utilities.jl")
include("geometry_utilities.jl")
include("groups.jl")
include("lattices.jl")
include("marching_squares.jl")
include("bands.jl")
include("fs_mesh.jl")
include("integration.jl")
include("properties.jl")

###
### Exports
###

# GeometryUtilities
export in_polygon

# Groups
export Group, PermutationGroup, PermutationGroupElement
export get_cyclic_group, get_dihedral_group
export get_matrix_representation, order, is_identity

# Lattices
export AbstractLattice, Lattice, NoLattice, primitives, reciprocal_lattice_vectors
export point_group, lattice_type, get_bz, get_ibz, map_to_bz

# MarchingSquares
export Isoline, IsolineBundle, contour_intersection, get_bounding_box

# FSMesh
export Patch, VirtualPatch, AbstractPatch, energy, momentum, velocity, band, area
export HamiltonianBand, InterpolatedBand, IBZInterpolatedBand, band_velocity
export Mesh, patches, corners, corner_indices
export mesh_region, ibz_mesh, bz_mesh, isotropic_mesh
export BZSymmetryMap, bz_symmetry_map

# Integration
export electron_electron, electron_impurity, electron_phonon

# Utilities
export f0, symmetrize, enforce_particle_conservation!
export fill_from_ibz!, ibz_matvec!, diagonalize_ibz

# Properties
export inner_product
export electrical_conductivity, longitudinal_electrical_conductivity
export thermal_conductivity, thermoelectric_conductivity, peltier_tensor
export ηB1g, ηB2g
export σ_lifetime, η_lifetime

###
### Makie extension (loaded when Makie is available)
###

"""
    plot_mesh!(ax, mesh; color, colormap, colorrange, strokecolor, strokewidth)

Plot the Fermi surface `mesh` as a polygon collection into an existing `Makie.Axis`.

Each patch is drawn as a quadrilateral colored by its energy by default.
Requires `Makie` (or any backend such as `CairoMakie`) to be loaded.

# Arguments
- `ax`: a `Makie.Axis`.
- `mesh::Mesh`: the Fermi surface mesh to plot.

# Keyword Arguments
- `color`: per-patch colors; defaults to `energy.(mesh.patches)`.
- `colormap=:viridis`: colormap passed to `poly!`.
- `colorrange`: `(lo, hi)` colormap limits; defaults to `(-emax, emax)` where
  `emax = maximum(abs, energies)`.
- `strokecolor=:black`: patch edge color.
- `strokewidth=0.3`: patch edge width in scene units.

# Returns
The `PolyPlot` returned by the underlying `poly!` call.

See also [`plot_mesh`](@ref).
"""
function plot_mesh! end

"""
    plot_mesh(mesh; color, colormap, colorrange, strokecolor, strokewidth, axis, figure)

Create a `Figure` containing a plot of the Fermi surface `mesh`.

Requires `Makie` (or any backend such as `CairoMakie`) to be loaded.

# Arguments
- `mesh::Mesh`: the Fermi surface mesh to plot.

# Keyword Arguments
- `color`: per-patch colors; defaults to `energy.(mesh.patches)`.
- `colormap=:viridis`: colormap passed to `poly!`.
- `colorrange`: `(lo, hi)` colormap limits; defaults to `(-emax, emax)` where
  `emax = maximum(abs, energies)`.
- `strokecolor=:black`: patch edge color.
- `strokewidth=0.3`: patch edge width in scene units.
- `axis=(;)`: keyword arguments forwarded to `Makie.Axis`.
- `figure=(;)`: keyword arguments forwarded to `Makie.Figure`.

# Returns
A `Makie.FigureAxisPlot`.

See also [`plot_mesh!`](@ref).
"""
function plot_mesh end

export plot_mesh!, plot_mesh

end # module Ludwig
