# Ludwig.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://davisthuillier.github.io/Ludwig.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://davisthuillier.github.io/Ludwig.jl/dev)

Ludwig.jl computes the linearized Boltzmann collision operator for two-dimensional and quasi-two-dimensional electron systems, including electron-electron, electron-impurity, and electron-phonon scattering. From the resulting collision matrix it derives transport coefficients — electrical and thermal conductivity, thermopower, viscosity, and effective scattering lifetimes — using a linear-solver back end of the user's choice.

The package is unit-agnostic but uses $\hbar = k_B = 1$ internally; the convention throughout the public API is to express energies and temperatures in eV and momenta in inverse lattice constants.

## Installation

Ludwig.jl is registered in the Julia General registry:

```julia
julia> ]
pkg> add Ludwig
```

For the development version:

```julia
pkg> add https://github.com/DavisThuillier/Ludwig.jl
```

To plot Fermi-surface meshes, also load `CairoMakie` (or another Makie back end); Ludwig provides a Makie extension that activates automatically.

## Quick start

The example below reproduces the simple electrical-conductivity calculation in `docs/src/tutorials/simple_calculation.md` for the nearest-neighbor tight-binding model on a square lattice with a Hubbard interaction.

```julia
using Ludwig
using LinearAlgebra

# 1. Lattice and dispersion
l = Lattice([1.0 0.0; 0.0 1.0])           # square lattice, a = 1
t  = 1.0     # hopping (eV)
μ  = -1.0    # chemical potential (eV)
ε(k) = -2t * (cos(k[1]) + cos(k[2])) - μ
T  = 0.025   # temperature (eV)

# 2. Fermi-surface mesh over the full BZ
Δε    = 0.1   # energy spacing of boundary contours
n_arc = 10    # angular patches per sheet
mesh  = bz_mesh(l, ε, T, Δε, n_arc)
grid  = patches(mesh)
N     = length(grid)

# 3. Symmetry map: only IBZ rows need to be filled
sym = bz_symmetry_map(grid, l)

# 4. Effective scattering vertex (Hubbard: |U|² regardless of momenta)
U = 1.0
Weff_squared(p1, p2, p3, p4; kwargs...) = U^2

# 5. Build the collision matrix on the IBZ, expand by symmetry
f0s = f0.(energy.(grid), T)
L   = zeros(N, N)
Threads.@threads for i in sym.ibz_inds
    for j in 1:N
        L[i, j] = electron_electron(grid, f0s, i, j, [ε], T, Weff_squared, l)
    end
end
fill_from_ibz!(L, sym)

# 6. Enforce particle conservation and detailed balance
dV = [p.dV for p in grid]
E  = energy.(grid)
v  = velocity.(grid)
enforce_particle_conservation!(L)
L = symmetrize(L, dV, E, T)

# 7. Transport coefficient
σ = electrical_conductivity(L, v, E, dV, T)
@show real(σ[1, 1])      # σ_xx in Siemens (up to lattice-constant rescaling)
```

The full version — including residual-conservation checks, the IBZ-only `diagonalize_ibz` path, and the dimensional argument — is in [`docs/src/tutorials/simple_calculation.md`](docs/src/tutorials/simple_calculation.md) and in `scripts/simple_calculation.jl`.

## Features

- **Brillouin-zone geometry.** `Lattice` infers the 2D Bravais lattice type from the primitive vectors and assigns the corresponding point group (C₂, D₂, D₄, or D₆); `get_bz` and `get_ibz` return the first BZ and irreducible BZ as polygons.
- **Fermi-surface meshing.** `ibz_mesh`, `bz_mesh`, and `mesh_region` build patch-wise meshes via marching-squares contouring with arclength-uniform resampling, multi-sheet handling, and recursive corner re-meshing. `isotropic_mesh` is a specialised path for the free electron gas.
- **Symmetry-aware integration.** `BZSymmetryMap` records the point-group action on the BZ mesh; `fill_from_ibz!`, `ibz_matvec!`, and `diagonalize_ibz` let you compute only the IBZ rows of the collision operator and recover the full matrix — or its action — on demand.
- **Scattering kernels.** `electron_electron` integrates the 5D phase-space kernel exactly via the Pournin polytope-volume formula (or via a hyperspherical approximation). `electron_phonon` and `electron_impurity` cover the standard quasi-elastic and elastic channels.
- **Transport properties.** `electrical_conductivity`, `thermal_conductivity`, `thermoelectric_conductivity`, `peltier_tensor`, `ηB1g`, `ηB2g`, `σ_lifetime`, and `η_lifetime` evaluate weighted inner products of the form $\langle a | L^{-1} | b \rangle$ — with the linear solver pluggable (dense LU by default, but any iterative or matrix-free back end works).

## Citing Ludwig.jl

If you use Ludwig.jl in research, please cite it. A `CITATION.bib` entry will be added with the v0.3.0 release; in the interim the package can be cited via the GitHub repository:

```bibtex
@software{LudwigJl,
  author  = {Thuillier, Davis and Scaffidi, Thomas},
  title   = {{Ludwig.jl}},
  url     = {https://github.com/DavisThuillier/Ludwig.jl},
  version = {0.3.0},
  year    = {2026},
}
```

## License

Ludwig.jl is released under the [MIT license](LICENSE.md).

## Authors

[Davis Thuillier](https://github.com/DavisThuillier) and Thomas Scaffidi.
