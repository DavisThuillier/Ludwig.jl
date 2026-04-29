# A Simple Transport Calculation

This tutorial computes the electrical conductivity of the nearest-neighbor tight-binding (NNTB) model on a square lattice using electron-electron scattering, continuing from [Generating and Validating a Fermi Surface Mesh](@ref).

## Setup

```julia
using Ludwig
using LinearAlgebra

l = Lattice([1.0 0.0; 0.0 1.0])

# Momenta are in units of the inverse lattice constant a^{-1}
t = 1.0   # eV
μ = -1.0  # eV
ε(k) = -2t * (cos(k[1]) + cos(k[2])) - μ

T = 0.025  # eV
```

## Generating the BZ Mesh

The collision matrix requires sampling the full Brillouin zone:

```julia
Δε    = 0.1   # energy spacing between boundary contours (eV)
n_arc = 10    # patches per arc segment
N_marching_squares = 1001
α     = 6.0

bz   = bz_mesh(l, ε, T, Δε, n_arc, N_marching_squares, α)
grid = bz.patches
N    = length(grid)

println("Total BZ patches: ", N)
```

## Building the Symmetry Map

Because the square lattice has D₄ symmetry, only 1/8 of the BZ patches are independent. A `BZSymmetryMap` records which BZ patches are related by symmetry and how to permute column indices when copying rows:

```julia
sym = bz_symmetry_map(grid, l)
println("IBZ patches: ", length(sym.ibz_inds), " (out of ", N, ")")
```

## Precomputing the Fermi-Dirac Distribution

`electron_electron` requires the Fermi-Dirac values at every patch center. Computing them once outside the loop avoids redundant evaluations:

```julia
f0s = f0.(energy.(grid), T)
```

## Defining the Scattering Vertex

Provide a user-defined function `Weff_squared(p1, p2, p3, p4; kwargs...)` that returns the effective squared scattering vertex. For a contact (Hubbard-like) interaction:

```julia
U = 1.0  # eV, interaction strength

Weff_squared(p1, p2, p3, p4; kwargs...) = U^2
```

## Building the Collision Matrix

Only the IBZ rows need to be computed explicitly. `fill_from_ibz!` then reconstructs all remaining rows from the point-group invariance ``L[O \cdot i, O \cdot j] = L[i, j]``, reducing the computational cost by a factor equal to the order of the point group (8 for the square lattice):

```julia
L = zeros(N, N)

Threads.@threads for i in sym.ibz_inds
    for j in 1:N
        L[i, j] = electron_electron(grid, f0s, i, j, [ε], T, Weff_squared, l)
    end
end

fill_from_ibz!(L, sym)
```

!!! tip "Parallelism"
    The outer loop over `sym.ibz_inds` is embarrassingly parallel. Each IBZ row is
    independent, so no synchronization is needed.

### Diagonalization Without Filling

If only the eigenvalues and eigenvectors of `L` are needed (e.g. to identify the slowest-decaying modes or check for zero modes), `diagonalize_ibz` avoids the `fill_from_ibz!` step entirely. It reconstructs the full matrix implicitly, column by column, using `ibz_matvec!`:

```julia
result = diagonalize_ibz(L, sym)
vals, vecs = result.values, result.vectors
```

This saves both the memory and time of explicitly storing the full symmetry-expanded matrix, at the cost of not having `L` available for subsequent linear solves.

## Imposing and Checking Particle Conservation

The numerical integration of `electron_electron` does not guarantee that each row of `L` sums to exactly zero. `enforce_particle_conservation!` enforces the constraint by overwriting each diagonal entry with the negative sum of the of its row:

```julia
enforce_particle_conservation!(L)
```

After this step, `L * ones(N)` should be zero to machine precision.

## Enforcing Symmetry

The raw output of `electron_electron` may have small asymmetries due to numerical integration. To enforce exact symmetry of the auxiliary matrix ``L^{(\text{s})}_{ij} = f^{(0)}_i (1 - f^{(0)}_i) \Delta V_i L_{ij}`` before inverting, use `symmetrize`:

```julia
dV = [p.dV for p in grid]
E  = energy.(grid)

L = symmetrize(L, dV, E, T)
```

## Computing the Electrical Conductivity

With the collision matrix in hand, compute the conductivity tensor:

```julia
v = velocity.(grid)   # vector of SVector{2} group velocities

σ = electrical_conductivity(L, v, E, dV, T)
println("σ_xx = ", real(σ[1,1]), " S")
println("σ_xy = ", real(σ[1,2]), " S")
```

`electrical_conductivity` returns a 2×2 complex matrix in Siemens, provided the inputs follow the unit conventions described in its docstring.  For the dimensionless calculation above (lattice constant set to 1 without specifying physical units), the result must be multiplied by appropriate factors of the lattice constant and Planck's constant to obtain SI conductivity.

## Running the Calculation Script

All of the steps above, together with particle conservation and symmetry checks, are collected in `scripts/simple_calculation.jl`. Run it from the package root:

```bash
julia --project scripts/simple_calculation.jl
```
