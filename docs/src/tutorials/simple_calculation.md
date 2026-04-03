# A Simple Transport Calculation

This tutorial computes the electrical conductivity of the nearest-neighbor tight-binding (NNTB) model on a square lattice using electron-electron scattering, continuing from [Generating and Validating a Fermi Surface Mesh](@ref).

## Setup

```julia
using Ludwig
using Ludwig.FSMesh
using Ludwig.Lattices
using Ludwig.Utilities
using Ludwig.Integration
using LinearAlgebra

l = Lattice([1.0 0.0; 0.0 1.0])

t = 1.0   # eV
μ = -1.0  # eV
ε(k) = -2t * (cos(2π * k[1]) + cos(2π * k[2])) - μ

T = 0.025  # eV
```

## Generating the BZ Mesh

The collision matrix requires sampling the full Brillouin zone:

```julia
n_levels = 4
n_cuts   = 10

bz = bz_mesh(l, [ε], T, n_levels, n_cuts)
grid = bz.patches
N    = length(grid)

println("Total BZ patches: ", N)
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

Loop over all pairs `(i, j)` and accumulate the result into an `N×N` matrix. The convenience overload that accepts a `Lattice` object handles the reciprocal lattice vectors and BZ boundary internally:

```julia
L = zeros(N, N)

for i in 1:N
    for j in 1:N
        L[i, j] = electron_electron(grid, f0s, i, j, [ε], T, Weff_squared, l)
    end
end
```

!!! tip "Parallelism"
    For larger meshes, use `Threads.@threads for i in 1:N` to parallelize the outer loop.
    Each row is independent, so no synchronization is needed.

## Imposing and Checking Particle Conservation

The numerical integration of `electron_electron` does not guarantee that each row of `L` sums to exactly zero. Particle conservation is enforced explicitly by overwriting each diagonal entry with the negative sum of the rest of its row:

```julia
for i in 1:N
    L[i, i] -= sum(L[i, :])
end
```

After this step, `L * ones(N)` should be zero to machine precision:

```julia
residual = maximum(abs.(L * ones(N))) / maximum(abs.(diag(L)))
println("Max |L * 1| / max |L_ii|: ", residual)   # should be ≈ 0
```

A residual much larger than machine epsilon indicates a bug in the scattering vertex or the mesh.

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
