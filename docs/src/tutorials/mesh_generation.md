# Generating and Validating a Fermi Surface Mesh

This tutorial walks through generating a Fermi-surface mesh for the nearest-neighbor tight-binding (NNTB) model on a square lattice and validating that the mesh is well-formed before using it in a transport calculation.

## The Model

The NNTB dispersion on a square lattice with hopping $t$ and chemical potential $\mu$ is

```math
\varepsilon(\mathbf{k}) = -2t\bigl(\cos(2\pi k_x) + \cos(2\pi k_y)\bigr) - \mu,
```

where $k_x, k_y$ are measured in units of the inverse lattice constant.  We use $t = 1\,\text{eV}$ and $\mu = -1\,\text{eV}$, placing the Fermi surface well away from the van Hove singularities at $\pm 2t$.

## Setting Up

```julia
using Ludwig
using Ludwig.FSMesh
using Ludwig.Lattices
using LinearAlgebra

# Square lattice with unit lattice constant
l = Lattice([1.0 0.0; 0.0 1.0])

# NNTB dispersion (energies in eV, momenta in units of a^{-1})
t = 1.0   # eV
μ = -1.0  # eV
ε(k) = -2t * (cos(2π * k[1]) + cos(2π * k[2])) - μ

# Temperature (in eV, same units as the Hamiltonian)
T = 0.025  # ≈ 290 K
```

## Generating an IBZ Mesh

The square lattice has D₄ symmetry, so only 1/8 of the Brillouin zone (the IBZ) needs to be sampled explicitly. The remaining patches are recovered by applying symmetry operations.

```julia
n_levels = 7   # number of energy contour boundaries (n_levels - 1 patches in energy)
n_cuts   = 12  # number of angular cuts along the Fermi surface (n_cuts - 1 patches per sector)

mesh = ibz_mesh(l, [ε], T, n_levels, n_cuts)
grid = mesh.patches

println("Number of patches: ", length(grid))
```

The mesh spans an energy window of $\pm \alpha T$ around the Fermi surface (default $\alpha = 6$), sampled at `n_levels - 1` = 6 energy shells and `n_cuts - 1` = 11 angular segments per IBZ sector.

## Inspecting Patches

Each `Patch` stores the representative momentum, energy, group velocity, and integration weights at that point.  Access them via the accessor functions or struct fields:

```julia
p = grid[1]

println("Energy:   ", energy(p), " eV")
println("Momentum: ", momentum(p))
println("Velocity: ", velocity(p), " eV·a / 2π")
println("Patch area dV: ", p.dV)
println("Energy width de: ", p.de, " eV")
```

## Validating the Mesh

### 1. Energy bounds

All patch energies should lie within $\pm \alpha T$ of the Fermi energy ($\varepsilon = 0$ in our convention):

```julia
α = 6.0
E = energy.(grid)
@assert all(abs.(E) .< α * T) "Some patches lie outside the Fermi tube!"
println("Energy range: [", minimum(E), ", ", maximum(E), "] eV")
```

### 2. Nonzero velocities

Every patch on the Fermi surface should have a nonzero group velocity:

```julia
speeds = norm.(velocity.(grid))
@assert all(speeds .> 0) "Zero-velocity patch found!"
println("Speed range: [", minimum(speeds), ", ", maximum(speeds), "] eV·a / 2π")
```

### 3. Patch area coverage

The sum of patch areas `dV` is compared against an independent fine-grid estimate of the Fermi tube area. For the square lattice the IBZ is the triangle $k_x \in [0, \tfrac{1}{2}]$, $k_y \in [0, k_x]$; the estimate counts grid cells inside both this region and the energy window $|\varepsilon(\mathbf{k})| < \alpha T$:

```julia
total_dV = sum(p.dV for p in grid)
Ng  = 10000
dk  = 0.5 / Ng
area_est = 0.0
for i in 0:(Ng - 1), j in 0:(Ng - 1)
    kx = (i + 0.5) * dk
    ky = (j + 0.5) * dk
    ky > kx && continue
    abs(ε([kx, ky])) < α * T && (area_est += dk^2)
end
@assert isapprox(total_dV, area_est; rtol = 0.05) "Patch area deviates >5% from grid estimate!"
println("Σ dV: ", total_dV, ", tube area estimate: ", area_est)
```

A discrepancy larger than 5% indicates that the patches do not correctly tile the Fermi tube, likely due to an incorrect energy window or mesh generation parameters.

## Visualizing the Patches

Visualizing the mesh is a good sanity check that patches cover the Fermi surface uniformly.

!!! note "Optional dependency"
    The code below requires [CairoMakie](https://github.com/MakieOrg/Makie.jl),
    [GeometryBasics](https://github.com/JuliaGeometry/GeometryBasics.jl), and
    [LaTeXStrings](https://github.com/JuliaStrings/LaTeXStrings.jl), which are not
    dependencies of Ludwig.jl and must be installed separately.

```julia
using CairoMakie, GeometryBasics, LaTeXStrings

N = length(grid)
E = energy.(grid)

# Build one polygon per patch from its four corner points
polys = [Polygon([Point2f(mesh.corners[j]) for j in mesh.corner_inds[i]]) for i in 1:N]

fig = Figure()
ax  = Axis(fig[1, 1];
    aspect = DataAspect(),
    xlabel = L"k_x\ (2\pi/a_x)",
    ylabel = L"k_y\ (2\pi/a_y)",
    xgridvisible = false,
    ygridvisible = false,
)
pl  = poly!(ax, polys; color = E, colormap = :viridis)
ibz_verts = get_ibz(l)
lines!(ax, Point2f.(vcat(ibz_verts, [ibz_verts[1]])); color = :black, linewidth = 1.5)
Colorbar(fig[1, 2], pl; label = "Energy (eV)")
display(fig)
```

The color should vary smoothly from negative (inner contour) to positive (outer contour) energy, patches should tile the Fermi surface without gaps, and the black outline shows the IBZ boundary.

## Full Brillouin Zone Mesh

To sample the entire BZ (needed for the collision matrix), use `bz_mesh`, which applies all D₄ symmetry operations to the IBZ mesh automatically:

```julia
bz = bz_mesh(l, [ε], T, n_levels, n_cuts)
println("IBZ patches: ", length(mesh.patches))
println("BZ patches:  ", length(bz.patches), "  (should be 8×)")
@assert length(bz.patches) == 8 * length(mesh.patches) "BZ patch count is not 8× the IBZ patch count!"
```

For the square lattice the BZ mesh is 8× larger than the IBZ mesh. The [A Simple Transport Calculation](@ref) tutorial shows how to use the BZ mesh to compute transport coefficients.

## Running the Validation Script

All of the checks above are collected in `scripts/validate_mesh.jl`. Run it directly from the package root:

```bash
julia --project scripts/validate_mesh.jl          # validation only
julia --project scripts/validate_mesh.jl --plot   # validation + mesh plot
```
