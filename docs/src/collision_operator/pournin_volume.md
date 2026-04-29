# Pournin Volume

The dominant cost of assembling the electron-electron collision matrix is the inner phase-space integral ``\mathcal{K}_{ijm}``. With `exact = true` (the default in [`electron_electron`](@ref) and [`Ludwig.ee_kernel!`](@ref)), this integral is evaluated *exactly* — to floating-point precision — by reducing it to the volume of a hyperplane slice through a hypercube. The closed-form expression for that slice volume is given by Theorem 2.2 of Pournin (2024). This page explains how the formula maps onto the kernel and how it is implemented.

## Phase-space volume to hyperplane slice

Each patch ``\mathcal{P}_i`` is parameterized by an energy coordinate ``E`` and a tangential coordinate ``s`` chosen so that the patch occupies ``[-1, 1]^2`` in those rescaled variables. Three patches contribute to ``\mathcal{K}_{ijm}``, so the product domain is the 6D cube ``[-1, 1]^6``. The kernel ``\mathcal{K}_{ijm}`` retains a Dirac delta enforcing energy conservation,

```math
\varepsilon_a + \varepsilon_b - \varepsilon_c - \varepsilon(\mathbf{k}_a + \mathbf{k}_b - \mathbf{k}_c) = 0,
```

so on the rescaled cube the integrand is supported on a 5-dimensional hyperplane.

Linearizing the energy mismatch around the patch centers — the reason the matrix elements admit a closed-form volume rather than a quadrature — gives a single linear constraint

```math
u \cdot x = -\delta, \qquad x \in [-1, 1]^6,
```

where ``\delta = \varepsilon_a + \varepsilon_b - \varepsilon_c - \varepsilon_{abc}`` is the energy mismatch at the patch centers and ``u \in \mathbb{R}^6`` is the energy-coordinate gradient assembled by `Ludwig.fill_kernel_vectors!` from the patch widths ``\Delta\varepsilon_i``, the group velocity ``v``, and the patch Jacobians ``J_i^{-1}``. After integrating out the delta, the remaining factor of the kernel is

```math
\mathcal{K}_{ijm} \;\propto\; \frac{1}{\|u\|} \cdot \mathrm{vol}_5\!\left( \{\, x \in [-1, 1]^6 : u \cdot x = -\delta\,\}\right).
```

Computing this 5D volume *exactly*, rather than approximating it by a hypersphere of equal volume, is the role of Pournin's theorem.

## Pournin's theorem

Pournin's theorem applies on the unit cube ``[0, 1]^n``. The affine change of variables ``x = 2y - 1`` sends ``[-1, 1]^6 \to [0, 1]^6``; under this map the constraint becomes ``u \cdot y = b`` with

```math
b = \frac{\sum_i u_i - \delta}{2},
```

and the 5D surface measure picks up a factor of ``2^5 = 32``.

For a hyperplane ``u \cdot y = b`` cutting ``[0, 1]^n``, Pournin's formula expresses the ``(n-1)``-dimensional volume of the slice as a finite alternating sum over the cube vertices that lie on the same side of the hyperplane as the origin:

```math
V(u, b) \;=\; \sum_{\substack{v \in \{0, 1\}^n \\ u \cdot v \,\le\, b}} (-1)^{\sigma(v)} \frac{\|u\|\, (b - u \cdot v)^{d-1}}{(d-1)!\, \pi(u)},
```

where

- ``\sigma(v) = \sum_i v_i`` is the parity of the binary vertex ``v``,
- ``\pi(u) = \prod_{i:\, u_i \ne 0} u_i`` is the product over the nonzero components,
- ``d = \#\{i : u_i \ne 0\}`` is the effective dimension.

Coordinates with ``u_i = 0`` are unconstrained by the hyperplane and contribute a factor of one to the volume; they are dropped from the sum so the cost is ``O(2^d)`` rather than ``O(2^n)``.

Because every input to the formula — ``u``, ``\delta``, the patch widths and Jacobians — is read directly from patch metadata, no quadrature is performed. The volume returned is exact up to floating-point error.

## Putting the pieces together

Inside [`Ludwig.ee_kernel!`](@ref) the call site is

```julia
b_pournin = ((u[1] + u[2] + u[3] + u[4] + u[5] + u[6]) - δ) / 2
vol_5     = 32.0 * pournin_volume(u, b_pournin)
```

The factor of ``32`` undoes the ``[-1, 1]^6 \to [0, 1]^6`` rescaling; the patch Jacobian determinants ``\partial_E s_i = \texttt{djinv}`` and the surface-measure factor ``1/\|u\|`` are applied multiplicatively to give the final kernel value.

For an isotropic Fermi surface the same construction goes through with all of these geometric quantities replaced by their NoLattice analogues (see [`electron_electron`](@ref) with `l::NoLattice`).

## Implementation notes

The implementation in `src/integration.jl` is split between two helpers:

- [`Ludwig.pournin_volume`](@ref)`(u, b)` is the entry point. It collects the nonzero components of ``u``, normalizes them so that ``(b - u \cdot v)^{d-1}`` stays of order unity, dispatches on the effective dimension ``d`` to a `Val{d}`-specialized inner kernel, and folds the prefactor ``\|u\|^d / ((d-1)!\, \pi(u))`` back in once the alternating sum is done.

- `Ludwig.pournin_inner(u_compact, b, ::Val{D})` is a `@generated` function that emits, at compile time, a fully unrolled traversal of the ``2^D`` vertices of ``[0, 1]^D`` in *Gray-code order*. Consecutive Gray-code vertices differ in a single bit, so each step updates the running dot product ``u \cdot v`` by adding or subtracting one component of ``u`` rather than recomputing the full sum. The vertex parity, the bit to toggle at each step, and the sign of each contribution are baked in as compile-time literals; at runtime nothing remains except float arithmetic and predictable array reads.

## Exact vs. hyperspherical mode

`exact = false` replaces the slice volume by a hyperspherical approximation ``\rho_5^{5/2}\, \|u\|^{-1}``, where ``\rho_5`` is the 5D sphere radius implied by the energy-conservation constraint. The approximation is faster but loses the geometry of the cube — it under- or over-counts whenever the hyperplane is close to a face or corner of ``[-1, 1]^6``. For convergence studies and final calculations, leave the default `exact = true`.

## Reference

L. Pournin, *Hyperplane sections of unit hypercubes*, [European Journal of Combinatorics 120 (2024) 103966](https://doi.org/10.1016/j.ejc.2024.103966).

See also [`Ludwig.ee_kernel!`](@ref) and [`Ludwig.pournin_volume`](@ref).
