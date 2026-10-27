# Energy and Momentum Conserving Volume

The dominant cost of assembling the electron-electron collision matrix is the inner phase-space integral ``\mathcal{K}_{ijm}``.

## Phase-space volume to hyperplane slice

Each patch ``\mathcal{P}_i`` is parameterized by an energy coordinate ``E`` and a tangential coordinate ``s`` chosen so that the patch occupies ``[-1, 1]^2`` in those rescaled variables. Three patches contribute to ``\mathcal{K}_{ijm}``, so the product domain is the 6D cube ``[-1, 1]^6``. The kernel ``\mathcal{K}_{ijm}`` retains a Dirac delta enforcing energy conservation,

```math
\varepsilon_a + \varepsilon_b - \varepsilon_c - \varepsilon(\mathbf{k}_a + \mathbf{k}_b - \mathbf{k}_c) = 0,
```

so on the rescaled cube the integrand is supported on a 5-dimensional hyperplane.

Linearizing the energy mismatch around the patch centers gives a single linear constraint

```math
u \cdot x = -\delta, \qquad x \in [-1, 1]^6,
```

where ``\delta = \varepsilon_a + \varepsilon_b - \varepsilon_c - \varepsilon_{abc}`` is the energy mismatch at the patch centers and ``u \in \mathbb{R}^6`` is the energy-coordinate gradient assembled by `Ludwig.fill_kernel_vectors!` from the patch widths ``\Delta\varepsilon_i``, the group velocity ``v``, and the patch Jacobians ``J_i^{-1}``. After integrating out the delta, the remaining factor of the kernel is

```math
\mathcal{K}_{ijm} \;\propto\; \frac{1}{\|u\|} \cdot \mathrm{vol}_5\!\left( \{\, x \in [-1, 1]^6 : u \cdot x = -\delta\,\}\right).
```

The slice volume is approximated by a hyperspherical approximation ``\rho_5^{5/2}\, \|u\|^{-1}``, where ``\rho_5`` is the 5D sphere radius implied by the energy-conservation constraint.
