# Units and SI Conversion

This page documents the unit conventions used by every function in `properties.jl`, the dimensional reasoning behind the SI prefactor each one applies, and the relationship between the bare correlators returned by the package and the physical transport observables a reader is likely to be after (open-circuit vs closed-circuit conventions).

The intent is that every prefactor in the source has a single, auditable derivation here. If a value the package returns disagrees with a literature reference, this page is the first place to check.

## Package conventions

Throughout `properties.jl`:

| quantity | unit |
|---|---|
| energy ``\varepsilon`` | eV |
| temperature ``T`` | eV (with ``k_B = 1``) |
| momentum ``\mathbf{k}`` | ``1/a`` (``a`` is the lattice constant) |
| velocity ``\mathbf{v}`` | ``a \cdot \mathrm{eV} / \hbar`` |
| patch area ``\Delta V`` | ``1/a^2`` |

The reduced Planck constant is set to ``\hbar = 1``. The inner product (see [`inner_product`](@ref)) uses the canonical Fermi-window weight, with the ``1/T`` already folded in:

```math
\langle a \,|\, b \rangle = \sum_{\mathbf{k}} \frac{f^{(0)}_{\mathbf{k}} \bigl(1 - f^{(0)}_{\mathbf{k}}\bigr)}{T} \, \Delta V_{\mathbf{k}} \, a^*_{\mathbf{k}} \, b_{\mathbf{k}}
= \sum_{\mathbf{k}} \left(- \frac{\partial f^{(0)}}{\partial \varepsilon}\right)_{\mathbf{k}} \Delta V_{\mathbf{k}} \, a^*_{\mathbf{k}} \, b_{\mathbf{k}}.
```

This weight is constructed by the internal helper `Ludwig.boltzmann_weight(E, dV, T)`. Because the temperature factor lives in the inner product itself, the prefactors below carry no further explicit ``1/T`` beyond what is intrinsic to the channel (one factor per energy gradient, etc.).

## SI conversion factors

Every transport coefficient takes the form

```math
X_{ij} = (\text{prefactor}) \times \langle \phi^{(i)} \,|\, (\hat{L}'\,)^{-1} \,|\, \phi^{(j)} \rangle
```

where ``\phi^{(i)}`` is the channel-dependent driving vector and ``\hat{L}'`` may be ``\hat{L}`` itself or ``\hat{L}`` with a frequency / momentum shift. The prefactor combines the natural-units formula with the SI conversion factor for the chosen output unit.

Each section below states:

1. **Formula** — the natural-units expression in terms of the bare correlator.
2. **Dimensions** — the unit-by-unit accounting that determines what the SI prefactor must contain.
3. **Prefactor** — the coefficient multiplied onto the inner-product result.
4. **Output unit** — the SI unit of the returned value.

The package is two-dimensional. All transport coefficients below are *sheet* quantities: ``\sigma`` has units of conductance (Siemens), ``\kappa`` has units of thermal conductance (W/K), and so on. Multiplication by an inverse length is required to obtain bulk SI units when applied to a quasi-2D material with finite layer separation.

### Electrical conductivity ``\sigma_{ij}``

[`electrical_conductivity`](@ref), [`longitudinal_electrical_conductivity`](@ref).

- **Formula**: ``\sigma_{ij}(\omega, \mathbf{q}) = 2 e^2 \, \langle v_i \,|\, (\hat{L} - i\omega + i\mathbf{q}\cdot\hat{\mathbf{v}})^{-1} \,|\, v_j \rangle.`` The single ``1/T`` of the natural-units derivation is already absorbed into the inner-product weight.
- **Dimensions**: in package units, ``\langle v | L^{-1} | v\rangle`` has units of ``\mathrm{eV}^2 \cdot a^2 / \hbar^2 \cdot [L^{-1}]``. The result reduces to Siemens.
- **Prefactor**: ``\dfrac{G_0}{2\pi}``, where ``G_0 = 2e^2/h = e^2/(\pi \hbar)``.
- **Output unit**: Siemens (sheet conductance). For a quasi-2D material, multiply by the interlayer spacing to obtain the proper measured conductance.

### Viscosity ``\eta_{B_{1g}}``, ``\eta_{B_{2g}}``

[`ηB1g`](@ref), [`ηB2g`](@ref).

- **Formula** (B1g): ``\eta_{B_{1g}} = \dfrac{1}{4} \langle D_{xx} - D_{yy} \,|\, \hat{L}^{-1} \,|\, D_{xx} - D_{yy} \rangle.``
- **Formula** (B2g): ``\eta_{B_{2g}} = \langle D_{xy} \,|\, \hat{L}^{-1} \,|\, D_{xy} \rangle.``
- **Dimensions**: deformation potentials ``D`` have units of energy (eV); the inner-product weight already supplies the ``1/T`` factor needed for the correct phase-space density.
- **Prefactor**: ``2 \hbar e_{\text{ch}}``, where ``\hbar`` ensures the result has units of action (J·s) and ``e_{\text{ch}}`` converts eV → J.
- **Output unit**: Pa·s (sheet viscosity).

### Effective lifetimes ``\tau_\sigma``, ``\tau_\eta``

[`σ_lifetime`](@ref), [`η_lifetime`](@ref).

- **Formula**: ``\tau_\sigma = \hbar \, \dfrac{\langle v_x \,|\, \hat{L}^{-1} \,|\, v_x \rangle}{\langle v_x \,|\, v_x \rangle}.`` (And similarly for ``\tau_\eta`` with the deformation potential.)
- **Dimensions**: the ratio of inner products is dimensionless × time/``\hbar`` (since ``L`` has units of 1/time in natural units); multiplication by ``\hbar`` gives seconds.
- **Prefactor**: ``\hbar``.
- **Output unit**: seconds.

## Adding or auditing a prefactor

Each prefactor in `src/properties.jl` should match exactly one entry in this page. The audit recipe is:

1. Identify the natural-units formula (above).
2. Multiply through the per-channel dimensional factors (``e``, ``\hbar``, ``T``, ``e_{\text{ch}}``).
3. Check that the resulting expression is the SI unit listed.
4. Cross-check against a known limit (Drude lifetime, Wiedemann–Franz, free-electron Seebeck, …).

If steps 1–3 disagree with the source, fix the source. If the source matches but step 4 fails, the formula in step 1 is the suspect.
