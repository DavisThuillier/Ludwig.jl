# Transport Coefficients

Transport coefficients are expressed as matrix elements of the inverse collision operator under the [weighted inner product](@ref "Inner Product"):

```math
X_{ij} = \langle \phi_i | L^{-1} | \phi_j \rangle \equiv \sum_{\mathbf{k}} f^{(0)}_{\mathbf{k}} (1 - f^{(0)}_{\mathbf{k}}) \, \Delta V_{\mathbf{k}} \, \phi^{(i)}_{\mathbf{k}} \, [L^{-1} \phi^{(j)}]_{\mathbf{k}}.
```

The driving vectors ``\phi`` differ by transport channel: ``\phi = v_i`` for charge transport, ``\phi = \varepsilon v_i`` for heat transport. Numerically, ``L^{-1} \phi`` is computed by solving the linear system ``L \psi = \phi`` via a biconjugate gradient iteration.

## Electrical Conductivity

See [`electrical_conductivity`](@ref) and [`longitudinal_electrical_conductivity`](@ref).

## Thermal Conductivity

See [`thermal_conductivity`](@ref).

## Thermoelectric Conductivity

See [`thermoelectric_conductivity`](@ref).

## Peltier Tensor

See [`peltier_tensor`](@ref).

## Viscosity

For two-dimensional materials with tetragonal symmetry, the shear viscosity decomposes into B1g and B2g irreducible representations of the D4h point group. These are computed from the corresponding deformation potentials ``D(\mathbf{k})``, which encode the coupling of the quasiparticle momentum to the relevant strain mode. See [`ηB1g`](@ref) and [`ηB2g`](@ref).

## Effective Lifetimes

See [`σ_lifetime`](@ref) and [`η_lifetime`](@ref).
