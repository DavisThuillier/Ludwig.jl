# Inner Product

The linearized Boltzmann collision operator computed by `electron_electron` is not Hermitian with respect to the standard inner product. However, one can define the symmetrized auxiliary matrix

```math
L^{\text{(s)}}_{ij} = f^{(0)}_i (1 - f^{(0)}_i) \, \Delta V_i \, L_{ij}
```

which is symmetric. This is equivalent to saying that ``L`` is Hermitian with respect to the weighted inner product

```math
\langle a | b \rangle \equiv \sum_{\mathbf{k}} f^{(0)}(\epsilon_{\mathbf{k}}) \bigl(1 - f^{(0)}(\epsilon_{\mathbf{k}})\bigr) \, a_{\mathbf{k}}^* \, b_{\mathbf{k}} \, \Delta V_{\mathbf{k}}.
```

Transport coefficients are expressed as matrix elements of ``L^{-1}`` under this inner product, i.e., ``\langle a | L^{-1} | b \rangle``, which are evaluated numerically by solving the linear system ``L \phi = b`` and then summing ``\sum_\mathbf{k} w_\mathbf{k} \, a_\mathbf{k} \, \phi_\mathbf{k}``.

```@docs
Ludwig.inner_product
```
