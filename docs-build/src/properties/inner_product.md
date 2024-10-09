# Inner Product

The linearized Boltzmann collision operator computed by `electron_electron()` is not Hermitian. However, we can define the auxiliary matrix
```math
    L^\text{(s)}_{ij} = f^{(0)}_i (1 - f^{(0)}_i) \frac{dV_i} \mathbf{L}_{ij}
```
which is symmetric. By defining the inner product as
```math
    \langle a | b \rangle \equiv \frac{1}{V}\sum_\mathbf{k} f^{(0)}(\epsilon_\mathbf{k}) (1 - f^{(0)}(\epsilon_\mathbf{k})) a^_\k b_\k
```
then the matrix computed by `electron_electron()` is Hermitian with respect to the above inner product.


@docs Ludwig.inner_product
