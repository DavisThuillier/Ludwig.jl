# Collision Operator

## Single Band
For a single band with a momentum independent scattering potential ``V(\mathbf{q}) = V``,
```math
    \mathbf{L}_{ij} = \frac{1}{d A_i} \frac{2\pi}{1 - f^{(0)}_i} |V|^2 \frac{1}{(2\pi)^6} \sum_{m} (f^{(0)}_j (1 - f^{(0)}_m) \mathcal{K}_{ijm} - 2f^{(0)}_m (1 - f^{(0)}_j)\mathcal{K}_{imj}) 
```
where
```math
    \mathcal{K}_{ijm} = \int_i d^2 \mathbf{k}_i \int_j d^2 \mathbf{k}_j \int_m d^2 \mathbf{k}_m (1 - f^{(0)}(\mathbf{k}_i + \mathbf{k}_j - \mathbf{k}_m)) \delta(\varepsilon_i + \varepsilon_j - \varepsilon_m - \varepsilon(\mathbf{k}_i + \mathbf{k}_j - \mathbf{k}_m)).
```

```@docs
Ludwig.Kabc!
```

## Improved Multiband
In multiband scattering, the bands consist of hybridized orbitals. To account for this, a simple model one can propose for the scattering term is
```math
    U\sum_{i} \sum_{a,b} n_{i,a} n_{i,b}
```
where ``i`` represents a site index and ``a, b`` are orbital indices. To handle this perturbation in our framework of scattering using the Born approximation, we need to evaluate this interaction term in the eigenbasis of the bare Hamiltonian.
```math
\begin{aligned}
    \sum_i n_{i,a} n_{i, b} &= \sum_{i}c^\dagger_{i,a} c_{i,a} c^\dagger_{i,b} c_{i,b}\\
    &= \frac{1}{N^2} \sum_{\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3, \mathbf{k}_4} \left( e^{i (\mathbf{k}_3 + \mathbf{k}_4 - \mathbf{k}_1 - \mathbf{k}_2) \cdot \mathbf{R}_i}\right ) c^\dagger_{\mathbf{k}_3,a} c_{\mathbf{k}_1,a} c^\dagger_{\mathbf{k}_4,b} c_{\mathbf{k}_2,b}\\
    &= \frac{1}{N^2} \sum_{\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3, \mathbf{k}_4} N \delta_{\mathbf{k}_4,\mathbf{k}_1 + \mathbf{k}_2 - \mathbf{k}_3} c^\dagger_{\mathbf{k}_3,a} c_{\mathbf{k}_1,a} c^\dagger_{\mathbf{k}_4,b} c_{\mathbf{k}_2,b}\\
    &= \frac{1}{N} \sum_{\mathbf{k}_1, \mathbf{k}_2, \mathbf{q}} c^\dagger_{\mathbf{k}_1 - \mathbf{q},a} c_{\mathbf{k}_1,a} c^\dagger_{\mathbf{k}_2 + \mathbf{q},b} c_{\mathbf{k}_2,b}\\
    &= \frac{1}{N} \sum_{\mathbf{k}_1, \mathbf{k}_2, \mathbf{q}} \sum_{\mu\nu\sigma\tau} c^\dagger_{\mathbf{k}_1 - \mathbf{q},\sigma} \left( W_{\mathbf{k}_1 - \mathbf{q}}^{a \sigma}\right)^* W_{\mathbf{k}_1}^{a \mu} c_{\mathbf{k}_1, \mu} c^\dagger_{\mathbf{k}_2 + \mathbf{q},\tau} \left( W_{\mathbf{k}_2 + \mathbf{q}}^{b \tau}\right)^* W_{\mathbf{k}_2}^{b \nu} c_{\mathbf{k}_2, \nu}
\end{aligned}
```
From the above, we can define the *interband connection* as
```math
    F_{\mathbf{k}_1, \mathbf{k}_2}^{\mu\nu} \equiv \sum_a \left(W_{\mathbf{k}_1}^{a \mu} \right)^* W_{\mathbf{k}_2}^{a \nu} = \sum_a \left(W^\dagger_{\mathbf{k}_1}\right)^{\mu a} W_{\mathbf{k}_2}^{a \nu}  = \left(W^\dagger_{\mathbf{k}_1} W_{\mathbf{k}_2} \right)^{\mu \nu } 
```
so the interaction term becomes
```math
U\sum_{i} \sum_{a,b} n_{i,a} n_{i,b} = \frac{U}{N} \sum_{\mathbf{k}_1, \mathbf{k}_2, \mathbf{q}} \sum_{\mu\nu\sigma\tau} c^\dagger_{\mathbf{k}_1 - \mathbf{q},\sigma}  F^{\sigma\mu}_{\mathbf{k}_1 - \mathbf{q}, \mathbf{k}_1}  c_{\mathbf{k}_1, \mu} c^\dagger_{\mathbf{k}_2 + \mathbf{q},\tau} F^{\tau\nu}_{\mathbf{k}_2 + \mathbf{q}, \mathbf{k}_2} c_{\mathbf{k}_2, \nu}.
```

By summing over spin index, we can define an effective transition rate ``W^2``:
```math
    W^2_{eff}(p_1,p_2,p_3,p_4) = W^2_{\uparrow\uparrow\uparrow\uparrow}(p_1,p_2,p_3,p_4) + 2 W^2_{\uparrow\downarrow\downarrow\uparrow}(p_1,p_2,p_3,p_4)
```

For the on-site multiorbital interaction model, one finds
```math
W^2_{\uparrow\uparrow\uparrow\uparrow} = | U F^{\mu_1,\mu_3}_{k_1,k_3} F^{\mu_2,\mu_4}_{k_2,k_4} - U F^{\mu_1,\mu_4}_{k_1,k_4} F^{\mu_2,\mu_3}_{k_2,k_3} |^2 
```
and
```math
W^2_{\uparrow\downarrow\downarrow\uparrow} = | - U F^{\mu_1,\mu_4}_{k_1,k_4} F^{\mu_2,\mu_3}_{k_2,k_3} |^2
```
which finally gives
```
W^2_{eff}(p_1,\mu_1,p_2,\mu_2,p_3,\mu_3,p_4,\mu_4) &= U^2 \left( |  F^{\mu_1,\mu_3}_{k_1,k_3} F^{\mu_2,\mu_4}_{k_2,k_4} -  F^{\mu_1,\mu_4}_{k_1,k_4} F^{\mu_2,\mu_3}_{k_2,k_3} |^2 + 2 |  F^{\mu_1,\mu_4}_{k_1,k_4} F^{\mu_2,\mu_3}_{k_2,k_3} |^2 \right)
```

Then, the linearized collision integral is given by
```math
    \mathbf{L}_{ij} &= \frac{1}{\d A_i} \frac{2\pi / \hbar}{1 - f^{(0)}_i} 
\frac{1}{(2\pi)^6}   &\frac12 \sum_{m}  \left( W^2_{eff}(p_i,p_j,p_m,p_{i+j-m})  f^{(0)}_j (1 - f^{(0)}_m) \mathcal{K}_{ijm}\right.\\
&\left.- ( W^2_{eff}(p_i,p_m,p_j,p_{i+m-j}) + W^2_{eff}(p_i,p_m,p_{i+m-j},p_j) ) f^{(0)}_m(1 - f^{(0)}_j) \mathcal{K}_{imj} \right)
```

```@docs
Ludwig.electron_electron
```
