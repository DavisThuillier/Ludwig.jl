# Collision Operator

In Ludwig.jl, scattering rates are taken to be given by Fermi's golden rule, with an antisymmetrized interaction vertex:
```math
    \Gamma(\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3, \mathbf{k}_4) = \frac{1}{2} W^2(\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3, \mathbf{k}_4) \frac{2\pi}{\hbar}\delta(\varepsilon(\mathbf{k}_1) + \varepsilon(\mathbf{k}_2) - \varepsilon(\mathbf{k}_3) - \varepsilon(\mathbf{k}_4))
```
where 
```math
    W^2(\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3, \mathbf{k}_4) = | \langle \mathbf{k}_1 \mathbf{k_2} | \hat{W}| \mathbf{k_3} \mathbf{k_4} \rangle  - \langle \mathbf{k}_1 \mathbf{k_2} | \hat{W}| \mathbf{k_4} \mathbf{k_3} \rangle |^2
```

In the above we have ignored spin degrees of freedom. In Ludwig.jl we assume the following constraints:
1. The equilibrium distribution is not spin-polarized
2. Only the spin-summed fermion population is relevant
Under these constraints the we can write down an effective, spin-summed, antisymmetrized scattering vertex
```math
    W_\text{eff}^2(\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3, \mathbf{k}_4) = W_{\uparrow\uparrow\uparrow\uparrow}^2(\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3, \mathbf{k}_4) + 2 W_{\uparrow\downarrow\downarrow\uparrow}^2(\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3, \mathbf{k}_4)
```
To allow for flexibility in scattering models, a user-defined effective scattering vertex of the form `Weff_squared(p1::AbstractPatch, p2::AbstractPatch, p3::AbstractPatch, p4::AbstractPatch; kwargs...)` must be supplied. For a vertex depending only on the band index and momentum differences between patches, all required information is already contained in the fields of each `AbstractPatch`, making the calculation efficient. If the scattering vertex depends on some function value defined on the patches (for example the orbital mixing character at the sampled point), passing an array of values calulated ahead of time via a keyword argument prevents needed to repeatedly evaluate a costly function each time the vertex function is invoked. All keyword arguments passed to `electron_electron` will be bypassed to the user's `Weff_squared` function.

The linearized collision integral is given by
```math
\begin{aligned}
\mathbf{L}_{ij} = \frac{1}{d A_i} \frac{2\pi / \hbar}{1 - f^{(0)}_i} 
\frac{1}{(2\pi)^6}   &\frac12 \sum_{m}  \left( W^2_{eff}(p_i,p_j,p_m,p_{i+j-m})  f^{(0)}_j (1 - f^{(0)}_m) \mathcal{K}_{ijm}\right.\\
&\left.- ( W^2_{eff}(p_i,p_m,p_j,p_{i+m-j}) + W^2_{eff}(p_i,p_m,p_{i+m-j},p_j) ) f^{(0)}_m(1 - f^{(0)}_j) \mathcal{K}_{imj} \right)
\end{aligned}
```
where
```math
    \mathcal{K}_{ijm} = \int_i d^2 \mathbf{k}_i \int_j d^2 \mathbf{k}_j \int_m d^2 \mathbf{k}_m (1 - f^{(0)}(\mathbf{k}_i + \mathbf{k}_j - \mathbf{k}_m)) \delta(\varepsilon_i + \varepsilon_j - \varepsilon_m - \varepsilon(\mathbf{k}_i + \mathbf{k}_j - \mathbf{k}_m)).
```

```@docs
Ludwig.electron_electron(grid::Vector{Patch}, f0s::Vector{Float64}, i::Int, j::Int, bands, T::Real, Weff_squared, rlv, bz; umklapp = true, kwargs...)
```
