export f0, map_to_first_bz, symmetrize

"""
        f0(E, T)

Return the value of the Fermi-Dirac distribution for energy `E` and temperature `T`.

```math
    f^{(0)}(\\varepsilon) = \\frac{1}{1 + e^{\\varepsilon/k_B T}}
```
"""
f0(E::Float64, T::Float64) = 1 / (exp(E/T) + 1)

"""
    map_to_first_bz(k)
Map a vector `k` to the ``d``-dimensional centered unit cube where ``d`` is the dimension of `k`.
"""
map_to_first_bz(k) = mod.(k .+ 0.5, 1.0) .- 0.5

"""
        symmetrize(L, dV, E, T)

"""
function symmetrize(L, dV, E, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    D = diagm(dV .* fd .* (1 .- fd))

    return 0.5 * (L + inv(D) * L' * D)
end