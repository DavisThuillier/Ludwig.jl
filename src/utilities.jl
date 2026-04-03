module Utilities

import LinearAlgebra: diagm
export f0, symmetrize

"""
        f0(E, T)

Return the value of the Fermi-Dirac distribution for energy `E` and temperature `T`.

```math
    f^{(0)}(\\varepsilon) = \\frac{1}{1 + e^{\\varepsilon/k_B T}}
```
"""
f0(E::Float64, T::Float64) = 1 / (exp(E/T) + 1)


"""
        symmetrize(L, dV, E, T)
        
Enforce that L is symmetric under the inner product ``\\langle a | b \\rangle = \\int d^2\\mathbf{k} \\left ( - \\frac{\\partial f^{(0)}(\\varepsilon_\\mathbf{k})}{\\partial \\varepsilon_\\mathbf{k}}\\right ) a*(\\mathbf{k}) b(\\mathbf{k})``.
"""
function symmetrize(L, dV, E, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    D = diagm(dV .* fd .* (1 .- fd))

    return 0.5 * (L + inv(D) * L' * D)
end

end
