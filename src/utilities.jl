"""
    f0(E, T)

Return the value of the Fermi-Dirac distribution for energy `E` and temperature `T`.

`E` and `T` must share units (the package convention is eV for both, with ``k_B = 1``).
"""
f0(E, T) = 1 / (exp(E/T) + 1)

"""
    enforce_particle_conservation!(L::AbstractMatrix)

Overwrite each diagonal entry of `L` with the negative sum of the off-diagonal entries in
its row, so that ``\\sum_j L_{ij} = 0`` exactly for every `i`.
"""
function enforce_particle_conservation!(L::AbstractMatrix)
    n = size(L, 1)
    for i in 1:n
        L[i, i] -= sum(view(L, i, :))
    end
    return L
end

"""
    symmetrize(L, dV, E, T)

Return the symmetrization of `L` under the Fermi-window inner product
``\\langle a | b \\rangle = \\int d^2\\mathbf{k}\\, (-\\partial f^{(0)}/\\partial \\varepsilon)\\, a^*(\\mathbf{k})\\, b(\\mathbf{k})``.

Concretely, ``L_\\text{sym} = (L + D^{-1} L^\\top D)/2`` where ``D = \\mathrm{diag}(\\Delta V \\cdot f^{(0)}(1 - f^{(0)}))``.
Use this to enforce detailed balance on a numerically constructed collision operator.
"""
function symmetrize(L, dV, E, T)
    fd = f0.(E, T) # Fermi dirac on grid points
    D = diagm(dV .* fd .* (1 .- fd))

    return 0.5 * (L + inv(D) * L' * D)
end
