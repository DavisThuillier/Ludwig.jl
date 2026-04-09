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

"""
    fill_from_ibz!(L::AbstractMatrix, symmetry_map::BZSymmetryMap)

Given that the rows `symmetry_map.ibz_inds` of `L` have already been populated (e.g. via
[`electron_electron`](@ref) for each `i ∈ symmetry_map.ibz_inds`), fill all remaining rows
using the point-group invariance `L[O*i, O*j] = L[i, j]`.
"""
function fill_from_ibz!(L::AbstractMatrix, symmetry_map)
    for (i_bz, i_ibz) in symmetry_map.ibz_preimage
        g = symmetry_map.ibz_g_idx[i_bz]
        L[i_bz, :] = L[i_ibz, symmetry_map.g_inv_perms[g]]
    end
    return nothing
end

"""
    ibz_matvec!(y, x, L::AbstractMatrix, sym::BZSymmetryMap) -> y

Compute `y = L * x` in-place using only the IBZ rows of `L` (those indexed by
`sym.ibz_inds`). Non-IBZ rows are reconstructed on the fly via the point-group
invariance `L[O*i, O*j] = L[i, j]`, without calling [`fill_from_ibz!`](@ref).
"""
function ibz_matvec!(y::AbstractVector, x::AbstractVector, L::AbstractMatrix, sym)
    for i_ibz in sym.ibz_inds
        y[i_ibz] = dot(view(L, i_ibz, :), x)
    end
    for (i_bz, i_ibz) in sym.ibz_preimage
        g = sym.ibz_g_idx[i_bz]
        y[i_bz] = dot(view(L, i_ibz, :), view(x, sym.g_perms[g]))
    end
    return y
end

"""
    diagonalize_ibz(L::AbstractMatrix, sym::BZSymmetryMap) -> LinearAlgebra.Eigen

Diagonalize the full N×N collision operator `L` without calling [`fill_from_ibz!`](@ref).
Only the IBZ rows of `L` (those indexed by `sym.ibz_inds`) need to be populated.

Internally reconstructs the full matrix column-by-column via [`ibz_matvec!`](@ref) applied
to each standard basis vector, then returns `eigen(L_full)`. `L` is not mutated.

# Example
```julia
L_sym = zeros(N, N)
for i in sym.ibz_inds, j in 1:N
    L_sym[i, j] = electron_electron(grid, f0s, i, j, bands, T, Weff_squared, l)
end
result = diagonalize_ibz(L_sym, sym)
vals, vecs = result.values, result.vectors
```
"""
function diagonalize_ibz(L::AbstractMatrix, sym)
    N      = size(L, 1)
    L_full = similar(L)
    e_j    = zeros(eltype(L), N)
    col    = Vector{eltype(L)}(undef, N)

    for j in 1:N
        e_j[j] = one(eltype(L))
        ibz_matvec!(col, e_j, L, sym)
        L_full[:, j] .= col
        e_j[j] = zero(eltype(L))
    end

    return eigen(L_full)
end
