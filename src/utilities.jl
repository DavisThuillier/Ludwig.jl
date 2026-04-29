"""
    f0(E, T)

Return the value of the Fermi-Dirac distribution for energy `E` and temperature `T`.

`E` and `T` must share units (the package convention is eV for both, with ``k_B = 1``).

```math
    f^{(0)}(\\varepsilon) = \\frac{1}{1 + e^{\\varepsilon/k_B T}}
```

# Examples
```jldoctest
julia> f0(0.0, 1.0)
0.5

julia> f0(-Inf, 1.0)
1.0

julia> f0(Inf, 1.0)
0.0
```
"""
f0(E, T) = 1 / (exp(E/T) + 1)

"""
    enforce_particle_conservation!(L::AbstractMatrix)

Overwrite each diagonal entry of `L` with the negative sum of the off-diagonal entries in
its row, so that ``\\sum_j L_{ij} = 0`` exactly for every `i`.

The linearized collision operator must conserve particle number row-by-row, but numerical
integration of [`electron_electron`](@ref) (and the other scattering kernels) leaves a
small residual in each row sum. This routine absorbs that residual into the diagonal —
physically equivalent to assigning the rounding error to the scattering-out rate. Returns
the mutated matrix `L`.

# Examples
```jldoctest
julia> L = [ 0.7  -0.4  -0.5;
            -0.3   0.6  -0.6;
            -0.4  -0.5   0.8];

julia> enforce_particle_conservation!(L);

julia> L * ones(3)
3-element Vector{Float64}:
 0.0
 0.0
 0.0
```

See also [`symmetrize`](@ref).
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
