###
### Hermitian eigensolvers
###

"Maximum number of Jacobi sweeps in [`hermitian_eigen`](@ref) before it gives up."
const jacobi_maxsweeps::Int = 32

"""
    hermitian_eigvals(H)

Return the eigenvalues of the Hermitian matrix `H` in ascending order.

An `SMatrix` is diagonalized by cyclic Jacobi rotations of a stack copy, so nothing is
allocated and LAPACK is not called; StaticArrays would instead hand any complex or
larger-than-3×3 `SMatrix` to LAPACK as a heap copy. The sweeps stop once every
off-diagonal element is below `eps * norm(H)`, and an error is thrown if that takes more
than `jacobi_maxsweeps` (32) sweeps. The method is backward stable, so the eigenvalues lie
within a few `eps * norm(H)` of the exact ones, as LAPACK's do. Both triangles of `H` are
read, so `H` must be stored in full. Any other matrix is passed to
`eigvals(Hermitian(H))`.

# Examples
```jldoctest
julia> using Ludwig, StaticArrays

julia> Ludwig.hermitian_eigvals(@SMatrix [2.0+0im 1im; -1im 2.0])
2-element SVector{2, Float64} with indices SOneTo(2):
 1.0
 3.0
```

See also [`hermitian_eigen`](@ref).
"""
function hermitian_eigvals(H::SMatrix)
    return sort(jacobi_diagonal!(MMatrix(float(H)), nothing, jacobi_maxsweeps))
end

hermitian_eigvals(H::AbstractMatrix) = eigvals(Hermitian(H))

"""
    hermitian_eigen(H)

Return the eigenvalues of the Hermitian matrix `H` in ascending order and its orthonormal
eigenvectors, as the columns of `vectors`, in an `Eigen`.

For an `SMatrix` the rotations of [`hermitian_eigvals`](@ref) are accumulated into the
eigenvectors, still on the stack, and the `Eigen` holds an `SVector` and an `SMatrix`, so
this is a non-allocating replacement for `eigen(Hermitian(H))`. Each eigenvector is fixed
only up to a phase, which need not match LAPACK's. Any other matrix is passed to
`eigen(Hermitian(H))`.

# Examples
```jldoctest
julia> using Ludwig, StaticArrays, LinearAlgebra

julia> F = Ludwig.hermitian_eigen(@SMatrix [2.0+0im 1im; -1im 2.0]);

julia> F.values
2-element SVector{2, Float64} with indices SOneTo(2):
 1.0
 3.0

julia> F.vectors' * F.vectors ≈ I
true
```

See also [`hermitian_eigvals`](@ref).
"""
function hermitian_eigen(H::SMatrix)
    A = MMatrix(float(H))
    V = MMatrix(one(A))
    λ = jacobi_diagonal!(A, V, jacobi_maxsweeps)
    order = sortperm(λ)
    return Eigen(λ[order], SMatrix(V)[:, order])
end

hermitian_eigen(H::AbstractMatrix) = eigen(Hermitian(H))

"""
    jacobi_diagonal!(A::MMatrix, V, maxsweeps)

Diagonalize the Hermitian matrix `A` in place by cyclic Jacobi rotations and return its
diagonal, unsorted; see [`hermitian_eigvals`](@ref).

Each rotation is also applied to the columns of `V`, unless `V` is `nothing`, so that a
`V` starting as the identity ends holding the eigenvectors. Throw an error if the
off-diagonal elements are not negligible after `maxsweeps` sweeps.
"""
@inline function jacobi_diagonal!(A::MMatrix{N,N}, V, maxsweeps) where {N}
    threshold² = (eps(real(eltype(A))) * norm(A))^2
    @inbounds for _ in 1:maxsweeps
        rotated = false
        for q in 2:N, p in 1:q-1
            h = A[p, q]
            r² = abs2(h)
            if r² ≤ threshold²
                A[p, q] = A[q, p] = zero(h)
                continue
            end
            rotated = true
            # Remove the phase of h, then rotate in the (p, q) plane by the real angle θ
            # with cot 2θ = τ, as in Numerical Recipes §11.1
            r = sqrt(r²)
            phase = conj(h) / r
            a, b = real(A[p, p]), real(A[q, q])
            τ = (b - a) / 2r
            t = copysign(inv(abs(τ) + sqrt(1 + τ^2)), τ)
            c = inv(sqrt(1 + t^2))
            s = t * c
            A[p, p] = a - t * r
            A[q, q] = b + t * r
            A[p, q] = A[q, p] = zero(h)
            for m in 1:N
                (m == p || m == q) && continue
                x, y = A[m, p], phase * A[m, q]
                A[m, p] = c * x - s * y
                A[m, q] = c * y + s * x
                A[p, m] = conj(A[m, p])
                A[q, m] = conj(A[m, q])
            end
            rotate_columns!(V, p, q, phase, c, s)
        end
        rotated || return SVector(ntuple(i -> real(A[i, i]), Val(N)))
    end
    error("Jacobi rotations did not converge in $(maxsweeps) sweeps.")
end

"""
    rotate_columns!(V, p, q, phase, c, s)

Apply one rotation of [`jacobi_diagonal!`](@ref) to columns `p` and `q` of `V`.
"""
@inline function rotate_columns!(V::MMatrix, p, q, phase, c, s)
    @inbounds for m in axes(V, 1)
        x, y = V[m, p], phase * V[m, q]
        V[m, p] = c * x - s * y
        V[m, q] = c * y + s * x
    end
    return V
end

rotate_columns!(::Nothing, p, q, phase, c, s) = nothing
