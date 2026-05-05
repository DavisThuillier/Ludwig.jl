
"""
    AbstractLattice

Abstract supertype for all lattice representations.

See also [`Lattice`](@ref), [`NoLattice`](@ref).
"""
abstract type AbstractLattice end

"""
    NoLattice <: AbstractLattice

Singleton type representing the absence of a periodic lattice (free electron gas).

Use `NoLattice()` when no lattice periodicity is needed, for example when constructing
a circular Fermi surface mesh.

See also [`Lattice`](@ref).
"""
struct NoLattice <: AbstractLattice end 

"""
    Lattice(A::AbstractArray)

Construct a Bravais lattice from primitive vectors.

# Fields
- `primitives::SMatrix{2,2,Float64,4}`: 2×2 matrix whose columns are the primitive
  lattice vectors ``\\mathbf{a}_1`` and ``\\mathbf{a}_2``.
```

See also [`NoLattice`](@ref), [`primitives`](@ref), [`lattice_type`](@ref).
"""
struct Lattice{D, T<:AbstractFloat, L} <: AbstractLattice
    primitives::SMatrix{D, D, T, L}
    reciprocal_vectors::SMatrix{D, D, T, L}
    brillouin_zone::BrillouinZone{D, T}

    function Lattice(A::AbstractMatrix{T}) where {T}
        size(A)[1] == size(A)[2] || throw(DimensionMismatch(
            "Lattice primitive matrix must be square."
        ))
        iszero(det(A)) && throw(ArgumentError(
            "Specified primitive vectors are not linearly independent."
        ))
        D = size(A)[1]
        return new{D,T,L}(SMatrix{D, D, T, D^2}(A))
    end
end

"""
    reciprocal_lattice_vectors(A)

Return the reciprocal-lattice basis ``B = 2\\pi (A^{-1})^\\mathsf{T}`` of the
Bravais lattice whose primitive vectors are the columns of `A`. The columns of
the result are ``\\mathbf{b}_i`` satisfying
``\\mathbf{a}_i \\cdot \\mathbf{b}_j = 2\\pi \\delta_{ij}``.
"""
reciprocal_lattice_vectors(A::AbstractMatrix) = 2π * inv(transpose(A))
