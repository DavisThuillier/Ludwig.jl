
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
struct NoLattice{D} <: AbstractLattice end 

"""
    Lattice(A::AbstractArray)

Construct a Bravais lattice from primitive vectors.

# Fields
- `primitives::SMatrix{2,2,Float64,4}`: 2×2 matrix whose columns are the primitive
  lattice vectors ``\\mathbf{a}_1`` and ``\\mathbf{a}_2``.
```

See also [`NoLattice`](@ref), [`primitives`](@ref), [`lattice_type`](@ref).
"""
struct Lattice{D, T, L} <: AbstractLattice
    primitives::SMatrix{D, D, T, L}
    function Lattice(A::AbstractArray)
        size(A)[1] == size(A)[2] || throw(DimensionMismatch(
            "Lattice primitive matrix must be square."
        ))
        iszero(det(A)) && throw(ArgumentError(
            "Specified primitive vectors are not linearly independent."
        ))
        D = size(A)[1]
        return new(SMatrix{D, D, T, D^2}(A))
    end
end
