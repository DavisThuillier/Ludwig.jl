"""
    Group

Abstract supertype for finite groups.

All concrete subtypes must provide an `elements` field whose length equals the group order.

See also [`PermutationGroup`](@ref), [`order`](@ref), [`get_table`](@ref).
"""
abstract type Group end

"""
    GroupElement

Abstract supertype for elements of a [`Group`](@ref).

See also [`PermutationGroupElement`](@ref), [`inverse`](@ref), [`is_identity`](@ref).
"""
abstract type GroupElement end

"""
    order(G::Group)

Return the number of elements in group `G`.

# Examples
```jldoctest
julia> using Ludwig

julia> G = get_cyclic_group(4);

julia> order(G)
4
```

See also [`get_table`](@ref).
"""
order(G::Group) = length(G.elements)

"""
    PermutationGroupElement

A [`GroupElement`](@ref) represented by a permutation of `1:n`.

The permutation is stored as a `Vector{Int}` where entry `i` gives the image of `i` under
the permutation.

# Examples
```jldoctest
julia> using Ludwig

julia> g = PermutationGroupElement([2, 3, 1]);

julia> g.permutation
3-element Vector{Int64}:
 2
 3
 1
```

See also [`PermutationGroup`](@ref), [`inverse`](@ref), [`is_identity`](@ref),
[`get_matrix_representation`](@ref).
"""
struct PermutationGroupElement <: GroupElement
    permutation::Vector{Int}
    PermutationGroupElement(perm) = begin
        sort(perm) != collect(eachindex(perm)) ? error("Improper permutation.") : new(perm)
    end
end

Base.length(g::PermutationGroupElement) = Base.length(g.permutation)
Base.:(==)(g::PermutationGroupElement, h::PermutationGroupElement) = g.permutation == h.permutation
Base.show(io::IO, ::MIME"text/plain", g::PermutationGroupElement) = begin
    X = Matrix(hcat(1:length(g), g.permutation)')
    show(io, "text/plain", X)
end

function get_transposition(i::Int, j::Int, n::Int)
    permutation = collect(1:n)
    permutation[i] = j
    permutation[j] = i
    return PermutationGroupElement(permutation)
end

function Base.:*(g::PermutationGroupElement, h::PermutationGroupElement)::PermutationGroupElement
    length(g) != length(h) && throw(DimensionMismatch("Group elements must belong to the same permutation group."))

    gh_permutation = map(i -> g.permutation[i], h.permutation)
    return PermutationGroupElement(gh_permutation)
end

"""
    inverse(g::PermutationGroupElement)

Return the inverse of permutation group element `g`.

# Examples
```jldoctest
julia> using Ludwig

julia> g = PermutationGroupElement([2, 3, 1]);

julia> inverse(g).permutation
3-element Vector{Int64}:
 3
 1
 2
```

See also [`is_identity`](@ref).
"""
function inverse(g::PermutationGroupElement)
    invperm = Vector{Int}(undef, length(g))
    for i in eachindex(g.permutation)
        invperm[g.permutation[i]] = i
    end
    return PermutationGroupElement(invperm)
end

"""
    is_identity(g::PermutationGroupElement)

Return `true` if `g` is the identity permutation.

# Examples
```jldoctest
julia> using Ludwig

julia> is_identity(PermutationGroupElement([1, 2, 3]))
true

julia> is_identity(PermutationGroupElement([2, 1, 3]))
false
```

See also [`inverse`](@ref).
"""
is_identity(g::PermutationGroupElement) = g.permutation == collect(eachindex(g.permutation))
get_identity_permutation(n::Int) = PermutationGroupElement(collect(1:n))

function closure(generators...)
    length(generators) == 0 && error("At least one generator must be passed.")

    elements = Vector{typeof(first(generators))}(undef, 0)

    for g in generators
        gn = g
        while true
            (gn in elements) ? break : push!(elements, gn)
            gn = gn * g # Compute g^n+1
        end

        for h in elements
            prod = g*h
            !(prod in elements) && push!(elements, prod)
        end
    end

    identity_index = findfirst(is_identity, elements)
    circshift!(elements, 1 - identity_index) # Shift identity to first element

    return elements
end

"""
    PermutationGroup

A finite [`Group`](@ref) whose elements are [`PermutationGroupElement`](@ref)s.

Constructed from one or more generator elements; the full group is computed via closure.

# Examples
```jldoctest
julia> using Ludwig

julia> G = PermutationGroup(PermutationGroupElement([2, 1]));

julia> order(G)
2
```

See also [`get_cyclic_group`](@ref), [`get_dihedral_group`](@ref),
[`get_symmetric_group`](@ref).
"""
struct PermutationGroup <: Group
    elements::Vector{PermutationGroupElement}
    PermutationGroup(elements...) = new(closure(elements...))
end

Base.iterate(G::PermutationGroup) = iterate(G.elements)
Base.iterate(G::PermutationGroup, s::Int) = iterate(G.elements, s)

"""
    get_table(G::Group)

Return the multiplication table of `G` as an integer matrix.

Entry `[i, j]` is the index of `G.elements[i] * G.elements[j]` in `G.elements`.

# Examples
```jldoctest
julia> using Ludwig

julia> G = get_cyclic_group(2);

julia> get_table(G)
2×2 Matrix{Int64}:
 1  2
 2  1
```

See also [`order`](@ref).
"""
function get_table(G::Group)
    table = Matrix{Int}(undef, order(G), order(G))
    for i in eachindex(G.elements)
        for j in eachindex(G.elements)
            table[i,j] = findfirst(==(G.elements[i] * G.elements[j]), G.elements)
        end
    end
    return table
end

"""
    get_cyclic_group(n)

Return the cyclic group ``C_n`` of order `n` as a [`PermutationGroup`](@ref).

# Examples
```jldoctest
julia> using Ludwig

julia> G = get_cyclic_group(6);

julia> order(G)
6
```

See also [`get_dihedral_group`](@ref), [`get_symmetric_group`](@ref).
"""
function get_cyclic_group(n::Int)
    g = PermutationGroupElement(circshift(collect(1:n), -1))
    return PermutationGroup(g)
end

"""
    get_dihedral_group(n)

Return the dihedral group ``D_n`` of order `2n` as a [`PermutationGroup`](@ref).

``D_n`` is the symmetry group of a regular `n`-gon, generated by a rotation and a
reflection.

# Examples
```jldoctest
julia> using Ludwig

julia> G = get_dihedral_group(4);

julia> order(G)
8
```

See also [`get_cyclic_group`](@ref), [`get_symmetric_group`](@ref).
"""
function get_dihedral_group(n::Int)
    r = PermutationGroupElement(circshift(collect(1:n), -1)) # Rotation
    s = PermutationGroupElement(circshift(reverse(collect(1:n)), 1)) # Reflection
    return PermutationGroup(r, s)
end

"""
    get_symmetric_group(n)

Return the symmetric group ``S_n`` of order `n!` as a [`PermutationGroup`](@ref).

# Examples
```jldoctest
julia> using Ludwig

julia> G = get_symmetric_group(3);

julia> order(G)
6
```

See also [`get_cyclic_group`](@ref), [`get_dihedral_group`](@ref).
"""
function get_symmetric_group(n::Int)
    if n < 1
        throw(DomainError("Sₙ not defined for n < 1."))
    elseif n == 1
        return PermutationGroup(get_identity_permutation(n))
    else
        elements = Vector{PermutationGroupElement}(undef, 0)
        for i in 1:n-1
            for j in i+1:n
                push!(elements, get_transposition(i, j, n))
            end
        end
        return PermutationGroup(elements...)
    end
end

function _unity_root_vector(p::Int, n::Int)
    return [cos(2pi * p / n), sin(2pi * p / n)]
end

"""
    get_matrix_representation(g::PermutationGroupElement)

Return the 2×2 orthogonal matrix representing `g` as a point-group operation.

The matrix is computed by mapping the first two vertices of a regular `n`-gon (where
`n = length(g)`) under the permutation, expressed in the standard basis.

# Examples
```jldoctest
julia> using Ludwig

julia> g = PermutationGroupElement([1, 2, 3, 4]);  # identity of C₄

julia> get_matrix_representation(g)
2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```

See also [`PermutationGroupElement`](@ref), [`get_cyclic_group`](@ref).
"""
function get_matrix_representation(g::PermutationGroupElement)
    n = length(g)
    n < 2 && error(DimensionMismatch("Permutation must have length at least 2 to be represented by a 2x2 matrix."))

    v₁ = _unity_root_vector(0, n)
    v₂ = _unity_root_vector(1, n)
    T = hcat(v₁, v₂) # Root of unity basis

    w₁ = _unity_root_vector(g.permutation[1] - 1, n)
    w₂ = _unity_root_vector(g.permutation[2] - 1, n)

    M = hcat(w₁, w₂) * inv(T) # Change of basis composed with permutation map of roots of polygon vertices

    for i in eachindex(M)
        abs(M[i]) < 1e-12 && (M[i] = 0.0)
    end

    return M
end
