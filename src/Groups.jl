module Groups

export Group, PermutationGroup, PermutationGroupElement, GroupElement
export get_cyclic_group, get_dihedral_group, get_symmetric_group, get_table, inverse
export get_matrix_representation

abstract type Group end
abstract type GroupElement end

order(G::Group) = length(G.elements)

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

function inverse(g::PermutationGroupElement)
    invperm = Vector{Int}(undef, length(g))
    for i in eachindex(g.permutation)
        invperm[g.permutation[i]] = i
    end
    return PermutationGroupElement(invperm)
end

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

struct PermutationGroup <:Group
    elements::Vector{PermutationGroupElement}
    PermutationGroup(elements...) = new(closure(elements...))
end

Base.iterate(G::PermutationGroup) = iterate(G.elements)
Base.iterate(G::PermutationGroup, s::Int) = iterate(G.elements, s)

function get_table(G::Group)
    table = Matrix{Int}(undef, order(G), order(G))
    for i in eachindex(G.elements)
        for j in eachindex(G.elements)
            table[i,j] = findfirst(==(G.elements[i] * G.elements[j]), G.elements)
        end
    end
    return table
end

function get_cyclic_group(n::Int)
    g = PermutationGroupElement(circshift(collect(1:n), -1))
    return PermutationGroup(g)
end

function get_dihedral_group(n::Int)
    r = PermutationGroupElement(circshift(collect(1:n), -1)) # Rotation
    s = PermutationGroupElement(circshift(reverse(collect(1:n)), 1)) # Reflection 
    return PermutationGroup(r, s)
end

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

end
