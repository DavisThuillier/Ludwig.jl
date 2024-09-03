module Groups

# import Base: *

export Group, PermutationGroup, PermutationGroupElement, GroupElement
export inverse, is_identity

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

function closure(generators)
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
    PermutationGroup(elements...) = new(closure(elements))
end

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
    display(g)
    return PermutationGroup(g)
end

function get_dihedral_group(n::Int)
    r = PermutationGroupElement(circshift(collect(1:n), -1)) # Rotation
    s = PermutationGroupElement(circshift(reverse(collect(1:n)), 1)) # Reflection 
    return PermutationGroup(r, s)
end

function get_symmetric_group(n::Int)

end

end
