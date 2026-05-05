struct PointGroup{D, T<:AbstractFloat, L} 
    operations::Vector{SMatrix{D,D,T,L}}
    function PointGroup(generators::Vector{<:AbstractMatrix}, lattice::Lattice{D, T, L}; tol = 1e-10) where {D, T, L}
        static_generators = [SMatrix{D,D,T,L}(g) for g in generators] # Throws Method error if sizes are incommensurate
        for (i, G) ∈ enumerate(static_generators)
            A = primitives(lattice)
            M = A \ G * A
            M_rounded = round.(M)
            drift = maximum(abs.(M .- M_rounded))

            drift < tol || throw(ArgumentError(
                "Generator $i does not preserve the lattice."
            ))
        end
        elements = close_group(static_generators; tol)
        return new{D, T, L}(elements)
    end
end

max_point_group_order(::Val{2}) = 12
max_point_group_order(::Val{3}) = 48

"""
    close_group(generators) -> Vector{SMatrix}

Generate the full point group from a set of generator matrices via
breadth-first multiplication. Returns all unique elements of the group.
"""
function close_group(generators::Vector{SMatrix{D, D, T, L}};
                     tol = 1e-10) where {D, T, L}
    elements = SMatrix{D, D, T, L}[SMatrix{D, D, T, L}(I)] # G = {Id}
    frontier = copy(elements)
    max_order = max_point_group_order(Val(D))
    while !isempty(frontier) && length(elements) <= max_order
        new_frontier = SMatrix{D, D, T, L}[]
        for g in frontier, h in generators
            candidate = g * h
            if !any(e -> all(isapprox.(e, candidate; atol=tol)), elements)
                push!(elements, candidate)
                push!(new_frontier, candidate)
            end
        end
        frontier = new_frontier
    end
    return elements
end


function close_group(generators::Vector{<:AbstractMatrix})
    isempty(generators) && throw(ArgumentError("generator list must be nonempty"))
    D = size(first(generators), 1); L = D^2
    static_generators = [SMatrix{D,D,Float64,L}(g) for g in generators] # Throws MethodError if sizes are incommensurate
    return close_group(static_generators)
end
