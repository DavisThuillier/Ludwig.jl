abstract type PointGroup end

Base.@kwdef struct IBZ{D, T, L}
    vertices::Vector{SVector{D, T}}
    edges::Vector{Tuple{Int, Int}} # Indices of vertices
    faces::Vector{Vector{Int}} # Indices of vertices
    operations::Vector{SMatrix{D, D, T, L}}
    labels::Vector{Symbol}
end

function ibz end

"""
    close_group(generators) -> Vector{SMatrix}

Generate the full point group from a set of generator matrices via
breadth-first multiplication. Returns all unique elements of the group.
"""
function close_group(generators::Vector{<:SMatrix{D, D, T, L}};
                     tol = 1e-10) where {D, T}
    elements = SMatrix{D, D, T}[SMatrix{D, D, T, L}(I)] # G = {Id}
    frontier = copy(elements)
    while !isempty(frontier)
        new_frontier = SMatrix{D, D, T, L}[]
        for g in frontier, h in generators
            candidate = g * h
            if !any(e -> all(abs.(e .- candidate) .< tol), elements)
                push!(elements, candidate)
                push!(new_frontier, candidate)
            end
        end
        frontier = new_frontier
    end
    return elements
end

"""
    get_vertex(ibz, label) -> SVector

Look up an IBZ vertex by its symbolic label (e.g. :Γ, :X, :M).
"""
function get_vertex(ibz::IBZ, label::Symbol)
    i = findfirst(==(label), ibz.labels)
    i === nothing && error("Vertex label $label not found in IBZ")
    return ibz.vertices[i]
end
