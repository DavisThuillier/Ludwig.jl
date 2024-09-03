module Lattices

export Lattice, primitives, reciprocal_lattice_vectors, point_group, lattice_type

using LinearAlgebra
using StaticArrays
using ..Groups

const C₂ = get_cyclic_group(2)
const D₂ = get_dihedral_group(2)
const D₄ = get_dihedral_group(4)
const D₆ = get_dihedral_group(6)

const point_groups = Dict("Oblique" => C₂, "Rectangular" => D₂, "Square" => D₄, "Hexagonal" => D₆)
const num_bz_points = Dict("Oblique" => 6, "Rectangular" => 4, "Square" => 4, "Hexagonal" => 6)

struct Lattice
    primitives::Matrix
    Lattice(A::AbstractArray) = begin
        if size(A) == (2,2)
            if det(A) == 0
                error("Specified primitive vectors are not linearly independent.")
            else
                new(A)
            end
        else
            error("Invalid dimension.")
        end
    end
end

### Convenience Constructors ###
Lattice(a1::AbstractVector, a2::AbstractVector) = Lattice(hcat(a1, a2))

### Lattice Methods ###
Base.:(==)(l1::Lattice, l2::Lattice) = primitives(l1) == primitives(l2)
primitives(l::Lattice) = l.primitives
reciprocal_lattice_vectors(l::Lattice) = inv(primitives(l))
point_group(l::Lattice) = point_groups[lattice_type(l)]

function lattice_type(l::Lattice)
    p = primitives(l)
    a1 = p[:, 1]
    a2 = p[:, 2]

    cosθ = abs( dot(a1, a2) / (norm(a1) * norm(a2)) )

    if cosθ ≈ 0.0
        if norm(a1) == norm(a2)
            return "Square"
        else
            return "Rectangular"
        end
    elseif cosθ ≈ 0.5
        return "Hexagonal"
    else
        return "Oblique"
    end
end

function get_bz(l::Lattice)
    rlv = reciprocal_lattice_vectors(l)

    neighbors = map(x -> SVector{2}(rlv * x), collect.(Iterators.product(-2:2, -2:2))) |> vec
    sort!(neighbors, by = norm)
    deleteat!(neighbors, 1) # Deletes element corresponding to [0.0, 0.0]
    
    num_points = num_bz_points[lattice_type(l)]
    nearest_neighbors = neighbors[1:num_points]

    sort!(nearest_neighbors, by = x -> atan(x[2], x[1])) # Sort by angle

    vertices = Vector{SVector{2, Real}}(undef, num_points)

    for i in eachindex(nearest_neighbors)
        vertices[i] = get_perpendicular_bisector_intersection(nearest_neighbors[i], nearest_neighbors[mod(i, num_points) + 1])
    end
    
    deleteat!(vertices, vertices .== [[0.0, 0.0]])
    deleteat!(vertices, vertices .== [[]])

    return vertices
end

function map_to_bz(k, rlv) #FIXME: Fails for hexagonal lattice
    k1 = inv(rlv) * k # k in reciprocal lattice basis
    k1 = mod.(k1 .+ 0.5, 1.0) .- 0.5
    @show k1
    return rlv * k1
end

function in_polygon(k, p)
    w = _winding_number(p .- Ref(k)) # Winding number of polygon wrt k
    return !(abs(w) < 0.1) # Returns false is w ≈ 0
end

function _winding_number(v)
    w = 0
    for i in eachindex(v)
        j = (i == length(v)) ? 1 : i + 1 

        if v[i][2] * v[j][2] < 0 # [vᵢvⱼ] crosses the x-axis
            r = v[i][1] + v[i][2] * (v[j][1] - v[i][1]) / (v[i][2] - v[j][2]) # x-coordinate of intersection of [vᵢvⱼ] with x-axis
            if r > 0
                v[i][2] < 0 ? (w += 1) : (w -= 1)
            end
        elseif v[i][2] == 0 && v[i][1] > 0 # vᵢ on the positive x-axis
            v[j][2] > 0 ? (w += 0.5) : (w -= 0.5)
        elseif v[j][2] == 0 && v[j][2] > 0 # vⱼ on the positive x-axis
            v[i][2] < 0 ? (w += 0.5) : (w -= 0.5)
        end
    end
    return w
end

# function get_ibz(l::Lattice)
#     G = point_group(l)
#     bz = get_bz(l)
#     ibz = bz

#     for vertex in bz
#         for g in eachindex(G)
#             if !(g*vertex ≈ vertex)

#             end
#         end
#     end

# end

function get_perpendicular_bisector_intersection(v1, v2)
    V = [v1[2] v2[2]; -v1[1] -v2[1]]
    det(V) == 0 && return [0.0, 0.0] # No intersection
    
    t = 0.5 * inv(V) * (v2 - v1)
    return 0.5 * v1 + t[1] * V[:, 1]
end

end # module
