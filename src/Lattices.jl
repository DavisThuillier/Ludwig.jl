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
point_group(l::Lattice) = point_groups(lattice_type(l))

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
    b1 = rlv[:, 1]
    b2 = rlv[:, 2] 

    vertices = Vector{AbstractArray}(undef, 0)

    get_perpendicular_bisector_intersection(b1, b1 + b2) |> x -> push!(vertices, x)
    get_perpendicular_bisector_intersection(b1 + b2, b2) |> x -> push!(vertices, x)
    get_perpendicular_bisector_intersection(b2, b2 - b1) |> x -> push!(vertices, x)
    get_perpendicular_bisector_intersection(b2 - b1, -b1) |> x -> push!(vertices, x)
    get_perpendicular_bisector_intersection(-b1, -b1 - b2) |> x -> push!(vertices, x)
    get_perpendicular_bisector_intersection(-b1 - b2, -b2) |> x -> push!(vertices, x)
    get_perpendicular_bisector_intersection(-b2, b1 - b2) |> x -> push!(vertices, x)
    get_perpendicular_bisector_intersection(b1 - b2, b1) |> x -> push!(vertices, x)
    
    deleteat!(vertices, vertices .== [[0.0, 0.0]])
    deleteat!(vertices, vertices .== [[]])
end

function get_perpendicular_bisector_intersection(v1, v2)
    V = [v1[2] v2[2]; -v1[1] -v2[1]]
    det(V) == 0 && return [] # No intersection
    
    t = 0.5 * inv(V) * (v2 - v1)
    return (0.5 + t[1]) * v1
end

end # module
