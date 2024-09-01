module Lattices2D

using LinearAlgebra

const point_groups = Dict("Oblique" => "C₂", "Rectangular" => "D₂", "Square" => "D₄", "Hexagonal" => "D₆")

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

primitives(l::Lattice) = l.primitives
reciprocal_lattice_vectors(l::Lattice) = inv(primitives(l))
point_group(l::Lattice) = point_groups(lattice_type(l))

function lattice_type(l::Lattice)
    p = primitives(l)
    a1 = p[:, 1]
    a2 = p[:, 2]

    cosθ = abs( dot(a1, a2) / (norm(a1) * norm(a2)) )

    if cosθ == 0.0
        if norm(a1) == norm(a2)
            return "Square"
        else
            return "Rectangular"
        end
    elseif cosθ == 0.5
        return "Hexagonal"
    else
        return "Oblique"
    end
end

end # module
