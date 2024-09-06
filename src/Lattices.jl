module Lattices

export Lattice, primitives, reciprocal_lattice_vectors, point_group, lattice_type

export get_bz

using LinearAlgebra
using StaticArrays
using ..Ludwig: Groups

const C₂ = Groups.get_cyclic_group(2)
const D₂ = Groups.get_dihedral_group(2)
const D₄ = Groups.get_dihedral_group(4)
const D₆ = Groups.get_dihedral_group(6)

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
reciprocal_lattice_vectors(l::Lattice) = Array(inv(primitives(l))')
point_group(l::Lattice) = point_groups[lattice_type(l)]

# struct HalfSpace{T}
#     n<:AbstractVector{T}
#     d::T
# end

# https://en.wikipedia.org/wiki/Lattice_reduction
function gauss_reduce(A, max_iter = 1e4)
    i = 1
    u = A[:, 1]
    v = A[:, 2]
    while norm(v) < norm(u) && i < max_iter
        q = round(Int, dot(u, v) / norm(v)^2)
        r = u - q * v
        u = v
        v = r
        i += 1
    end

    return hcat(u, v)
end

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

    vertices = Vector{SVector{2, Float64}}(undef, num_points)

    for i in eachindex(nearest_neighbors)
        vertices[i] = get_perpendicular_bisector_intersection(nearest_neighbors[i], nearest_neighbors[mod(i, num_points) + 1])
    end
    
    deleteat!(vertices, vertices .== [[0.0, 0.0]])
    deleteat!(vertices, vertices .== [[]])

    sort!(vertices, by = x -> atan(x[2], x[1])) # Sort by angle

    return vertices
end

function map_to_bz(k, rlv) #FIXME: Fails for hexagonal lattice
    k1 = inv(rlv) * k # k in reciprocal lattice basis
    k1 = mod.(k1 .+ 0.5, 1.0) .- 0.5
    @show k1
    return rlv * k1
end

function in_polygon(k, p, atol = 1e-12)
    is_vertex = false
    for vertex in p
        isapprox(k, vertex, atol = atol) && (is_vertex = true)
    end
    is_vertex && return true

    w = _winding_number(p .- Ref(k), atol) # Winding number of polygon wrt k
    return !(abs(w) < atol) # Returns false is w ≈ 0
end

function _winding_number(v, atol = 1e-12)
    w = 0
    for i in eachindex(v)
        j = (i == length(v)) ? 1 : i + 1 

        
        if v[i][2] * v[j][2] < 0 # [vᵢvⱼ] crosses the x-axis
            m = (v[j][1] - v[i][1]) / (v[i][2] - v[j][2])
            r = v[i][1] + m * v[i][2]  # x-coordinate of intersection of [vᵢvⱼ] with x-axis
            if abs(r) < atol
                m > 0 ? (w += 0.5) : (w -= 0.5)
            elseif r > 0
                v[i][2] < 0 ? (w += 1) : (w -= 1)
            end
        elseif abs(v[i][2]) < atol && abs(v[j][2]) > atol && v[i][1] > 0 # vᵢ on the positive x-axis
            v[j][2] > 0 ? (w += 0.5) : (w -= 0.5)
        elseif abs(v[i][2]) > atol && abs(v[j][2]) < atol && v[j][1] > 0 # vⱼ on the positive x-axis
            v[i][2] < 0 ? (w += 0.5) : (w -= 0.5)
        end
    end
    return w
end

function _signed_area(x, y, z)
    return 0.5 * (-y[1] * x[2] + z[1] * x[2] + x[1] * y[2] - z[1] * y[2] - x[1]*z[2] + y[1] * z[2])
end

function get_ibz(l::Lattice)
    G = point_group(l)
    bz = get_bz(l)
    ibz = deepcopy(bz)

    ops = deepcopy(G.elements)
    v = bz[1]

    while length(ops) > 0
        g = pop!(ops)
        O = Groups.get_matrix_representation(g)
        if !(O*v ≈ v) # Symmetry does not fix vertex
            midpoint = (O*v + v) / 2.0
            n = O*v - v

            σ1 = sign(_signed_area(v, midpoint, [n[2], -n[1]]))

            for (j, w) in enumerate(ibz)
                σ2 = _signed_area(w, midpoint, [n[2], -n[1]])

                if σ1 * σ2 < 0
                    w ∈ ibz && deleteat!(ibz, findfirst(==(w), ibz))
                end

                k = (j == length(bz)) ? 1 : j + 1 
                int = intersection(midpoint, [n[2], -n[1]], bz[j], bz[k] - bz[j])
                (int == [0.0, 0.0] || !in_polygon(int, bz)) && continue

                if !any(map(z -> int ≈ z, bz))

                    push!(ibz, int)
                end
            end
        end
    end
    push!(ibz, [0.0, 0.0])

    return ibz
end

function intersection(a1, v1, a2, v2)
    V = hcat(v1, -v2)
    det(V) == 0 && return [0.0, 0.0] # No intersection

    t = inv(V) * (a2 - a1)
    return a1 + t[1] * v1
end

function get_perpendicular_bisector_intersection(v1, v2)
    return intersection(0.5 * v1, [v1[2], -v1[1]], 0.5 * v2, [v2[2], -v2[1]])
end

end # module
