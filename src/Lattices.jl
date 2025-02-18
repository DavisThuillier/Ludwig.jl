module Lattices

export AbstractLattice, Lattice, NoLattice, primitives, reciprocal_lattice_vectors, point_group, lattice_type
export get_bz, get_ibz, map_to_bz

using LinearAlgebra
using StaticArrays
using ..Groups
using ..GeometryUtilities

const C₂ = Groups.get_cyclic_group(2)
const D₂ = Groups.get_dihedral_group(2)
const D₄ = Groups.get_dihedral_group(4)
const D₆ = Groups.get_dihedral_group(6)

const point_groups = Dict("Oblique" => C₂, "Rectangular" => D₂, "Square" => D₄, "Hexagonal" => D₆)
const num_bz_points = Dict("Oblique" => 6, "Rectangular" => 4, "Square" => 4, "Hexagonal" => 6)

abstract type AbstractLattice end

struct NoLattice <: AbstractLattice end # Singleton type for free electron gas

struct Lattice <: AbstractLattice
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
        vertices[i] = perpendicular_bisector_intersection(nearest_neighbors[i], nearest_neighbors[mod(i, num_points) + 1])
    end
    
    deleteat!(vertices, vertices .== [[0.0, 0.0]])
    deleteat!(vertices, vertices .== [[]])

    sort!(vertices, by = x -> atan(x[2], x[1])) # Sort by angle

    return vertices
end

function get_ibz(l::Lattice)
    G = point_group(l)
    bz = get_bz(l)
    ibz = deepcopy(bz)

    ops = deepcopy(G.elements) 

    tol = 1e-4 * diameter(bz) # Tolerance for comparing whether two points are equivalent scaled by size of BZ
    
    for v in bz
        ops_to_delete = [] 
        for (i,g) in enumerate(ops)
            Groups.is_identity(g) && continue
            
            O = Groups.get_matrix_representation(g)

            if !(O*v ≈ v) # Symmetry does not fix vertex
                push!(ops_to_delete, i)

                midpoint = (O*v + v) / 2.0
                n = O*v - v

                σ1 = sign(signed_area(v, midpoint, [n[2], -n[1]]))

                to_add = [] # Intersections to add to IBZ
                to_delete = Int[] # Indices of points in IBZ to delete
                for (j, w) in enumerate(ibz)
                    norm(w) < tol && continue

                    σ2 = signed_area(w, midpoint, [n[2], -n[1]])

                    if σ1 * σ2 < 0
                        push!(to_delete, findfirst(==(w), ibz))
                    end

                    k = (j == length(ibz)) ? 1 : j + 1 
                    int = intersection(midpoint, [n[2], -n[1]], ibz[j], ibz[k] - ibz[j])

                    any(isnan.(int)) && continue
                    !in_polygon(int, ibz) && continue

                    # Check if intersection coincides with a vertex of IBZ or an edge
                    if !any(map(z -> norm(int-z) < tol, vcat(ibz, to_add))) && abs(σ2) > tol^2
                        push!(to_add, int)
                    end
                end

                deleteat!(ibz, to_delete)
                append!(ibz, to_add)

                sort!(ibz, by = x -> atan(x[2] - ibz[1][2], x[1] - ibz[1][1])) 
            end
            
        end
        deleteat!(ops, ops_to_delete)
    end

    # Enforce that origin is in IBZ
    !any(norm.(ibz) .< tol) && push!(ibz, [0.0, 0.0])

    ## Map IBZ to have minimal average angle
    imin = 1
    θmin = 2pi
    for i in eachindex(G.elements)
        O = Groups.get_matrix_representation(G.elements[i])
        Oibz = map(x -> O*x, ibz[norm.(ibz) .> tol])
        θ = sum(map(x -> atan(x[2], x[1]), Oibz)) / length(Oibz) # Average angle of IBZ points
        if abs(θ) < θmin
            θmin = abs(θ)
            imin = i
        end
    end
    
    O = Groups.get_matrix_representation(G.elements[imin])
    ibz = map(x -> SVector{2}(O*x), ibz)

    sort!(ibz, by = x -> atan(x[2], x[1])) # Sort by angle

    return ibz
end

function map_to_bz(k, bz, rlv, invrlv)
    if in_polygon(k, bz)
        return k
    else 
        n = round.(invrlv * k)
        k -= rlv * n
    
        if !in_polygon(k, bz)
            min_d = norm(k)
            for i in -1:1
                for j in -1:1
                    i == 0 && j == 0 && continue
                    d = norm(k - rlv * [i,j])
                    if d < min_d
                        n = [i,j]
                        min_d = d
                    end
                end
            end

            return SVector{2}(k - rlv * n)
        else
            return k
        end
    end
end

map_to_bz(k, bz, rlv) = map_to_bz(k, bz, rlv, inv(rlv))
map_to_bz(k, l::Lattice) = map_to_bz(k, get_bz(l), reciprocal_lattice_vectors(l))

end # module
