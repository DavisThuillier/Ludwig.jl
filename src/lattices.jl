# 2D crystallographic point groups exposed as module-level constants.
# C₂ is the cyclic group of order 2 (oblique); D₂, D₄, D₆ are the dihedral groups of order
# 2n that fix a regular n-gon (rectangular, square, hexagonal lattices respectively). They
# are the point groups assigned to each Bravais-lattice type by `point_groups`.
const C₂ = get_cyclic_group(2)
const D₂ = get_dihedral_group(2)
const D₄ = get_dihedral_group(4)
const D₆ = get_dihedral_group(6)

"""
    point_groups

Mapping from Bravais-lattice type string to its 2D point group: Oblique → C₂,
Rectangular → D₂, Square → D₄, Hexagonal → D₆.
"""
const point_groups = Dict("Oblique" => C₂, "Rectangular" => D₂, "Square" => D₄, "Hexagonal" => D₆)

"""
    num_bz_points

Mapping from Bravais-lattice type string to the number of vertices of its first Brillouin
zone.
"""
const num_bz_points = Dict("Oblique" => 6, "Rectangular" => 4, "Square" => 4, "Hexagonal" => 6)

"""
    AbstractLattice

Abstract supertype for all lattice representations.

See also [`Lattice`](@ref), [`NoLattice`](@ref).
"""
abstract type AbstractLattice end

"""
    NoLattice <: AbstractLattice

Singleton type representing the absence of a periodic lattice (free electron gas).

Use `NoLattice()` when no lattice periodicity is needed, for example when constructing
a circular Fermi surface mesh.

See also [`Lattice`](@ref).
"""
struct NoLattice <: AbstractLattice end # Singleton type for free electron gas

"""
    Lattice(A::AbstractArray)
    Lattice(a1::AbstractVector, a2::AbstractVector)

Construct a 2D Bravais lattice from primitive vectors.

`A` must be a 2×2 matrix with linearly independent columns. The two-argument form
assembles the matrix as `hcat(a1, a2)`.

# Fields
- `primitives::SMatrix{2,2,Float64,4}`: 2×2 matrix whose columns are the primitive
  lattice vectors ``\\mathbf{a}_1`` and ``\\mathbf{a}_2``.

# Examples
```julia
julia> Lattice([1.0 0.0; 0.0 1.0])   # square lattice
julia> Lattice([1.0, 0.0], [0.5, √3/2])  # hexagonal lattice
```

See also [`NoLattice`](@ref), [`primitives`](@ref), [`lattice_type`](@ref).
"""
struct Lattice <: AbstractLattice
    primitives::SMatrix{2,2,Float64,4}
    function Lattice(A::AbstractArray)
        size(A) == (2, 2) || throw(DimensionMismatch(
            "Lattice primitive matrix must be 2×2; got size $(size(A))."
        ))
        iszero(det(A)) && throw(ArgumentError(
            "Specified primitive vectors are not linearly independent."
        ))
        return new(SMatrix{2,2,Float64,4}(A))
    end
end

###
### Convenience Constructors
###
Lattice(a1::AbstractVector, a2::AbstractVector) = Lattice(hcat(a1, a2))

###
### Lattice Methods
###
Base.:(==)(l1::Lattice, l2::Lattice) = primitives(l1) == primitives(l2)

"""
    primitives(l::Lattice)

Return the 2×2 matrix of primitive lattice vectors for `l`.

Columns of the returned matrix are the primitive vectors ``\\mathbf{a}_1`` and
``\\mathbf{a}_2``.

# Examples
```jldoctest
julia> primitives(Lattice([1.0 0.0; 0.0 1.0]))
2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```

See also [`reciprocal_lattice_vectors`](@ref).
"""
primitives(l::Lattice) = l.primitives

"""
    reciprocal_lattice_vectors(l::Lattice)

Return the 2×2 matrix of reciprocal lattice vectors for `l`.

Uses the convention ``B = 2π (A^{-1})^\\top``, where `A` is the primitive vector matrix,
so that ``A^\\top B = 2π I``.

# Examples
```jldoctest
julia> reciprocal_lattice_vectors(Lattice([1.0 0.0; 0.0 1.0])) ≈ [2π 0.0; 0.0 2π]
true
```

See also [`primitives`](@ref), [`get_bz`](@ref).
"""
reciprocal_lattice_vectors(l::Lattice) = 2π * SMatrix{2,2,Float64,4}(inv(primitives(l))')

"""
    point_group(l::Lattice)

Return the point group of `l` as a [`Group`](@ref).

The group is selected based on [`lattice_type`](@ref): Oblique → C₂, Rectangular → D₂,
Square → D₄, Hexagonal → D₆.

See also [`lattice_type`](@ref), [`get_ibz`](@ref).
"""
point_group(l::Lattice) = point_groups[lattice_type(l)]

"""
    lattice_type(l::Lattice; atol = 1e-8)

Return the Bravais lattice type of `l` as a string.

Classifies the lattice by comparing the angle between primitive vectors and their norms.
Returns one of `"Square"`, `"Rectangular"`, `"Hexagonal"`, or `"Oblique"`.

`atol` is the absolute tolerance applied both to ``|\\cos\\theta|`` (so that primitives
within `atol` of perpendicular are treated as such, and likewise within `atol` of the
hexagonal angle ``\\cos\\theta = 1/2``) and as a relative tolerance on the norm comparison
that distinguishes Square from Rectangular. The default is loose enough to absorb
double-precision construction noise (e.g. evaluating `√3/2` at runtime) but tight enough
that genuinely oblique lattices near 89.99° are not promoted to Square.

# Examples
```jldoctest
julia> lattice_type(Lattice([1.0 0.0; 0.0 1.0]))
"Square"

julia> lattice_type(Lattice([1.0 0.0; 0.0 2.0]))
"Rectangular"

julia> lattice_type(Lattice([1.0 0.5; 0.0 √3/2]))
"Hexagonal"
```

See also [`point_group`](@ref), [`get_bz`](@ref).
"""
function lattice_type(l::Lattice; atol = 1e-8)
    p = primitives(l)
    a1 = p[:, 1]
    a2 = p[:, 2]
    n1, n2 = norm(a1), norm(a2)

    cosθ = abs(dot(a1, a2) / (n1 * n2))

    if cosθ ≤ atol
        return isapprox(n1, n2; rtol = atol) ? "Square" : "Rectangular"
    elseif abs(cosθ - 0.5) ≤ atol
        return "Hexagonal"
    else
        return "Oblique"
    end
end

"""
    get_bz(l::Lattice)

Return the vertices of the first Brillouin zone for `l` as a vector of
`SVector{2,Float64}`, sorted counterclockwise by angle from the origin.

The BZ is constructed as the Wigner-Seitz cell of the reciprocal lattice via
perpendicular bisectors of nearest-neighbor reciprocal lattice vectors.

See also [`get_ibz`](@ref), [`reciprocal_lattice_vectors`](@ref), [`map_to_bz`](@ref).
"""
function get_bz(l::Lattice)
    rlv = reciprocal_lattice_vectors(l)

    neighbors = vec([rlv * SVector{2,Int}(i, j) for i in -2:2, j in -2:2])
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

    sort!(vertices, by = x -> atan(x[2], x[1])) # Sort by angle

    return vertices
end

"""
    get_ibz(l::Lattice)

Return the vertices of the irreducible Brillouin zone for `l` as a vector of
`SVector{2,Float64}`, sorted counterclockwise by angle from the origin.

The IBZ is constructed by iteratively bisecting the BZ with the mirror planes of the
point group, then rotated to minimize the average angle of its vertices. The origin is
always included as a vertex.

See also [`get_bz`](@ref), [`point_group`](@ref), [`ibz_mesh`](@ref).
"""
function get_ibz(l::Lattice)
    G = point_group(l)
    bz = get_bz(l)
    ibz = deepcopy(bz)

    ops = deepcopy(G.elements)

    tol = 1e-4 * diameter(bz) # Tolerance for comparing whether two points are equivalent scaled by size of BZ

    for v in bz
        ops_to_delete = Int[]
        for (i,g) in enumerate(ops)
            is_identity(g) && continue

            O = get_matrix_representation(g)

            if !(O*v ≈ v) # Symmetry does not fix vertex
                push!(ops_to_delete, i)

                midpoint = (O*v + v) / 2.0
                n = O*v - v

                σ1 = sign(signed_area(v, midpoint, [n[2], -n[1]]))

                to_add = SVector{2,Float64}[] # Intersections to add to IBZ
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
        O = get_matrix_representation(G.elements[i])
        Oibz = map(x -> O*x, ibz[norm.(ibz) .> tol])
        θ = sum(map(x -> atan(x[2], x[1]), Oibz)) / length(Oibz) # Average angle of IBZ points
        if abs(θ) < θmin
            θmin = abs(θ)
            imin = i
        end
    end

    O = get_matrix_representation(G.elements[imin])
    ibz = map(x -> SVector{2}(O*x), ibz)

    sort!(ibz, by = x -> atan(x[2], x[1])) # Sort by angle

    return ibz
end

"""
    map_to_bz(k, bz, rlv, invrlv)
    map_to_bz(k, bz, rlv)
    map_to_bz(k, l::Lattice)

Map the momentum vector `k` into the first Brillouin zone `bz`.

If `k` already lies inside `bz`, it is returned unchanged. Otherwise, a reciprocal
lattice translation ``\\mathbf{G} = rlv \\cdot \\mathbf{n}`` is found such that
``k - \\mathbf{G}`` lies inside `bz`. `invrlv` is the inverse of `rlv`; if omitted it is
computed internally. The `Lattice` method computes `bz` and `rlv` from `l`.

# Arguments
- `k`: 2-vector momentum to fold back.
- `bz`: polygon vertices of the first Brillouin zone (from [`get_bz`](@ref)).
- `rlv`: 2×2 matrix of reciprocal lattice vectors.
- `invrlv`: precomputed inverse of `rlv`.

See also [`get_bz`](@ref), [`reciprocal_lattice_vectors`](@ref).
"""
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
                    nij = SVector{2,Float64}(i, j)
                    d = norm(k - rlv * nij)
                    if d < min_d
                        n = nij
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
