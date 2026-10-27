###
### Shared readers for the files the plotting scripts consume
###
### Two formats. A collision HDF5 file holds a whole run: electron_electron.jl writes `L`, `nodes`,
### `weights`, `energies`, `D`, the `lattice` group and the `provenance` attributes,
### diagonalize_collision_operator.jl later adds `eigenvalues` and `eigenvectors`, and irrep_test.jl
### caches `L_irrep_basis`, all in that same file. A `.quad` rule from generate_quadrature_rule.jl is
### plain text: a `#` header of `key = value` build parameters, then one `k…, w` row per node.
###
### The loaders cover the field sets the plotting scripts need. Any other combination of HDF5
### datasets can be assembled from the `read_*` helpers, which take an already-open file and so
### compose inside a single `h5open` block.
###

using HDF5
using StaticArrays

###
### Dataset helpers
###

"""
    required_dataset(file, name, producer) -> Any

Dataset `name` of an open collision `file`. Errors naming the `producer` script when it is absent.
"""
function required_dataset(file, name, producer)
    haskey(file, name) ||
        error("$(HDF5.filename(file)) has no `$name` dataset; run $producer on it first")
    return read(file, name)
end

"""
    read_optional(file, name) -> Union{Vector{Float64},Nothing}

Dataset `name` of an open collision `file`, or `nothing` for files written before it existed. Covers
the per-node band energies (`energies`) and detailed-balance factors (`D`).
"""
read_optional(file, name) = haskey(file, name) ? Vector{Float64}(read(file, name)) : nothing

"""
    read_nodes(file) -> Vector{SVector{D,Float64}}

Cartesian quadrature nodes of an open collision `file`.
"""
function read_nodes(file)
    node_matrix = Matrix{Float64}(required_dataset(file, "nodes", "electron_electron.jl"))
    D = size(node_matrix, 1)
    return [SVector{D,Float64}(view(node_matrix, :, i)) for i ∈ axes(node_matrix, 2)]
end

"""
    read_lattice(file) -> (; primitives, generators)

Lattice primitives and Cartesian point-group generators of an open collision `file`, or `nothing` for
files written before the lattice group existed.
"""
function read_lattice(file)
    haskey(file, "lattice") || return nothing

    primitive_matrix = Matrix{Float64}(read(file, "lattice/primitives"))
    generator_stack  = Array{Float64,3}(read(file, "lattice/generators"))

    D = size(primitive_matrix, 1)
    primitives = SMatrix{D,D,Float64}(primitive_matrix)
    generators = [SMatrix{D,D,Float64}(view(generator_stack, :, :, g)) for g ∈ axes(generator_stack, 3)]

    return (; primitives, generators)
end

"""
    read_provenance(file) -> NamedTuple

Every attribute of the `provenance` group of an open collision `file`, keyed by attribute name.
"""
function read_provenance(file)
    haskey(file, "provenance") ||
        error("$(HDF5.filename(file)) has no `provenance` group; run electron_electron.jl on it first")

    # `attributes` is qualified because Makie exports the name too, and the plotting scripts have both
    # in scope.
    group = file["provenance"]
    names = keys(HDF5.attributes(group))
    return (; (Symbol(name) => HDF5.read_attribute(group, name) for name ∈ names)...)
end

###
### Loaders
###

"""
    load_spectrum(path) -> (; eigenvalues, provenance)

Collision-operator eigenvalues and provenance attributes of the collision file at `path`.
"""
function load_spectrum(path)
    h5open(path, "r") do file
        eigenvalues = Vector{Float64}(
            required_dataset(file, "eigenvalues", "diagonalize_collision_operator.jl"))
        return (; eigenvalues, provenance = read_provenance(file))
    end
end

"""
    load_eigenmodes(path) -> (; eigenvalues, eigenvectors, nodes, energies, D, lattice, provenance)

Collision-operator spectrum, quadrature nodes with their band energies and detailed-balance factors,
and lattice and provenance metadata of the collision file at `path`. Column `i` of `eigenvectors`
belongs to `eigenvalues[i]`.
"""
function load_eigenmodes(path)
    h5open(path, "r") do file
        eigenvalues = Vector{Float64}(
            required_dataset(file, "eigenvalues", "diagonalize_collision_operator.jl"))
        eigenvectors = Matrix{Float64}(
            required_dataset(file, "eigenvectors", "diagonalize_collision_operator.jl"))
        return (; eigenvalues, eigenvectors, nodes = read_nodes(file),
                  energies = read_optional(file, "energies"), D = read_optional(file, "D"),
                  lattice = read_lattice(file), provenance = read_provenance(file))
    end
end

"""
    load_collision_operator(path) -> (; L, nodes, weights, energies, D, lattice, provenance)

IBZ block `L` of the collision matrix, the quadrature nodes with their weights, band energies and
detailed-balance factors, and the lattice and provenance metadata of the collision file at `path`.
"""
function load_collision_operator(path)
    h5open(path, "r") do file
        L = Matrix{Float64}(required_dataset(file, "L", "electron_electron.jl"))
        weights = Vector{Float64}(required_dataset(file, "weights", "electron_electron.jl"))
        return (; L, nodes = read_nodes(file), weights,
                  energies = read_optional(file, "energies"), D = read_optional(file, "D"),
                  lattice = read_lattice(file), provenance = read_provenance(file))
    end
end

###
### Quadrature rules
###

"""
    header_value(text) -> Union{Int,Float64,String}

Header value `text` parsed as the narrowest of `Int`, `Float64` or `String` that fits.
"""
function header_value(text)
    integer_value = tryparse(Int, text)
    integer_value === nothing || return integer_value

    real_value = tryparse(Float64, text)
    real_value === nothing || return real_value

    return String(text)
end

"""
    header_entry(line) -> Union{Pair{Symbol},Nothing}

The `key = value` pair carried by a `#` header `line`, or `nothing` for the title and column-label
lines that carry none. Units are dropped from the key, so `# T [K] = 40.0` yields `:T => 40.0`.
"""
function header_entry(line)
    parts = split(lstrip(lstrip(line, '#')), "="; limit = 2)
    length(parts) == 2 || return nothing

    name = strip(first(split(first(parts), "[")))
    return Symbol(name) => header_value(strip(last(parts)))
end

"""
    load_quadrature_rule(path) -> (; nodes, weights, header)

Cartesian nodes and quadrature weights of the `.quad` rule at `path`, together with every `key =
value` pair of its header. The dimension follows the file's column count. The header's
`material_file` is relative to the directory the rule was generated in, so only its basename is
meaningful; the caller resolves it against `scripts/materials/`.
"""
function load_quadrature_rule(path)
    header = Pair{Symbol,Any}[]
    rows = Vector{Float64}[]

    for line ∈ eachline(path)
        stripped = strip(line)
        if isempty(stripped)
            continue
        elseif startswith(stripped, "#")
            entry = header_entry(stripped)
            entry === nothing || push!(header, entry)
            continue
        end
        push!(rows, parse.(Float64, split(stripped, ",")))
    end

    isempty(rows) && error("$path has no quadrature nodes; run generate_quadrature_rule.jl on it first")

    D = length(first(rows)) - 1
    nodes = [SVector{D,Float64}(view(row, 1:D)) for row ∈ rows]
    weights = [last(row) for row ∈ rows]

    return (; nodes, weights, header = (; header...))
end
