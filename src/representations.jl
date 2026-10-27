###
### Class sums
###

"""
    class_sums(multiplication_table[, classes]) -> Vector{Matrix{Int}}

Sum of the regular-representation permutation matrices of each conjugacy class. These span the centre
of the group algebra, so they commute and share a block decomposition.

Pass `classes` to reuse a `conjugacy_classes` partition already at hand; the result is indexed to
match it.
"""
class_sums(multiplication_table::Matrix{Int}) =
    class_sums(multiplication_table, conjugacy_classes(multiplication_table))

class_sums(multiplication_table::Matrix{Int}, classes) =
    [sum(operation_permutation(g, multiplication_table) for g in class) for class in classes]

class_sums(point_group::PointGroup) = class_sums(group_multiplication_table(point_group))

"""
    constant_ranges(values) -> Vector{UnitRange{Int}}

Index ranges over which the sorted vector `values` holds a single value.
"""
function constant_ranges(values)
    ranges = UnitRange{Int}[]
    isempty(values) && return ranges

    start = firstindex(values)
    for i ∈ (start + 1):lastindex(values)
        values[i] == values[start] && continue
        push!(ranges, start:(i - 1))
        start = i
    end
    push!(ranges, start:lastindex(values))

    return ranges
end

"""
    real_irrep_count(multiplication_table) -> Int
    real_irrep_count(classes, inverses) -> Int

Number of real irreducible representations of the group: its conjugacy classes counted up to the
pairing `C ↔ C⁻¹`. This equals the class count exactly when every class is closed under inversion; in
any other group each pair of mutually inverse classes carries a single real representation built from
a conjugate pair of complex ones.

Pass `classes` and `inverses` to reuse a partition and an `inverse_table` already at hand.
"""
function real_irrep_count(multiplication_table::Matrix{Int})
    inverses = inverse_table(multiplication_table)
    return real_irrep_count(conjugacy_classes(multiplication_table, inverses), inverses)
end

function real_irrep_count(classes, inverses::Vector{Int})
    class_of = zeros(Int, length(inverses))
    for (i, class) in enumerate(classes), g in class
        class_of[g] = i
    end

    counted = falses(length(classes))
    total = 0
    for (i, class) in enumerate(classes)
        counted[i] && continue
        counted[i] = true
        counted[class_of[inverses[first(class)]]] = true
        total += 1
    end
    return total
end

###
### Real irreducible decomposition of the regular representation
###

"""
    irrep_basis(point_group; N_restarts = 20) -> (component_ranges, basis)

Orthogonal basis of the regular representation in which it is block diagonal, one block per real
irreducible representation, together with the index range each block spans.
"""
function irrep_basis(point_group::PointGroup; N_restarts = 20)
    multiplication_table = group_multiplication_table(point_group)
    inverses = inverse_table(multiplication_table)
    classes = conjugacy_classes(multiplication_table, inverses)
    return irrep_basis(class_sums(multiplication_table, classes),
                       real_irrep_count(classes, inverses); N_restarts)
end

function irrep_basis(class_sum_operators, N_real_irreps::Int; N_restarts = 20)
    N_classes = length(class_sum_operators)

    for _ in 1:N_restarts
        # Integer coefficients keep the spectrum integral, so eigenvalues group exactly. Adding the
        # transpose pairs each class with its inverse, which is what keeps the eigenvalues real for
        # the groups whose bare class sums are asymmetric.
        coefficients = rand(1:(4 * N_classes^2), N_classes)
        combination = sum(coefficients .* class_sum_operators)
        symmetrized = Symmetric(Float64.(combination + transpose(combination)))

        eigenvalues, basis = eigen(symmetrized)   # Ascending eigenvalues, orthogonal basis
        maximum(abs(value - round(value)) for value in eigenvalues) < 1e-6 || continue

        component_ranges = constant_ranges(round.(Int, eigenvalues))
        # A class-sum combination lies in the centre of the group algebra, so every eigenspace is a
        # union of isotypic components and the count can never exceed `N_real_irreps`. Reaching it
        # therefore proves the components are separated one to a block.
        length(component_ranges) == N_real_irreps && return component_ranges, basis
    end

    error("no separating class-sum combination found after $N_restarts restarts; the group has " *
          "$N_real_irreps real irreducible representations and every draw left two of them sharing " *
          "an eigenvalue")
end

"""
    irrep_character_table(classes, class_sum_operators, component_ranges, basis, group_order)
        -> (characters, reality)

Character of every real irreducible representation on every conjugacy class, and whether each is
irreducible over the complex numbers (`:real`) or the sum of a conjugate pair (`:complex_pair`).
"""
function irrep_character_table(classes, class_sum_operators, component_ranges, basis, group_order)
    class_sizes = length.(classes)
    characters = Matrix{Float64}(undef, length(component_ranges), length(classes))
    reality = Vector{Symbol}(undef, length(component_ranges))

    for (μ, range) in enumerate(component_ranges)
        representative = @view basis[:, first(range)]
        class_sum_eigenvalues = map(class_sum_operators) do class_sum
            eigenvalue = dot(representative, class_sum * representative) / dot(representative, representative)
            is_integer(eigenvalue) ? round(eigenvalue) : eigenvalue
        end

        dimension = sqrt(group_order / sum(class_sum_eigenvalues .^ 2 ./ class_sizes))
        # A conjugate pair of complex representations merges into one real block whose character is
        # `χ + conj(χ)`. The Rayleigh quotient returns `Re χ`, which scales the dimension by √2 and so
        # leaves it short of an integer — the only signature that tells the two cases apart.
        complex_pair = !is_integer(dimension)
        scale = complex_pair ? dimension * sqrt(2) : round(dimension)

        for i in eachindex(classes)
            value = scale * class_sum_eigenvalues[i] / class_sizes[i]
            characters[μ, i] = is_integer(value) ? round(value) : value
        end
        reality[μ] = complex_pair ? :complex_pair : :real
    end

    return characters, reality
end

###
### Mulliken symbols
###

# The labeller reasons in three dimensions. A planar operation lifted with `z → z` becomes a rotation
# about, or a mirror containing, the out-of-plane axis, so every geometric rule carries over
# unchanged. A lifted planar group can hold neither inversion nor a horizontal mirror, which is why a
# `PointGroup{2}` comes back with bare symbols: it has no inversion centre, `-I` being a proper half
# turn in the plane.
lift_operation(g::SMatrix{3,3,Float64}) = g
lift_operation(g::SMatrix{2,2,Float64}) =
    SMatrix{3,3,Float64}(g[1, 1], g[2, 1], 0, g[1, 2], g[2, 2], 0, 0, 0, 1)

parallel_axes(first_axis, second_axis) = norm(cross(first_axis, second_axis)) < 1e-6
perpendicular_axes(first_axis, second_axis) = abs(dot(first_axis, second_axis)) < 1e-6

function unique_axes(directions)
    distinct = SVector{3,Float64}[]
    for direction in directions
        direction === nothing && continue
        any(seen -> norm(seen - direction) < 1e-6, distinct) || push!(distinct, direction)
    end
    return distinct
end

# The in-plane direction the `B` subscript is measured against: the reference projected perpendicular
# to the principal axis, falling back to another coordinate direction when the two are parallel.
function in_plane_direction(reference::SVector{3,Float64}, principal)
    principal === nothing && return nothing
    for candidate in (reference, SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0))
        component = candidate - dot(candidate, principal) * principal
        norm(component) > 1e-6 && return component / norm(component)
    end
    return nothing
end

"""
    mulliken_labels(point_group, classes, characters; reference_direction) -> Vector{String}

Mulliken symbol of each real irreducible representation, given its `characters` on the conjugacy
`classes` of `point_group`.

Every part of the symbol but the `A`/`B` subscript follows from the characters alone. That subscript
needs an in-plane direction, taken from `reference_direction` (Cartesian `x` by default): the
reference class is the one holding an operation that fixes it, among classes whose axis is
perpendicular to the principal axis, preferring a proper rotation to a mirror.
"""
function mulliken_labels(point_group::PointGroup, classes, characters;
                         reference_direction = SVector(1.0, 0.0, 0.0))
    ops = [lift_operation(g) for g in operations(point_group)]
    kinds = [operation_kind(ops[first(class)]) for class in classes]
    proper = first.(kinds)
    orders = last.(kinds)
    class_axes = [operation_axis(ops[first(class)]) for class in classes]

    identity_class = findfirst(class -> first(class) == 1, classes)
    inversion_class = findfirst(i -> !proper[i] && orders[i] == 2, eachindex(classes))

    rotation_classes = [i for i in eachindex(classes) if proper[i] && orders[i] > 1]
    highest_order = isempty(rotation_classes) ? 1 : maximum(orders[i] for i in rotation_classes)
    highest_axes = unique_axes(operation_axis(ops[g]) for i in rotation_classes
                               if orders[i] == highest_order for g in classes[i])

    # A cubic group has several axes of its highest order and so no principal axis; its symbols are
    # keyed against the four-fold class instead.
    cubic = highest_order >= 3 && length(highest_axes) > 1
    principal = if cubic
        nothing
    elseif !isempty(highest_axes)
        first(highest_axes)
    else
        rotoreflection = findfirst(i -> !proper[i] && orders[i] >= 4, eachindex(classes))
        rotoreflection === nothing ? nothing : class_axes[rotoreflection]
    end

    # Three equivalent two-fold axes leave no principal axis either, and Mulliken numbers the three
    # non-symmetric representations after the axes taken in the order z, y, x.
    two_fold = [i for i in eachindex(classes) if proper[i] && orders[i] == 2]
    orthorhombic = !cubic && highest_order == 2 &&
                   length(unique_axes(operation_axis(ops[g]) for i in two_fold for g in classes[i])) == 3

    principal_class = principal_reference(classes, proper, orders, class_axes,
                                          identity_class, principal, cubic)
    secondary_class = secondary_reference(ops, classes, proper, class_axes, identity_class,
                                          principal, cubic, orthorhombic,
                                          in_plane_direction(reference_direction, principal),
                                          principal_class)
    horizontal_mirror = principal === nothing ? nothing :
        findfirst(i -> !proper[i] && orders[i] == 1 && class_axes[i] !== nothing &&
                       parallel_axes(class_axes[i], principal), eachindex(classes))

    ordered_two_fold = sort(two_fold; by = i -> -argmax(abs.(class_axes[i])))

    labels = Vector{String}(undef, size(characters, 1))
    for μ in axes(characters, 1)
        dimension = round(Int, characters[μ, identity_class])
        base = if dimension == 3
            "T"
        elseif dimension != 1
            "E"
        elseif cubic || principal_class == 0 || characters[μ, principal_class] > 0
            "A"
        else
            "B"
        end

        subscript = ""
        if orthorhombic && dimension == 1
            positive = findall(i -> characters[μ, i] > 0, ordered_two_fold)
            if length(positive) < length(ordered_two_fold)
                base = "B"
                subscript = string(first(positive))
            end
        elseif secondary_class != 0 && base in ("A", "B", "T")
            subscript = characters[μ, secondary_class] > 0 ? "1" : "2"
        end

        suffix = if inversion_class !== nothing
            characters[μ, inversion_class] > 0 ? "g" : "u"
        elseif horizontal_mirror !== nothing
            characters[μ, horizontal_mirror] > 0 ? "'" : "''"
        else
            ""
        end

        labels[μ] = base * subscript * suffix
    end

    number_repeated_labels!(labels, characters, principal_class)
    return labels
end

# The class of the highest-order operation about the principal axis; `A` and `B` are told apart by its
# character. A proper rotation wins a tie so that D₆ₕ is keyed against C₆ rather than S₆, and -4
# against S₄ rather than C₂.
function principal_reference(classes, proper, orders, class_axes, identity_class, principal, cubic)
    if cubic
        four_fold = findfirst(i -> proper[i] && orders[i] == 4, eachindex(classes))
        four_fold === nothing &&
            (four_fold = findfirst(i -> !proper[i] && orders[i] == 4, eachindex(classes)))
        return four_fold === nothing ? 0 : four_fold
    end
    principal === nothing && return 0

    candidates = [i for i in eachindex(classes)
                  if i != identity_class && class_axes[i] !== nothing &&
                     parallel_axes(class_axes[i], principal)]
    isempty(candidates) && return 0

    best = maximum(orders[i] for i in candidates)
    pool = [i for i in candidates if orders[i] == best]
    proper_pool = filter(i -> proper[i], pool)
    return first(isempty(proper_pool) ? pool : proper_pool)
end

# The class the `A`/`B` subscript is measured against: the one fixing the in-plane reference
# direction. Restricting to axes perpendicular to the principal axis drops the horizontal mirror, and
# preferring a proper rotation matters — since `σ_n = i·C2(n)`, a mirror reference flips the subscript
# between a representation and its inversion partner.
function secondary_reference(ops, classes, proper, class_axes, identity_class, principal, cubic,
                             orthorhombic, in_plane, principal_class)
    cubic && return principal_class
    (orthorhombic || principal === nothing || in_plane === nothing) && return 0

    fixing = [i for i in eachindex(classes)
              if i != identity_class && class_axes[i] !== nothing &&
                 perpendicular_axes(class_axes[i], principal) &&
                 any(g -> norm(ops[g] * in_plane - in_plane) < 1e-8, classes[i])]
    isempty(fixing) && return 0

    proper_fixing = filter(i -> proper[i], fixing)
    return first(isempty(proper_fixing) ? fixing : proper_fixing)
end

# Representations that every rule above leaves indistinguishable are the `E1`, `E2` … family, numbered
# by their character on the principal rotation.
function number_repeated_labels!(labels, characters, principal_class)
    principal_class == 0 && return labels
    for label in unique(labels)
        repeated = findall(isequal(label), labels)
        length(repeated) == 1 && continue
        for (k, μ) in enumerate(sort(repeated; by = μ -> -characters[μ, principal_class]))
            labels[μ] = label[1:1] * string(k) * label[nextind(label, 1):end]
        end
    end
    return labels
end

"""
    mulliken_sort_key(label) -> (parity, base letter, subscript)

Sort key placing Mulliken symbols in the order published character tables print them: the gerade or
unprimed representations first, then the ungerade or primed ones, and within each block by base
letter and then by ascending subscript.

Alphabetical base letters coincide with non-decreasing dimension (`A`, `B` at 1, `E` at 2, `T` at 3),
so the two readings of the convention never disagree.
"""
function mulliken_sort_key(label::AbstractString)
    # Every symbol is ASCII — a base letter, an optional subscript, then an optional parity suffix —
    # so it indexes by integer position.
    subscript = 0
    position = 2
    while position <= length(label) && isdigit(label[position])
        subscript = 10 * subscript + (label[position] - '0')
        position += 1
    end
    suffix = label[position:end]
    return ((suffix == "u" || suffix == "''") ? 1 : 0, label[1], subscript)
end

###
### Character table
###

"""
    CharacterTable{D}

Conjugacy classes of a `PointGroup{D}` with their Schoenflies symbols, the character of every real
irreducible representation on each of them, the Mulliken symbol of each representation, and the basis
of the regular representation the characters were read from.
"""
struct CharacterTable{D}
    point_group::PointGroup{D}
    classes::Vector{Vector{Int}}
    class_symbols::Vector{String}
    characters::Matrix{Float64}
    irrep_labels::Vector{String}
    reality::Vector{Symbol}
    component_ranges::Vector{UnitRange{Int}}
    basis::Matrix{Float64}
end

point_group(table::CharacterTable) = table.point_group
conjugacy_classes(table::CharacterTable) = table.classes
class_symbols(table::CharacterTable) = table.class_symbols
characters(table::CharacterTable) = table.characters
irrep_labels(table::CharacterTable) = table.irrep_labels
reality(table::CharacterTable) = table.reality
component_ranges(table::CharacterTable) = table.component_ranges
basis(table::CharacterTable) = table.basis

"""
    character_table(point_group; reference_direction, N_restarts = 20) -> CharacterTable

Character table of `point_group`, with each real irreducible representation carrying its Mulliken
symbol.

Rows follow the order published tables print, the one `mulliken_sort_key` defines. Columns are in the
order the conjugacy classes were discovered, which carries no meaning on its own — read them through
`class_symbols`.
"""
function character_table(point_group::PointGroup{D};
                         reference_direction = SVector(1.0, 0.0, 0.0),
                         N_restarts = 20) where {D}
    multiplication_table = group_multiplication_table(point_group)
    inverses = inverse_table(multiplication_table)
    classes = conjugacy_classes(multiplication_table, inverses)
    class_sum_operators = class_sums(multiplication_table, classes)
    component_ranges, basis = irrep_basis(class_sum_operators,
                                          real_irrep_count(classes, inverses); N_restarts)
    chars, reality_flags = irrep_character_table(classes, class_sum_operators, component_ranges,
                                                 basis, order(point_group))
    labels = mulliken_labels(point_group, classes, chars; reference_direction)
    symbols = [class_symbol(class, operations(point_group)) for class in classes]

    # The decomposition comes back sorted by a random draw of class-sum eigenvalues. Reordering the
    # rows makes the table readable against a published one; `basis` is untouched because
    # `component_ranges` indexes into it by column and stays correct under any row order.
    published = sortperm(labels; by = mulliken_sort_key)

    return CharacterTable{D}(point_group, classes, symbols, chars[published, :], labels[published],
                             reality_flags[published], component_ranges[published], basis)
end

character_table(crystal::Crystal; kwargs...) = character_table(point_group(crystal); kwargs...)

function Base.show(io::IO, ::MIME"text/plain", table::CharacterTable{D}) where {D}
    println(io, "CharacterTable{$D} of an order-$(order(table.point_group)) point group")
    println(io, "  ", rpad("", 6), join(rpad.(table.class_symbols, 6)))
    for μ in eachindex(table.irrep_labels)
        merged = table.reality[μ] === :complex_pair ? "  (conjugate pair)" : ""
        # Every character of a crystallographic point group is an integer, so the table prints them
        # as such — which also disposes of the signed zeros the eigendecomposition leaves behind.
        # The stored matrix stays Float64 because the projections do real arithmetic with it.
        println(io, "  ", rpad(table.irrep_labels[μ], 6),
                join(rpad(round(Int, character), 6) for character in view(table.characters, μ, :)),
                merged)
    end
end

# `@show`, `print` and `string` reach the two-argument `show`; without this they fall back to the
# default field dump, which prints the whole regular-representation basis.
Base.show(io::IO, table::CharacterTable) = show(io, MIME("text/plain"), table)

###
### Projection of a field onto the irreducible representations
###

# The group acts on a scalar field as `(g·f)(k) = f(g⁻¹k)`, which sampling at `ops[a]·k` carries to
# left translation on the sampled values. Handedness is inert for everything below: each projects onto
# a whole isotypic component, and that component is a two-sided ideal of the group algebra, so left
# and right translation single out the same subspace. It becomes load bearing again only for something
# that resolves the partner functions inside a representation.

# Seeds with mutually irrational components, so no operation but the identity fixes one and each orbit
# is a faithful copy of the group. That makes the sampled inner product group invariant, hence the
# projections orthogonal and their weights a partition of unity.
function projection_seeds(::Val{D}, count::Int) where {D}
    base = (Float64(π) - 3, Float64(ℯ) - 2, sqrt(2.0) - 1)
    step = (sqrt(3.0) - 1, sqrt(5.0) - 2, sqrt(7.0) - 2)
    return [SVector{D,Float64}(ntuple(i -> base[i] + offset * step[i], Val(D)))
            for offset in 0:(count - 1)]
end

"""
    irrep_content(table, field, nodes, weights) -> Vector{Float64}

Fraction of the weighted square norm `∑ᵢ wᵢ·fᵢ²` of a scalar `field` over the full-BZ grid `nodes`
carried by each irreducible representation of `table`, in the row order of the table.

`nodes` is `order(point_group(table))` contiguous blocks of a common set of representatives, each
block one operation's image of the first, and `weights` are the quadrature weights of the same grid.
Which operation a positional block carries is recovered from the coordinates, so the result does not
depend on how the point group enumerates its elements.

The shares partition the field only when `weights` are constant over each point-group orbit, which
the full-BZ rule satisfies by replicating the IBZ weights across blocks.
"""
function irrep_content(table::CharacterTable, field::AbstractVector{<:Real}, nodes,
                       weights::AbstractVector{<:Real})
    symmetry_group = point_group(table)
    group_order = order(symmetry_group)
    N = length(field)
    length(nodes) == N || throw(DimensionMismatch(
        "`field` holds $N values but `nodes` holds $(length(nodes)) points"))
    length(weights) == N || throw(DimensionMismatch(
        "`field` holds $N values but `weights` holds $(length(weights)) entries"))
    iszero(N % group_order) || throw(DimensionMismatch(
        "$N values is not a multiple of the point-group order $group_order; `field` must span " *
        "$group_order equal blocks of point-group images of a common node set"))
    N_ibz = N ÷ group_order

    # A grid point is `k_{r,h} = M_h·k_r`, so `(g·f)(k_{r,h}) = f(k_{r,g⁻¹h})`: the group permutes the
    # block index alone and carries the representative index along untouched. The field therefore
    # decomposes on the block axis of `[representative, block]` against the same regular-representation
    # basis the table was read from, relabelled to positional blocks so it indexes the grid directly.
    basis_matrix = basis(table)[block_operation_map(nodes, symmetry_group), :]
    orbit_values = reshape(field, N_ibz, group_order)
    orbit_weights = reshape(weights, N_ibz, group_order)

    total = sum(weight * value^2 for (weight, value) in zip(orbit_weights, orbit_values))
    total > 0 || throw(ArgumentError("the field has zero weighted norm over the grid"))

    return map(component_ranges(table)) do range
        component = @view basis_matrix[:, range]
        projected = (orbit_values * component) * transpose(component)
        return sum(orbit_weights[i] * projected[i]^2 for i in eachindex(projected)) / total
    end
end

"""
    irrep_content(table, f; seeds = 3) -> Vector{Float64}

Fraction of the scalar field `f` carried by each irreducible representation of `table`, sampled on
the orbits of a few generic points. `f` takes a Cartesian momentum and returns a number.
"""
function irrep_content(table::CharacterTable{D}, f; seeds::Int = 3) where {D}
    points = projection_seeds(Val(D), seeds)
    nodes = [M * point for M in operations(point_group(table)) for point in points]
    field = [Float64(f(node)) for node in nodes]
    # Synthetic orbits carry no measure of their own, so every sample counts equally.
    return irrep_content(table, field, nodes, ones(length(field)))
end

# The content of a field carried by one representation is a unit vector on that representation, so a
# representation counts as carried once its share of the field exceeds `atol`. Normalizing first makes
# that threshold meaningful even for a rule whose weights are not constant over each point-group orbit,
# where the shares do not add to one on their own.
function dominant_irrep(table::CharacterTable, content::AbstractVector{<:Real}, atol)
    normalized = content ./ sum(content)
    carried = findall(>(atol), normalized)
    length(carried) == 1 || throw(ArgumentError(
        "the field spreads over $(irrep_labels(table)[carried]) with weights " *
        "$(round.(normalized[carried]; digits = 6)); `which_irrep` needs a field carried by one " *
        "representation"))
    return irrep_labels(table)[only(carried)]
end

"""
    which_irrep(table, field, nodes, weights; atol = 1e-6) -> String

Mulliken symbol of the irreducible representation carrying a scalar `field` over the full-BZ grid
`nodes`. Throws when the field spreads over more than one.
"""
which_irrep(table::CharacterTable, field::AbstractVector{<:Real}, nodes,
            weights::AbstractVector{<:Real}; atol = 1e-6) =
    dominant_irrep(table, irrep_content(table, field, nodes, weights), atol)

"""
    which_irrep(table, f; seeds = 3, atol = 1e-6) -> String

Mulliken symbol of the irreducible representation carrying the scalar field `f`. Throws when `f`
spreads over more than one.
"""
which_irrep(table::CharacterTable, f; seeds::Int = 3, atol = 1e-6) =
    dominant_irrep(table, irrep_content(table, f; seeds), atol)
