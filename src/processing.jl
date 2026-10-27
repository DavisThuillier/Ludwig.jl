###
### Detailed-balance symmetrization and diagonalization
###

"""
    detailed_balance_factors(weights, energies, temperature) -> Vector{Float64}

Detailed-balance factors `Dᵢ = wᵢ·nᵢ·(1 - nᵢ)`, with `nᵢ = fermi_dirac(εᵢ, temperature)`, for full-BZ
quadrature `weights` and `energies`.
"""
function detailed_balance_factors(weights::AbstractVector, energies::AbstractVector, temperature)
    n = fermi_dirac.(energies, temperature)
    return weights .* n .* (1 .- n)
end

"""
    enforce_particle_conservation!(L, balance_factors) -> L

Set the diagonal of `L` so that particle number is conserved, where
`balance_factors[i] = Dᵢ = wᵢ·nᵢ·(1 - nᵢ)`.

Which condition that is depends on the shape of `L`. A stored block with more rows than columns
(`N × N_ibz`, the IBZ restricted to `kⱼ`) gets the detailed-balance column condition
`∑ᵢ Dᵢ L[i,j] = 0`. A square or wider `L` — the full `N × N` operator, or an `N_ibz × N` block with
the IBZ restricted to `kᵢ` — gets the unweighted row condition `∑ⱼ L[i,j] = 0`.

The correction is a decrement, so whatever `L` already carries on the diagonal cancels out of the
result and a second application changes nothing.
"""
function enforce_particle_conservation!(L::AbstractMatrix{Float64}, balance_factors::AbstractVector{Float64})
    N_rows, N_columns = size(L)
    use_row_sum = N_rows <= N_columns
    for j in 1:min(N_rows, N_columns)
        if use_row_sum
            L[j, j] -= sum(@view L[j, :])
        else
            L[j, j] -= dot(balance_factors, @view L[:, j]) / balance_factors[j]
        end
    end
    return L
end

###
### Point-group unfolding
###
###
### A full-BZ grid is `order(point_group)` contiguous blocks of `N_ibz` nodes, each block one group
### operation's image of the first. Which operation a positional block carries is fixed by whoever
### built the grid — `bz_quadrature_rule` uses the enumeration order of `operations`, which
### `close_group` derives from the generator list. Nothing stored alongside a collision matrix
### records that list, so the correspondence is either asserted by the caller or recovered from the
### node coordinates.
###

# Grid index of representative `r` in positional block `g`.
grid_index(g::Int, r::Int, N_ibz::Int) = (g - 1) * N_ibz + r

# True when `operation` carries every representative of block 1 onto the matching representative of
# block `g`.
function block_matches_operation(nodes, operation::SMatrix, g::Int, N_ibz::Int, atol::Float64)
    for r in 1:N_ibz
        norm(nodes[grid_index(g, r, N_ibz)] - operation * nodes[r]) ≤ atol || return false
    end
    return true
end

"""
    block_operation_map(nodes, point_group::PointGroup; atol) -> Vector{Int}

The operation each positional block of `nodes` carries: entry `g` is the index in
`operations(point_group)` of the operation taking block 1 onto block `g`.

Matched from the node coordinates, so the result does not depend on how `operations(point_group)`
enumerates the group; a point group rebuilt from a permuted generator list gives a permuted map that
describes the same geometry. Throws unless the blocks resolve to distinct operations.
"""
function block_operation_map(nodes, point_group::PointGroup; atol::Float64 = 1e-10)
    N = length(nodes)
    group_order = order(point_group)
    N > 0 || throw(ArgumentError("`nodes` is empty; the block layout needs at least one block."))
    iszero(N % group_order) || throw(DimensionMismatch(
        "$N nodes is not a multiple of the point-group order $group_order; `nodes` must be " *
        "$group_order equal blocks of point-group images of a common node set"))
    N_ibz = N ÷ group_order
    ops = operations(point_group)

    # Screening each block against the orbit of a single representative discards nearly every
    # candidate at one distance comparison; survivors are then confirmed on the whole block, which a
    # representative on a symmetry-fixed locus would otherwise let through under several operations.
    representative_orbit = [M * nodes[1] for M in ops]
    block_op = Vector{Int}(undef, group_order)
    for g in 1:group_order
        anchor = nodes[grid_index(g, 1, N_ibz)]
        found = 0
        for op_index in 1:group_order
            norm(representative_orbit[op_index] - anchor) ≤ atol || continue
            block_matches_operation(nodes, ops[op_index], g, N_ibz, atol) || continue
            found == 0 || throw(ArgumentError(
                "block $g of `nodes` is matched by both operation $found and operation $op_index; " *
                "the representatives do not determine the group action (all of them lie on a " *
                "common symmetry element)."))
            found = op_index
        end
        found == 0 && throw(ArgumentError(
            "no operation of the point group carries block 1 of `nodes` onto block $g; `nodes` is " *
            "not a point-group orbit of its first block, or its blocks have been reordered internally."))
        block_op[g] = found
    end
    allunique(block_op) || throw(ArgumentError(
        "distinct blocks of `nodes` carry the same point-group operation; `nodes` repeats an image " *
        "instead of covering the orbit."))
    return block_op
end

"""
    unfolding_tensor(point_group::PointGroup)
    unfolding_tensor(point_group::PointGroup, block_op::AbstractVector{Int})

Construct the tensor which unfolds an operator ``L_{ij}`` with i in the full space and j restricted to
the irreducible wedge of `point_group` into a full operator.

Assumes that `i` can be reshaped as `(ri, gi)` where `gi` indexes the images of the irreducible wedge under
`operations(point_group)` and `ri` is the representative index inside a given image. That is,

i--[L]--rj ==> ri--[L]--rj
                    |
                    gi

and the unfolding tensor `F` acts as
 ri--[L]--rj ==> i'--[L_full]--j
      |
gi'--[F]--gj

The one-argument form indexes all three group axes by group element, so it applies to a grid whose
positional block `g` holds the image under `operations(point_group)[g]`. Passing the `block_op` of
`block_operation_map` relabels those axes to positional block indices, which is the same tensor for a
grid in that canonical layout and the correct one for any other.

Each slice `F[:, :, g]` is `operation_permutation(g, table)`, the regular representation of `g`, and
is built from it. The index order follows the diagram above: the unfolded row block `gi'` first, then
the stored block `gi` that is contracted away, then the column block `gj`.
"""
function unfolding_tensor(point_group::PointGroup)
    multiplication_table = group_multiplication_table(point_group)
    group_order = order(point_group)
    F = zeros(Float64, group_order, group_order, group_order) # gi', gi, gj
    # F[gi', gi, gj] = 1 ⟺ gi' = gj·gi, the index form of L_gi',gj = L_gj⁻¹gi',e. That is exactly the
    # regular representation of gj, so each slice is one `operation_permutation` and the Cayley table
    # is walked in exactly one place.
    for gj ∈ 1:group_order
        F[:, :, gj] = operation_permutation(gj, multiplication_table)
    end
    return F
end

unfolding_tensor(point_group::PointGroup, block_op::AbstractVector{Int}) =
    unfolding_tensor(point_group)[block_op, block_op, block_op]

# The unfolding tensor in whichever index convention the stored block uses, together with the
# positional relabel it was built from (`nothing` when the caller asserts the canonical layout).
# `nodes` must span the whole grid, not just the representatives, for the blocks to be matchable.
function layout_unfolding_tensor(point_group::PointGroup, nodes, N::Int)
    nodes === nothing && return unfolding_tensor(point_group), nothing
    length(nodes) == N || throw(DimensionMismatch(
        "`nodes` has length $(length(nodes)) but the operator spans $N grid points"))
    block_op = block_operation_map(nodes, point_group)
    return unfolding_tensor(point_group, block_op), block_op
end

# Grid extent `N` and representatives per block `N_ibz` of a stored block in the orientation
# `ibz_index` declares, rejecting a shape that orientation cannot carry.
function stored_block_extents(L::AbstractMatrix, ibz_index::Symbol, group_order::Int)
    if ibz_index === :column
        N, stored_representatives = size(L)
        restricted_extent = "columns"
    elseif ibz_index === :row
        stored_representatives, N = size(L)
        restricted_extent = "rows"
    else
        throw(ArgumentError("ibz_index must be :column or :row, got :$ibz_index"))
    end

    stored_representatives * group_order == N || throw(DimensionMismatch(
        "L has $stored_representatives $restricted_extent, which cannot carry a grid of $N nodes in \
         $group_order point-group blocks"))
    return N, stored_representatives
end

"""
    full_collision_matrix(L, point_group; ibz_index, nodes) -> Matrix{Float64}

Unfold the stored IBZ block `L` into the full-BZ collision operator `L_full` (`N × N`) by
point-group invariance.

`ibz_index` names which momentum of `L` is restricted to the `N_ibz` IBZ representatives:
`:column` for `L[i, r] = L(kᵢ, k_r)` (`N × N_ibz`), `:row` for `L[r, j] = L(k_r, kⱼ)` (`N_ibz × N`).
`L` must have the shape its declared orientation implies.

`L` is laid out as `order(point_group)` contiguous blocks of `N_ibz` entries, each block holding the
image of the IBZ representatives under one group operation. By default block `g` is taken to carry
`operations(point_group)[g]`, so the point group must be built from the same generator list, in the
same order, as the one that produced the grid — `close_group` numbers the operations from that list,
and nothing stored alongside `L` records the numbering. Passing the full-BZ `nodes` recovers the
correspondence from the coordinates instead, which makes the result independent of that ordering and
rejects a layout that is not a point-group orbit at all.
"""
function full_collision_matrix(L::AbstractMatrix, point_group::PointGroup;
                               ibz_index::Symbol = :column, nodes = nothing)
    group_order = order(point_group)
    N, N_ibz = stored_block_extents(L, ibz_index, group_order)

    # Invariance gives L(kᵢ, M·k_r) = L(M⁻¹·kᵢ, k_r), so the unfolding is a permutation of the
    # stored block's group index alone; `unfolding_tensor` carries that permutation. Transposing a
    # point-group-invariant operator leaves it invariant, so `:row` is the same tensor with its two
    # block indices exchanged.
    F, _ = layout_unfolding_tensor(point_group, nodes, N)
    if ibz_index === :column
        stored = reshape(L, N_ibz, group_order, N_ibz)
        @tensor unfolded[ri, gi, rj, gj] := F[gi, g, gj] * stored[ri, g, rj]
    else
        stored = reshape(L, N_ibz, N_ibz, group_order)
        @tensor unfolded[ri, gi, rj, gj] := F[gj, g, gi] * stored[ri, rj, g]
    end

    return reshape(unfolded, N, N)
end

"""
    symmetrize_collision_matrix!(L, balance_factors) -> L

Replace the full-BZ collision operator `L` (`N × N`) with its detailed-balance symmetric part
`½(L + D⁻¹·Lᵀ·D)`, so that `Dᵢ·L[i,j] == Dⱼ·L[j,i]` to roundoff.

The map is a projection: applying it twice changes nothing, and the diagonal is a fixed point.
"""
function symmetrize_collision_matrix!(L::AbstractMatrix, balance_factors::AbstractVector)
    N = LinearAlgebra.checksquare(L)
    length(balance_factors) == N ||
        throw(DimensionMismatch("balance_factors has length $(length(balance_factors)), expected $N"))

    # One visit per unordered pair. The diagonal is skipped rather than averaged with itself.
    for j ∈ 1:N, i ∈ j+1:N
        Lij = L[i, j]
        Lji = L[j, i]
        L[i, j] = 0.5 * (Lij + Lji * balance_factors[j] / balance_factors[i])
        L[j, i] = 0.5 * (Lji + Lij * balance_factors[i] / balance_factors[j])
    end
    return L
end

"""
    block_constancy(balance_factors, N_ibz) -> Float64

Largest deviation of `balance_factors` from their values on the first block of `N_ibz`
representatives, relative to their largest magnitude.

The factors are constant over each point-group orbit whenever the quadrature weights are replicated
across blocks and the dispersion is point-group invariant.
"""
function block_constancy(balance_factors::AbstractVector, N_ibz::Int)
    deviation = maximum(i -> abs(balance_factors[i] - balance_factors[(i - 1) % N_ibz + 1]),
                        eachindex(balance_factors))
    return deviation / maximum(abs, balance_factors)
end

# Positional block carrying the inverse of each block's operation. Read off the unfolding tensor
# rather than the multiplication table so that it follows whatever relabelling `nodes` applied:
# `F[b, a, c] = 1` means `b = c·a`, so `F[1, a, c] = 1` names the `a` inverse to `c`, positional
# block 1 being the identity under every labelling `layout_unfolding_tensor` can produce.
inverse_block_map(F::AbstractArray{Float64,3}) =
    [findfirst(==(1.0), @view F[1, :, c]) for c ∈ axes(F, 3)]

# Detailed-balance symmetric part of a stored block, written into `symmetrized`. Both arrays carry
# the full-grid index on their first axis and the IBZ representative on their second, so a `:row`
# block reaches this through a transposed view of each; `ibz_axis_is_column` records which of the
# two representatives the operator sees on its column side, since the balance weight is `D` of the
# column-side representative over `D` of the row-side one and a transposed view does not reverse it.
function symmetrize_stored_block!(symmetrized::AbstractMatrix, stored::AbstractMatrix,
                                  inverse_block::AbstractVector{Int},
                                  ibz_balance_factors::AbstractVector, N_ibz::Int,
                                  group_order::Int, ibz_axis_is_column::Bool)
    for h ∈ 1:group_order, representative_ibz ∈ 1:N_ibz, representative_grid ∈ 1:N_ibz
        entry = grid_index(h, representative_grid, N_ibz)

        # Entries unfolding onto the full operator's diagonal keep their value, matching
        # `symmetrize_collision_matrix!`, which skips its diagonal. The remaining self-paired
        # entries — `h` an involution with the two representatives equal — are off-diagonal there
        # and are averaged, reproducing its arithmetic on them exactly.
        if h == 1 && representative_grid == representative_ibz
            symmetrized[entry, representative_ibz] = stored[entry, representative_ibz]
            continue
        end

        partner = grid_index(inverse_block[h], representative_ibz, N_ibz)
        column_representative, row_representative = ibz_axis_is_column ?
            (representative_ibz, representative_grid) : (representative_grid, representative_ibz)
        # Multiplying the partner before dividing matches `symmetrize_collision_matrix!`'s own
        # association, which is what makes the two placements agree bit for bit.
        symmetrized[entry, representative_ibz] = 0.5 * (stored[entry, representative_ibz] +
            stored[partner, representative_grid] *
            ibz_balance_factors[column_representative] / ibz_balance_factors[row_representative])
    end
    return symmetrized
end

"""
    symmetrized_collision_matrix(L, point_group, balance_factors; ibz_index, nodes, rtol) -> Matrix{Float64}

Collision operator ready to diagonalize or invert: detailed-balance symmetric in the sense of
[`symmetrize_collision_matrix!`](@ref) and carrying the conserved diagonal of
[`enforce_particle_conservation!`](@ref).

The result has the shape of the input. A stored IBZ block comes back as a stored block of the same
shape, symmetrized without ever forming the `N × N` operator — unfolding the result gives the same
matrix as unfolding `L` and symmetrizing that, so the cheaper placement costs nothing in accuracy. A
square `N × N` operator is symmetrized where it stands. `L` is never modified.

The order of the two steps is fixed. Symmetrization leaves the diagonal alone but changes the
off-diagonal sums, so enforcing first leaves particle number unconserved, while enforcing second
satisfies conservation and detailed balance together.

`ibz_index` and `nodes` are as in [`full_collision_matrix`](@ref) and apply to a stored block only.
`balance_factors` spans the full grid; for a stored block it also has to be constant over each
point-group orbit to within `rtol`, since symmetrizing with factors that vary within an orbit gives
an operator that satisfies detailed balance but is no longer point-group invariant, and no
downstream result reveals that.
"""
function symmetrized_collision_matrix(L::AbstractMatrix, point_group::PointGroup,
                                      balance_factors::AbstractVector;
                                      ibz_index::Symbol = :column, nodes = nothing,
                                      rtol::Real = 1e-8)
    ibz_index ∈ (:column, :row) ||
        throw(ArgumentError("ibz_index must be :column or :row, got :$ibz_index"))
    group_order = order(point_group)
    N_rows, N_columns = size(L)

    # A square operator spans the grid on both axes, so there is nothing to unfold and no invariance
    # for orbit-varying balance factors to break. An order-1 point group makes the two shapes
    # indistinguishable, and the paths then agree because `N_ibz == N`.
    if N_rows == N_columns
        N = N_rows
        length(balance_factors) == N || throw(DimensionMismatch(
            "balance_factors has length $(length(balance_factors)), expected $N to span the full grid"))
        symmetrized = Matrix{Float64}(L)
        symmetrize_collision_matrix!(symmetrized, balance_factors)
        enforce_particle_conservation!(symmetrized, balance_factors)
        return symmetrized
    end

    N, N_ibz = stored_block_extents(L, ibz_index, group_order)
    length(balance_factors) == N || throw(DimensionMismatch(
        "balance_factors has length $(length(balance_factors)), expected $N to span the full grid"))

    deviation = block_constancy(balance_factors, N_ibz)
    deviation ≤ rtol || throw(ArgumentError(
        "balance_factors vary by $deviation across a point-group orbit, above rtol = $rtol. " *
        "Symmetrizing with them would return a detailed-balanced operator that is no longer " *
        "point-group invariant. The factors follow from orbit-invariant energies and quadrature " *
        "weights replicated across blocks, so a spread this large means the grid, the weights or " *
        "the energies do not belong to this point group."))

    # Point-group invariance puts the transpose partner of a stored entry back inside the stored
    # block: `L_full[(gᵢ,rᵢ),(gⱼ,rⱼ)] = L[(gⱼ⁻¹gᵢ, rᵢ), rⱼ]`, so the partner of the entry labelled
    # `h` is the one labelled `h⁻¹` with its two representatives exchanged. Averaging those two
    # with the detailed-balance weight symmetrizes the whole operator while touching `N × N_ibz`
    # entries instead of `N²`. Block 1 holds the representatives themselves, so the first `N_ibz`
    # balance factors are the ones the representatives carry.
    F, _ = layout_unfolding_tensor(point_group, nodes, N)
    inverse_block = inverse_block_map(F)
    ibz_balance_factors = @view balance_factors[1:N_ibz]

    symmetrized = similar(L, Float64)
    if ibz_index === :column
        symmetrize_stored_block!(symmetrized, L, inverse_block, ibz_balance_factors,
                                 N_ibz, group_order, true)
    else
        symmetrize_stored_block!(transpose(symmetrized), transpose(L), inverse_block,
                                 ibz_balance_factors, N_ibz, group_order, false)
    end

    enforce_particle_conservation!(symmetrized, balance_factors)
    return symmetrized
end

"""
    collision_eigen(L_full, balance_factors) -> (eigenvalues, eigenvectors)

Eigenvalues and physical-basis eigenvectors of the full-BZ collision operator `L_full` (`N × N`),
obtained from the symmetric part of its detailed-balance form `S = D^{1/2}·L_full·D^{-1/2}`.
`L_full` is not modified; unfold a stored IBZ block with `full_collision_matrix` first.
"""
function collision_eigen(L_full::AbstractMatrix, balance_factors::AbstractVector)
    N = LinearAlgebra.checksquare(L_full)
    length(balance_factors) == N ||
        throw(DimensionMismatch("balance_factors has length $(length(balance_factors)), expected $N"))
    all(>(0), balance_factors) ||
        throw(ArgumentError("balance_factors must be strictly positive for D^{1/2} to be real."))

    # Scaling by the square roots keeps the dynamic range of the ratio at √(Dᵢ/Dⱼ); forming Dᵢ/Dⱼ
    # first and taking the root afterwards would square it.
    sqrtD = sqrt.(balance_factors)
    balanced_symmetric = Matrix{Float64}(undef, N, N)
    for j ∈ 1:N, i ∈ 1:N
        balanced = (sqrtD[i] / sqrtD[j]) * L_full[i, j]
        balanced_transpose = (sqrtD[j] / sqrtD[i]) * L_full[j, i]
        balanced_symmetric[i, j] = 0.5 * (balanced + balanced_transpose)
    end

    values, vectors = eigen(Symmetric(balanced_symmetric))
    vectors = vectors ./ sqrtD   # back to the physical basis: ψ = D^{-1/2}·φ
    return values, vectors
end

###
### Isotypic blocks
###

"""
    IsotypicBlocks(labels, blocks) <: AbstractVector

Blocks of an operator, one per irreducible representation, paired with the Mulliken symbol of the
representation each one carries. Indexable positionally in the order it was built with, and by
symbol: `blocks["A1"]`. `irrep_labels` returns the symbols in that same order.
"""
struct IsotypicBlocks{E} <: AbstractVector{E}
    labels::Vector{String}
    blocks::Vector{E}

    function IsotypicBlocks(labels::AbstractVector{<:AbstractString}, blocks::AbstractVector{E}) where {E}
        length(labels) == length(blocks) || throw(DimensionMismatch(
            "$(length(labels)) labels for $(length(blocks)) blocks"))
        allunique(labels) || throw(ArgumentError(
            "repeated Mulliken symbol among $(join(labels, ", ")); a symbol has to name one block"))
        return new{E}(convert(Vector{String}, labels), convert(Vector{E}, blocks))
    end
end

Base.size(blocks::IsotypicBlocks) = size(blocks.blocks)
Base.IndexStyle(::Type{<:IsotypicBlocks}) = IndexLinear()
Base.getindex(blocks::IsotypicBlocks, μ::Int) = blocks.blocks[μ]

function Base.getindex(blocks::IsotypicBlocks, label::AbstractString)
    μ = findfirst(isequal(label), blocks.labels)
    μ === nothing && throw(ArgumentError(
        "no block of irreducible representation $label; the blocks carry $(join(blocks.labels, ", "))"))
    return blocks.blocks[μ]
end

irrep_labels(blocks::IsotypicBlocks) = blocks.labels

function Base.show(io::IO, ::MIME"text/plain", blocks::IsotypicBlocks)
    println(io, "IsotypicBlocks of $(length(blocks)) irreducible representations")
    for (label, block) ∈ zip(blocks.labels, blocks.blocks)
        dimension = size(block.matrix, 1)
        println(io, "  ", rpad(label, 6), dimension, " × ", dimension)
    end
end

# `@show`, `print` and `string` reach the two-argument `show`; without this they fall back to the
# default field dump, which prints every block matrix.
Base.show(io::IO, blocks::IsotypicBlocks) = show(io, MIME("text/plain"), blocks)

symmetric_part(matrix) = Symmetric((matrix + transpose(matrix)) / 2)

"""
    isotypic_blocks(L, N_ibz, point_group, component_ranges, component_basis; nodes) -> Vector{NamedTuple}

Diagonal blocks of the full-BZ collision operator in the irreducible-representation basis
`component_basis`, built from the stored IBZ block `L` (`N × N_ibz`) without ever forming the `N × N`
operator. Each entry carries the `matrix` of one block and the `basis` spanning the invariant subspace
it lives on.

Pass what `irrep_basis` returns, or `component_ranges(table)` and `basis(table)` of a
`CharacterTable`. The blocks come back in the order of that decomposition, so they line up with its
Mulliken symbols and with nothing else: separating the isotypic components rests on a random draw of
class coefficients, and a second decomposition orders the representations differently.

`L` carries the block layout `full_collision_matrix` documents, and `nodes` lifts the same
generator-ordering assumption there: supplying the full-BZ nodes recovers the layout from the
coordinates and relabels `component_basis` to match, so the returned `basis` still expands onto the
grid in its stored node order.
"""
function isotypic_blocks(L::AbstractMatrix, N_ibz::Int, point_group::PointGroup,
                         component_ranges, component_basis::AbstractMatrix; nodes = nothing)
    group_order = order(point_group)
    size(L) == (N_ibz * group_order, N_ibz) || throw(DimensionMismatch(
        "L is $(size(L)), expected $((N_ibz * group_order, N_ibz)) for $N_ibz representatives and " *
        "an order-$group_order point group"))

    F, block_op = layout_unfolding_tensor(point_group, nodes, N_ibz * group_order)
    # `component_basis` is indexed by group element, so a relabelled tensor needs a relabelled basis.
    basis_matrix = block_op === nothing ? component_basis : component_basis[block_op, :]
    stored = reshape(L, N_ibz, group_order, N_ibz)

    return map(component_ranges) do component_range
        component = @view basis_matrix[:, component_range]
        component_dimension = length(component_range)
        @tensor begin
            projected[hi, gi, hj] := component[gi_prime, hi] * component[gj, hj] * F[gi_prime, gi, gj]
            block[ri, hi, rj, hj] := projected[hi, gi, hj] * stored[ri, gi, rj]
        end
        (; matrix = reshape(block, component_dimension * N_ibz, component_dimension * N_ibz),
           basis = component)
    end
end

"""
    isotypic_blocks(L, N_ibz, table::CharacterTable; nodes) -> IsotypicBlocks

Diagonal blocks of the full-BZ collision operator in the irreducible-representation basis `table` was
read from, one per row of the table. Reachable positionally in the table's row order, so
`irrep_labels(table)[μ]` names block `μ`, and by Mulliken symbol: `blocks["A1"]`.

Taking the decomposition from the table is what makes the blocks nameable: the ranges and the basis
have to come from the same call to `irrep_basis`, and a second decomposition orders the
representations differently.
"""
isotypic_blocks(L::AbstractMatrix, N_ibz::Int, table::CharacterTable; nodes = nothing) =
    IsotypicBlocks(irrep_labels(table),
                   isotypic_blocks(L, N_ibz, point_group(table), component_ranges(table),
                                   basis(table); nodes))

"""
    block_modes(block, N_ibz, sqrtD) -> (eigenvalues, eigenvectors)

Ascending eigenvalues and physical-basis eigenvectors of one isotypic `block` of the
detailed-balance form of the collision operator. Column `m` of `eigenvectors` is a field over the
full-BZ grid in its stored node order, so it indexes against the grid directly.

`sqrtD` is the elementwise square root of the detailed-balance factors over the full grid; it is
taken already rooted so that a sweep over every block roots them once rather than once per block.
"""
function block_modes(block, N_ibz::Int, sqrtD::AbstractVector)
    values, coefficients = eigen(symmetric_part(block.matrix))

    # Block and grid both run the IBZ representative fastest, so expanding the basis restores the
    # node ordering with no permutation.
    component_basis = block.basis
    component_dimension = size(component_basis, 2)
    coefficients = reshape(coefficients, N_ibz, component_dimension, size(coefficients, 2))
    @tensor expanded[ri, gi, m] := coefficients[ri, hi, m] * component_basis[gi, hi]

    vectors = reshape(expanded, length(sqrtD), size(expanded, 3))
    return values, vectors ./ sqrtD   # back to the physical basis: ψ = D^{-1/2}·φ
end
