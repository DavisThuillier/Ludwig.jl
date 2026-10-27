###
### Point-group representation theory
###
#
# Covers src/representations.jl — the real irreducible decomposition of the regular
# representation, character tables and their Mulliken symbols — together with the two
# consumers of that decomposition in src/processing.jl: the unfolding tensor, whose slices
# are the regular representation, and the isotypic blocks of the collision operator.
# The group elements those representations act on are tested in crystal_symmetry_test.jl.

using Test
using Ludwig
using StaticArrays
using LinearAlgebra

using Ludwig: point_group, conjugacy_classes
using Ludwig: irrep_basis, real_irrep_count, reality, component_ranges

@isdefined(cubic_crystal) || include("crystal_fixtures.jl")

@testset "Character tables and Mulliken symbols" begin

    groups = [
        ("C4",    PointGroup([c4_planar()], square_primitives()),
                  ["A", "B", "E"]),
        ("4mm",   PointGroup([c4_planar(), mirror_x_planar()], square_primitives()),
                  ["A1", "A2", "B1", "B2", "E"]),
        ("6mm",   PointGroup([c6_planar(), mirror_x_planar()], planar_hexagonal_primitives()),
                  ["A1", "A2", "B1", "B2", "E1", "E2"]),
        ("m-3m",  point_group(cubic_crystal()),
                  ["A1g", "A1u", "A2g", "A2u", "Eg", "Eu", "T1g", "T1u", "T2g", "T2u"]),
        ("6/mmm", point_group(hexagonal_crystal()),
                  ["A1g", "A1u", "A2g", "A2u", "B1g", "B1u", "B2g", "B2u",
                   "E1g", "E1u", "E2g", "E2u"]),
    ]

    for (name, pg, expected_labels) in groups
        @testset "$name" begin
            table = character_table(pg)
            classes = conjugacy_classes(table)
            χ = characters(table)
            class_sizes = length.(classes)

            # A planar group has no inversion centre and no horizontal mirror, so its symbols carry
            # neither a parity subscript nor a prime.
            @test sort(irrep_labels(table)) == sort(expected_labels)
            @test size(χ) == (length(expected_labels), length(classes))
            @test length(class_symbols(table)) == length(classes)

            # Row orthogonality. A real block built from a conjugate pair carries both of them and so
            # has twice the norm of a single representation.
            for μ in eachindex(expected_labels), ν in eachindex(expected_labels)
                expected_overlap = μ != ν ? 0.0 :
                                   order(pg) * (reality(table)[μ] === :complex_pair ? 2 : 1)
                @test sum(class_sizes .* χ[μ, :] .* χ[ν, :]) ≈ expected_overlap atol = 1e-8
            end

            # Σ dμ² over the complex representations is the group order, a merged pair counting for
            # both of the representations it holds.
            identity_class = findfirst(class -> class == [1], classes)
            dimension_sum = sum(eachindex(expected_labels)) do μ
                dimension = χ[μ, identity_class]
                reality(table)[μ] === :complex_pair ? dimension^2 / 2 : dimension^2
            end
            @test dimension_sum ≈ order(pg) atol = 1e-8

            # The regular representation is exhausted by the isotypic blocks.
            @test sum(length, component_ranges(table)) == order(pg)
        end
    end

    @testset "basis functions" begin
        # The textbook basis function of each representation, projected through the group average.
        # These are independent of the characters the labels were derived from, so they check the
        # whole pipeline — in particular the `B` subscript, the one label fixed by a convention.
        checks = [
            ("C4", PointGroup([c4_planar()], square_primitives()), [
                (k -> k[1]^2 + k[2]^2,           "A"),
                (k -> k[1] * k[2],               "B"),
                (k -> k[1],                      "E")]),
            ("4mm", PointGroup([c4_planar(), mirror_x_planar()], square_primitives()), [
                (k -> k[1]^2 + k[2]^2,           "A1"),
                (k -> k[1]^2 - k[2]^2,           "B1"),
                (k -> k[1] * k[2],               "B2"),
                (k -> k[1],                      "E")]),
            ("6mm", PointGroup([c6_planar(), mirror_x_planar()], planar_hexagonal_primitives()), [
                (k -> k[1] * (k[1]^2 - 3k[2]^2), "B1"),
                (k -> k[2] * (3k[1]^2 - k[2]^2), "B2"),
                (k -> k[1],                      "E1"),
                (k -> k[1]^2 - k[2]^2,           "E2")]),
            ("m-3m", point_group(cubic_crystal()), [
                (k -> k[1],                      "T1u"),
                (k -> k[1] * k[2],               "T2g"),
                (k -> 2k[3]^2 - k[1]^2 - k[2]^2, "Eg"),
                (k -> k[1] * k[2] * k[3],        "A2u")]),
            ("6/mmm", point_group(hexagonal_crystal()), [
                (k -> k[3],                      "A2u"),
                (k -> k[1]^2 - k[2]^2,           "E2g"),
                (k -> k[1] * (k[1]^2 - 3k[2]^2), "B1u"),
                (k -> k[1] * k[3],               "E1g")]),
        ]

        for (name, pg, fields) in checks
            @testset "$name" begin
                table = character_table(pg)
                for (field, label) in fields
                    content = irrep_content(table, field)
                    # The sampled orbits are group invariant, so the projections are orthogonal and
                    # their weights partition the field exactly.
                    @test sum(content) ≈ 1.0 atol = 1e-10
                    @test content[findfirst(isequal(label), irrep_labels(table))] ≈ 1.0 atol = 1e-10
                    @test which_irrep(table, field) == label
                end
                # A field spread over several representations carries no single symbol.
                @test_throws ArgumentError which_irrep(table, k -> 1.0 + k[1])
            end
        end
    end

    @testset "fields on a full-BZ grid" begin
        # The classified field lives on the quadrature grid rather than on synthetic orbits: `N_ibz`
        # representatives replicated into one block per group operation.
        block_grid(pg, representatives) = [M * point for M in operations(pg) for point in representatives]

        grids = [
            ("6mm", PointGroup([c6_planar(), mirror_x_planar()], planar_hexagonal_primitives()),
                    PointGroup([mirror_x_planar(), c6_planar()], planar_hexagonal_primitives()),
                    [SVector(0.1415, 0.0718), SVector(0.3733, 0.2846), SVector(0.0917, 0.4142)],
             [(k -> k[1] * (k[1]^2 - 3k[2]^2), "B1"),
              (k -> k[2] * (3k[1]^2 - k[2]^2), "B2"),
              (k -> k[1],                      "E1"),
              (k -> k[1]^2 - k[2]^2,           "E2"),
              (k -> k[1]^2 + k[2]^2,           "A1")]),
            ("m-3m", cubic_point_group(cubic_primitives(1.0, :sc)),
                     PointGroup(reverse(cubic_generators()), cubic_primitives(1.0, :sc)),
                     [SVector(0.1415, 0.0718, 0.3121), SVector(0.2846, 0.4142, 0.1732)],
             [(k -> k[1] * k[2] * k[3],        "A2u"),
              (k -> k[1],                      "T1u"),
              (k -> k[1] * k[2],               "T2g")]),
        ]

        for (name, pg, permuted, representatives, fields) in grids
            @testset "$name" begin
                table = character_table(pg)
                N_ibz = length(representatives)
                # Block-constant weights, as a rule replicated over the point-group orbit gives.
                weights = repeat(range(0.4, 1.3; length = N_ibz), order(pg))

                for (grid_name, grid_group) in (("canonical", pg), ("permuted blocks", permuted))
                    nodes = block_grid(grid_group, representatives)
                    for (f, label) in fields
                        field = Float64[f(node) for node ∈ nodes]
                        content = irrep_content(table, field, nodes, weights)
                        @test sum(content) ≈ 1.0 atol = 1e-10
                        @test content[findfirst(isequal(label), irrep_labels(table))] ≈ 1.0 atol = 1e-10
                        @test which_irrep(table, field, nodes, weights) == label
                    end

                    mixed = Float64[1.0 + node[1] + node[1] * node[2] for node ∈ nodes]
                    @test_throws ArgumentError which_irrep(table, mixed, nodes, weights)
                end

                # The blocks of the permuted grid carry the operations in a different order, so the
                # cases above are only distinguishable when the correspondence is read off the
                # coordinates rather than assumed.
                permuted_nodes = block_grid(permuted, representatives)
                @test Ludwig.block_operation_map(permuted_nodes, pg) != collect(1:order(pg))

                # Weighting reaches the answer: it cannot move a field carried by one representation,
                # but it does reweight one that spreads.
                nodes = block_grid(pg, representatives)
                mixed = Float64[1.0 + node[1] + node[1] * node[2] for node ∈ nodes]
                uniform = irrep_content(table, mixed, nodes, ones(length(nodes)))
                weighted = irrep_content(table, mixed, nodes, weights)
                @test sum(weighted) ≈ 1.0 atol = 1e-10
                @test !isapprox(uniform, weighted; atol = 1e-6)

                # The synthetic-orbit form is the same kernel over a grid it builds itself.
                for (f, label) in fields
                    @test which_irrep(table, f; seeds = 4) == label
                end
            end
        end

        # The grid has to be a point-group orbit of a common set of representatives.
        pg = PointGroup([c4_planar(), mirror_x_planar()], square_primitives())
        table = character_table(pg)
        representatives = [SVector(0.1415, 0.0718), SVector(0.3733, 0.2846)]
        nodes = [M * point for M in operations(pg) for point in representatives]
        field = Float64[node[1] for node ∈ nodes]
        @test_throws DimensionMismatch irrep_content(table, field[1:(end - 1)], nodes, ones(length(nodes)))
        @test_throws DimensionMismatch irrep_content(table, field, nodes, ones(length(nodes) - 1))
        @test_throws DimensionMismatch irrep_content(table, vcat(field, 0.0), vcat(nodes, [nodes[1]]),
                                                     ones(length(nodes) + 1))
        @test_throws ArgumentError irrep_content(table, zero(field), nodes, ones(length(nodes)))
    end

    @testset "non-ambivalent groups" begin
        # A cyclic group has classes that are not closed under inversion, so a conjugate pair of
        # complex representations merges into one real block and the block count falls below the
        # class count. Demanding one block per class never succeeds.
        pg = PointGroup([c4_planar()], square_primitives())
        @test length(conjugacy_classes(pg)) == 4
        @test real_irrep_count(group_multiplication_table(pg)) == 3

        ranges, basis = irrep_basis(pg)
        @test length(ranges) == 3
        @test transpose(basis) * basis ≈ I atol = 1e-10

        table = character_table(pg)
        @test count(isequal(:complex_pair), reality(table)) == 1
        # `reality` is row-indexed, so it has to travel with the labels into the published order.
        @test irrep_labels(table) == ["A", "B", "E"]
        @test reality(table) == [:real, :real, :complex_pair]

        # Adding a mirror makes the group ambivalent, and then blocks and classes agree.
        for ambivalent in (PointGroup([c4_planar(), mirror_x_planar()], square_primitives()),
                           point_group(cubic_crystal()))
            @test real_irrep_count(group_multiplication_table(ambivalent)) ==
                  length(conjugacy_classes(ambivalent))
            @test all(isequal(:real), reality(character_table(ambivalent)))
        end
    end

    @testset "canonical row order" begin
        # Published tables list the gerade or unprimed block first and the ungerade or primed block
        # second, and inside a block run A, B, E, T and then by ascending subscript.
        expected = [
            ("C4",    PointGroup([c4_planar()], square_primitives()),
                      ["A", "B", "E"]),
            ("4mm",   PointGroup([c4_planar(), mirror_x_planar()], square_primitives()),
                      ["A1", "A2", "B1", "B2", "E"]),
            ("6mm",   PointGroup([c6_planar(), mirror_x_planar()], planar_hexagonal_primitives()),
                      ["A1", "A2", "B1", "B2", "E1", "E2"]),
            ("m-3m",  point_group(cubic_crystal()),
                      ["A1g", "A2g", "Eg", "T1g", "T2g", "A1u", "A2u", "Eu", "T1u", "T2u"]),
            ("6/mmm", point_group(hexagonal_crystal()),
                      ["A1g", "A2g", "B1g", "B2g", "E1g", "E2g",
                       "A1u", "A2u", "B1u", "B2u", "E1u", "E2u"]),
        ]

        for (name, pg, labels) in expected
            @testset "$name" begin
                table = character_table(pg)
                @test irrep_labels(table) == labels

                # The order belongs to the stored table, not to its display, so every row-indexed
                # field has to travel with the labels.
                identity_class = findfirst(class -> class == [1], conjugacy_classes(table))
                dimensions = characters(table)[:, identity_class]
                @test dimensions ≈ [label[1] == 'E' ? 2.0 : label[1] == 'T' ? 3.0 : 1.0
                                    for label in labels]
                @test length(component_ranges(table)) == length(labels)
                @test sum(length, component_ranges(table)) == order(pg)

                # Rule 1: the whole gerade block precedes the whole ungerade one.
                ungerade = endswith.(labels, "u") .| endswith.(labels, "''")
                @test issorted(ungerade)
                # Alphabetical base letters coincide with non-decreasing dimension inside a block, so
                # the letter and dimension readings of the convention never disagree.
                @test issorted(dimensions[.!ungerade])
                @test issorted(dimensions[ungerade])
            end
        end

        # Nothing downstream reads a row position: the blocks are built from `component_ranges`, so
        # they follow the same order as the labels, and a basis function still finds its own row.
        pg = PointGroup([c6_planar(), mirror_x_planar()], planar_hexagonal_primitives())
        table = character_table(pg)
        @test which_irrep(table, k -> k[1]^2 - k[2]^2) == "E2"
        @test which_irrep(table, k -> k[1] * (k[1]^2 - 3k[2]^2)) == "B1"

        N_ibz = 3
        N = order(pg) * N_ibz
        invariant = full_collision_matrix(randn(N, N_ibz), pg; ibz_index = :column)
        symmetrized = (invariant + transpose(invariant)) / 2
        blocks = isotypic_blocks(symmetrized[:, 1:N_ibz], N_ibz, pg,
                                 component_ranges(table), Ludwig.basis(table))
        # Block μ carries the representation labelled μ, so its size is set by that representation's
        # dimension: d²·N_ibz, which is 4·N_ibz for the two-dimensional E1 and E2.
        for (μ, label) in enumerate(irrep_labels(table))
            expected_dimension = startswith(label, "E") ? 2 : 1
            @test size(blocks[μ].matrix, 1) == expected_dimension^2 * N_ibz
        end
    end

    @testset "display" begin
        # `@show`, `print` and `string` all reach the two-argument `show`, which has to delegate to
        # the text/plain method; otherwise they fall back to Julia's default field dump and print the
        # whole regular-representation basis.
        table = character_table(PointGroup([c4_planar(), mirror_x_planar()], square_primitives()))
        plain = sprint(show, table)

        @test plain == sprint(show, MIME("text/plain"), table)
        @test !occursin("CharacterTable{2}(PointGroup", plain)   # what the default printer emits
        @test occursin("order-8 point group", plain)
        @test all(label -> occursin(label, plain), irrep_labels(table))
        @test all(symbol -> occursin(symbol, plain), class_symbols(table))

        # Characters of a crystallographic point group are integers, so nothing in a rendered table
        # carries a decimal point, and the signed zeros the eigendecomposition leaves behind print
        # as plain `0`.
        for pg in (PointGroup([c4_planar(), mirror_x_planar()], square_primitives()),
                   PointGroup([c6_planar(), mirror_x_planar()], planar_hexagonal_primitives()),
                   point_group(cubic_crystal()),
                   point_group(hexagonal_crystal()))
            rendered = sprint(show, MIME("text/plain"), character_table(pg))
            @test !occursin(".", rendered)
            @test !occursin("-0", rendered)
        end
    end

    @testset "reference direction" begin
        # The `B` subscript is the only label that depends on the in-plane reference direction.
        # Turning it onto the diagonal exchanges `B1` and `B2` and leaves every other symbol alone.
        pg = PointGroup([c4_planar(), mirror_x_planar()], square_primitives())
        diagonal = character_table(pg; reference_direction = SVector(1.0, 1.0, 0.0) / sqrt(2))

        @test sort(irrep_labels(diagonal)) == ["A1", "A2", "B1", "B2", "E"]
        @test which_irrep(diagonal, k -> k[1] * k[2]) == "B1"
        @test which_irrep(diagonal, k -> k[1]^2 - k[2]^2) == "B2"
        @test which_irrep(diagonal, k -> k[1]^2 + k[2]^2) == "A1"
        @test which_irrep(diagonal, k -> k[1]) == "E"
    end

end

@testset "Unfolding tensor slices are the regular representation" begin

    # `unfolding_tensor` reads row `g` of the Cayley table, so `F[:, :, g]` is the permutation matrix
    # of left multiplication by operation `g` — exactly what `operation_permutation` returns, which is
    # why the tensor is built from it. Reading row `g⁻¹` instead builds the opposite convention and
    # silently transposes every unfolded operator. Each group here has an element of order greater
    # than two: on a group of exponent two every element is its own inverse, so the two conventions
    # coincide and the slip leaves no trace.
    groups = [
        ("C4 (cyclic, abelian)", PointGroup([c4_planar()], square_primitives())),
        ("4mm",                  PointGroup([c4_planar(), mirror_x_planar()], square_primitives())),
        ("6mm",                  PointGroup([c6_planar(), mirror_x_planar()], planar_hexagonal_primitives())),
        ("m-3m",                 point_group(cubic_crystal())),
    ]

    for (name, pg) in groups
        @testset "$name" begin
            n = order(pg)
            mult = group_multiplication_table(pg)
            F = Ludwig.unfolding_tensor(pg)

            @test size(F) == (n, n, n)
            @test all(sum(F; dims = 1) .== 1)          # every slice is a permutation matrix
            @test all(sum(F; dims = 2) .== 1)
            @test F[:, :, 1] == Matrix(I, n, n)        # the identity operation acts trivially

            # The convention under test: slice `g` is left multiplication by `g`, not by `g⁻¹`.
            @test all(F[:, :, g] == operation_permutation(g, mult) for g in 1:n)

            # `P_a·P_b = P_{a·b}`: the slices are the regular representation, so the map is a
            # homomorphism. The inverted convention composes the other way round, giving `P_{b·a}`,
            # which no abelian group can distinguish — hence the non-abelian fixtures above.
            @test all(F[:, :, a] * F[:, :, b] == F[:, :, mult[a, b]] for a in 1:n, b in 1:n)
        end
    end

end

@testset "Isotypic blocks of the collision operator" begin

    # Unfolding an arbitrary stored block gives a point-group-invariant operator, and symmetrizing
    # leaves it invariant. Compressing that onto the isotypic components is an orthogonal change of
    # basis, so the blocks must partition its spectrum exactly — this is the property that makes the
    # block construction worth doing at all, since it never forms the N × N operator.
    groups = [
        ("6mm",  PointGroup([c6_planar(), mirror_x_planar()], planar_hexagonal_primitives()), 3),
        ("m-3m", point_group(cubic_crystal()),                                                2),
    ]

    temperature = 0.025

    for (name, pg, N_ibz) in groups
        @testset "$name" begin
            group_order = order(pg)
            N = group_order * N_ibz

            invariant = full_collision_matrix(randn(N, N_ibz), pg; ibz_index = :column)
            balanced = (invariant + transpose(invariant)) / 2   # the detailed-balance form S
            L = balanced[:, 1:N_ibz]                            # the stored block of `balanced`
            scale = maximum(abs, balanced)

            table = character_table(pg)
            blocks = isotypic_blocks(L, N_ibz, pg, component_ranges(table), Ludwig.basis(table))

            @test length(blocks) == length(irrep_labels(table))
            @test sum(size(block.matrix, 1) for block in blocks) == N   # the blocks exhaust the grid

            # The table-taking form pairs the ranges and the basis for the caller.
            from_table = isotypic_blocks(L, N_ibz, table)
            @test length(from_table) == length(blocks)
            @test all(from_table[μ].matrix == blocks[μ].matrix for μ in eachindex(blocks))
            @test all(from_table[μ].basis == blocks[μ].basis for μ in eachindex(blocks))

            pooled = sort!(reduce(vcat, eigvals(Ludwig.symmetric_part(block.matrix))
                                        for block in blocks))
            @test pooled ≈ sort(eigvals(Symmetric(balanced))) atol = 1e-8 * scale

            # Schur's lemma makes the operator `I_d ⊗ A` on the isotypic component of a
            # d-dimensional representation, so every eigenvalue of that block is d-fold degenerate.
            for block in blocks
                d = sqrt(size(block.basis, 2))
                (isinteger(d) && d > 1) || continue
                values = eigvals(Ludwig.symmetric_part(block.matrix))
                for multiplet in Iterators.partition(eachindex(values), Int(d))
                    @test values[last(multiplet)] - values[first(multiplet)] ≈ 0 atol = 1e-8 * scale
                end
            end

            # A block eigenvector lifted back to the full grid is an eigenvector of the physical
            # operator `L_full = D^{-1/2}·S·D^{1/2}`, whose stored block is what a solver actually
            # holds. Detailed-balance factors are point-group invariant — weights are replicated
            # across the blocks and energies are constant along an orbit — so `D` is block-constant,
            # which is what leaves `L_full` invariant and lets the compression commute past it.
            balance = detailed_balance_factors(repeat(rand(N_ibz) .+ 0.5, group_order),
                                               repeat(randn(N_ibz) ./ 40, group_order), temperature)
            sqrtD = sqrt.(balance)
            L_full = (1 ./ sqrtD) .* balanced .* transpose(sqrtD)
            @test !(L_full ≈ transpose(L_full))   # a real unbalancing, so the lift is not the identity

            for block in blocks
                values, vectors = block_modes(block, N_ibz, sqrtD)
                @test length(values) == size(block.matrix, 1)
                for m in eachindex(values)
                    mode = @view vectors[:, m]
                    @test norm(L_full * mode - values[m] * mode) ≤ 1e-8 * scale * norm(mode)
                end
            end

            @test_throws DimensionMismatch isotypic_blocks(
                L[1:end-1, :], N_ibz, pg, component_ranges(table), Ludwig.basis(table))
        end
    end

end
