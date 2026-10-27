###
### Lattices, Brillouin zones, and point-group elements
###
#
# Covers src/lattice.jl and src/point_group.jl, together with the recovery of the
# point-group action on a stored full-BZ grid (`block_operation_map`, src/processing.jl).
# Representation theory of the same groups lives in representations_test.jl.

using Test
using Ludwig
using StaticArrays
using LinearAlgebra

using Ludwig: point_group, conjugacy_classes, block_operation_map, component_ranges
using Ludwig: operation_symbol, operation_symbols, operation_axis, class_symbol

@isdefined(cubic_crystal) || include("crystal_fixtures.jl")

@testset "3D crystal structure" begin

    @testset "cubic point group (m-3m)" begin
        crystal = cubic_crystal()                       # simple cubic, full Oh group
        pg = point_group(crystal)
        @test order(pg) == 48                           # closure to the full cubic group

        # closure preserves the lattice: every element, not just the generators the constructor
        # checks, is an integer matrix in the lattice basis.
        prim = primitives(crystal)
        for R in operations(pg)
            M = prim \ R * prim
            @test maximum(abs.(M .- round.(M))) < 1e-10
        end

        # every operation is orthogonal (a real point-group symmetry) with det ±1.
        for R in operations(pg)
            @test R * transpose(R) ≈ SMatrix{3,3,Float64}(I)
            @test abs(det(R)) ≈ 1.0
        end
    end

    @testset "hexagonal point group (6/mmm)" begin
        crystal = hexagonal_crystal()
        pg = point_group(crystal)
        @test order(pg) == 24                            # full hexagonal group

        # integrality in the oblique hexagonal basis, where change-of-basis drift would show first.
        prim = primitives(crystal)
        for R in operations(pg)
            M = prim \ R * prim
            @test maximum(abs.(M .- round.(M))) < 1e-10
        end
    end

    @testset "cubic Brillouin-zone face counts" begin
        # The conventional cubic Bravais lattices have textbook Wigner-Seitz cells.
        @test length(brillouin_zone(reciprocal_lattice_vectors(cubic_crystal(kind = :sc))).faces) == 6   # cube
        @test length(brillouin_zone(reciprocal_lattice_vectors(cubic_crystal(kind = :fcc))).faces) == 14  # truncated octahedron
        @test length(brillouin_zone(reciprocal_lattice_vectors(cubic_crystal(kind = :bcc))).faces) == 12  # rhombic dodecahedron
    end

    @testset "BZ volume equals |det(rlv)|" begin
        for kind in (:sc, :fcc, :bcc)
            crystal = cubic_crystal(kind = kind)
            rlv = reciprocal_lattice_vectors(crystal)
            bz = brillouin_zone(rlv)
            @test isapprox(polytope_volume(bz), abs(det(rlv)); rtol = 1e-9)
        end
    end

    @testset "IBZ tiles the BZ: volume(IBZ) = volume(BZ) / order" begin
        # The unfolding-tiling correctness gate the quadrature path depends on.
        for kind in (:sc, :fcc, :bcc)
            crystal = cubic_crystal(kind = kind)
            rlv = reciprocal_lattice_vectors(crystal)
            pg = point_group(crystal)
            bz_volume = polytope_volume(brillouin_zone(rlv))
            ibz_volume = polytope_volume(irreducible_brillouin_zone(rlv, pg))
            @test isapprox(ibz_volume, bz_volume / order(pg); rtol = 1e-8)
        end

        # Hexagonal: same invariant with the order-24 group.
        crystal = hexagonal_crystal()
        rlv = reciprocal_lattice_vectors(crystal)
        pg = point_group(crystal)
        bz_volume = polytope_volume(brillouin_zone(rlv))
        ibz_volume = polytope_volume(irreducible_brillouin_zone(rlv, pg))
        @test isapprox(ibz_volume, bz_volume / order(pg); rtol = 1e-8)
    end

end

@testset "Conjugacy classes" begin

    # Class counts and sizes are the textbook ones (they are the row/column counts of the
    # character tables), so any indexing slip in the Cayley-table conjugation shows up here.
    groups = [
        ("C4 (cyclic, abelian)", PointGroup([c4_planar()], square_primitives()),
                                 [1, 1, 1, 1]),
        ("4mm",                  PointGroup([c4_planar(), mirror_x_planar()], square_primitives()),
                                 [1, 1, 2, 2, 2]),
        ("6mm",                  PointGroup([c6_planar(), mirror_x_planar()], planar_hexagonal_primitives()),
                                 [1, 1, 2, 2, 3, 3]),
        ("m-3m",                 point_group(cubic_crystal()),
                                 [1, 1, 3, 3, 6, 6, 6, 6, 8, 8]),
        ("6/mmm",                point_group(hexagonal_crystal()),
                                 [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
    ]

    for (name, pg, expected_sizes) in groups
        @testset "$name" begin
            classes = conjugacy_classes(pg)
            @test sort(vcat(classes...)) == collect(1:order(pg))   # a partition of the group
            @test sort(length.(classes)) == expected_sizes
            @test classes[1] == [1]                                # identity is alone, and first

            # Conjugate operations are similar matrices: equal trace and determinant.
            ops = operations(pg)
            for class in classes, i in class
                @test tr(ops[i]) ≈ tr(ops[first(class)]) atol = 1e-10
                @test det(ops[i]) ≈ det(ops[first(class)]) atol = 1e-10
            end
        end
    end

end

@testset "Operation symbols and axes" begin

    # Determinant and trace name every crystallographic operation, and the multiset of symbols is a
    # complete fingerprint of the group — a misread rotoreflection shows up here and nowhere else.
    groups = [
        ("C4",    PointGroup([c4_planar()], square_primitives()),
                  Dict("E" => 1, "C4" => 2, "C2" => 1)),
        ("4mm",   PointGroup([c4_planar(), mirror_x_planar()], square_primitives()),
                  Dict("E" => 1, "C4" => 2, "C2" => 1, "σ" => 4)),
        ("6mm",   PointGroup([c6_planar(), mirror_x_planar()], planar_hexagonal_primitives()),
                  Dict("E" => 1, "C6" => 2, "C3" => 2, "C2" => 1, "σ" => 6)),
        ("m-3m",  point_group(cubic_crystal()),
                  Dict("E" => 1, "C4" => 6, "C3" => 8, "C2" => 9,
                       "i" => 1, "S4" => 6, "S6" => 8, "σ" => 9)),
        ("6/mmm", point_group(hexagonal_crystal()),
                  Dict("E" => 1, "C6" => 2, "C3" => 2, "C2" => 7,
                       "i" => 1, "S3" => 2, "S6" => 2, "σ" => 7)),
    ]

    for (name, pg, expected) in groups
        @testset "$name" begin
            symbols = operation_symbols(pg)
            @test length(symbols) == order(pg)
            @test Dict(s => count(isequal(s), symbols) for s in unique(symbols)) == expected

            # Conjugate operations are similar, so every class is homogeneous in type. The class
            # symbol is read off one representative and rests on this.
            for class in conjugacy_classes(pg)
                @test all(isequal(symbols[first(class)]), symbols[class])
                @test class_symbol(class, operations(pg)) ==
                      (length(class) == 1 ? symbols[first(class)] :
                       "$(length(class))$(symbols[first(class)])")
            end
        end
    end

    @testset "operations multiplicative order cannot separate" begin
        # S₃ and S₆ both cube to a reflection and so share the order 6, and `i` and `σ` both square
        # to the identity. Determinant and trace tell each pair apart where the order alone cannot,
        # which is why these four are the cases worth asserting.
        rotoreflection_3 = SMatrix{3,3,Float64}(cos(2pi / 3), sin(2pi / 3), 0,
                                                -sin(2pi / 3), cos(2pi / 3), 0, 0, 0, -1)
        rotoreflection_6 = SMatrix{3,3,Float64}(cos(pi / 3), sin(pi / 3), 0,
                                                -sin(pi / 3), cos(pi / 3), 0, 0, 0, -1)
        inversion = SMatrix{3,3,Float64}(-1, 0, 0, 0, -1, 0, 0, 0, -1)
        mirror_z = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, -1)

        @test operation_symbol(rotoreflection_3) == "S3"
        @test operation_symbol(rotoreflection_6) == "S6"
        @test operation_symbol(inversion) == "i"
        @test operation_symbol(mirror_z) == "σ"

        @test_throws ArgumentError operation_symbol(SMatrix{2,2,Float64}(2, 0, 0, 1))
    end

    @testset "axes" begin
        inversion = SMatrix{3,3,Float64}(-1, 0, 0, 0, -1, 0, 0, 0, -1)
        mirror_z = SMatrix{3,3,Float64}(1, 0, 0, 0, 1, 0, 0, 0, -1)

        # The identity and inversion single out no direction; in the plane neither does a rotation,
        # `-I` being a proper half turn rather than an inversion centre.
        @test operation_axis(SMatrix{3,3,Float64}(I)) === nothing
        @test operation_axis(inversion) === nothing
        @test operation_axis(SMatrix{2,2,Float64}(I)) === nothing
        @test operation_axis(-SMatrix{2,2,Float64}(I)) === nothing

        @test operation_axis(mirror_z) ≈ SVector(0.0, 0.0, 1.0)
        @test operation_axis(mirror_x_planar()) ≈ SVector(0.0, 1.0)   # the line normal

        pg = point_group(cubic_crystal())
        ops = operations(pg)
        symbols = operation_symbols(pg)

        # `unique` compares with `isequal`, for which -0.0 and 0.0 differ, so a signed zero left in
        # an axis would report six four-fold axes instead of three.
        four_fold = unique(operation_axis(ops[i]) for i in eachindex(ops) if symbols[i] == "C4")
        @test length(four_fold) == 3
        @test all(axis -> count(!iszero, axis) == 1, four_fold)

        # Three two-fold axes along the cube edges and six along the face diagonals.
        two_fold = unique(operation_axis(ops[i]) for i in eachindex(ops) if symbols[i] == "C2")
        @test length(two_fold) == 9
        @test count(axis -> count(!iszero, axis) == 2, two_fold) == 6

        # A rotoreflection reports the axis it turns about, a mirror its plane normal.
        @test unique(operation_axis(ops[i]) for i in eachindex(ops) if symbols[i] == "S4") == four_fold
    end

end

@testset "block_operation_map" begin
    # The map is what lets the unfolding tolerate a point group enumerated differently from the grid,
    # so the test that gives it meaning is geometric: the operation it names for each block must
    # actually carry block 1 onto that block.
    function check_blocks(nodes, pg, block_op, N_ibz; atol)
        ops = operations(pg)
        @test length(block_op) == order(pg)
        @test sort(block_op) == collect(1:order(pg))   # a bijection onto the group
        @test block_op[1] == 1                         # operations(pg)[1] is the identity
        @test all(norm(nodes[(g - 1) * N_ibz + r] - ops[block_op[g]] * nodes[r]) ≤ atol
                  for g in 1:order(pg), r in 1:N_ibz)
    end

    @testset "simple cubic (m-3m), 3D" begin
        # The map reads only the block layout, so a grid assembled to the same convention covers
        # D = 3 and the order-48 group without the cost of a 3D quadrature rule.
        crystal = cubic_crystal()
        pg = point_group(crystal)
        @test order(pg) == 48

        # Generic representatives: all |components| distinct and nonzero, so every stabilizer is
        # trivial and the 48 blocks stay disjoint.
        ibz_nodes = [SVector(0.31 + 0.02r, 0.17 + 0.03r, 0.07 + 0.05r) for r in 1:4]
        nodes = [operations(pg)[g] * ibz_nodes[r] for g in 1:order(pg) for r in eachindex(ibz_nodes)]
        # m-3m operations are integer matrices, so the images are reproduced bit-exactly.
        check_blocks(nodes, pg, block_operation_map(nodes, pg), length(ibz_nodes); atol = 0.0)
    end

    @testset "independent of the generator ordering" begin
        # `close_group` enumerates from the generator list, so the same group built from a permuted
        # list numbers its elements differently, and nothing carries that numbering alongside a
        # stored collision matrix. Given the nodes, the layout is read off the coordinates instead
        # of assumed, and every consumer of the layout inherits that.
        cubic_cell = SMatrix{3,3,Float64}(I)
        generators = cubic_generators()                             # C4z, C3, inversion
        pg_built = PointGroup(generators, cubic_cell)
        pg_reordered = PointGroup(generators[[2, 1, 3]], cubic_cell)

        @test order(pg_built) == order(pg_reordered) == 48
        @test Set(operations(pg_built)) == Set(operations(pg_reordered))   # same group
        @test operations(pg_built) != operations(pg_reordered)             # different enumeration

        N_ibz = 2
        N = order(pg_built) * N_ibz
        ibz_nodes = [SVector(0.31 + 0.02r, 0.17 + 0.03r, 0.07 + 0.05r) for r in 1:N_ibz]
        nodes = [operations(pg_built)[g] * ibz_nodes[r]
                 for g in 1:order(pg_built) for r in 1:N_ibz]

        # The grid was laid out under pg_built's ordering; pg_reordered must still resolve it.
        check_blocks(nodes, pg_built, block_operation_map(nodes, pg_built), N_ibz; atol = 0.0)
        check_blocks(nodes, pg_reordered, block_operation_map(nodes, pg_reordered), N_ibz; atol = 0.0)

        # Non-vacuous: the canonical correspondence holds for pg_built and fails for pg_reordered,
        # which is exactly the assumption the unfolding makes when it is given no nodes.
        @test block_operation_map(nodes, pg_built) == collect(1:order(pg_built))
        @test block_operation_map(nodes, pg_reordered) != collect(1:order(pg_reordered))

        # Relabelling the unfolding tensor by the map is what carries that into the unfolding.
        block_op = block_operation_map(nodes, pg_reordered)
        @test Ludwig.unfolding_tensor(pg_reordered, block_op) ==
              Ludwig.unfolding_tensor(pg_reordered)[block_op, block_op, block_op]
        @test Ludwig.unfolding_tensor(pg_built, collect(1:order(pg_built))) ==
              Ludwig.unfolding_tensor(pg_built)

        # A genuinely point-group-invariant operator, so unfolding either of its stored blocks must
        # return it whichever ordering the group was built from.
        R = randn(N, N)
        L_full = zeros(N, N)
        for a in 1:order(pg_built)
            perm = grid_permutation(group_multiplication_table(pg_built), a, N_ibz)
            L_full .+= R[perm, perm]
        end
        L = L_full[:, 1:N_ibz]
        L_rows = L_full[1:N_ibz, :]

        @test full_collision_matrix(L, pg_built; nodes) ≈ L_full
        @test full_collision_matrix(L, pg_reordered; nodes) ≈ L_full     # the ordering must not matter
        @test full_collision_matrix(L_rows, pg_built; ibz_index = :row, nodes) ≈ L_full
        @test full_collision_matrix(L_rows, pg_reordered; ibz_index = :row, nodes) ≈ L_full

        # The grid is in the canonical layout for pg_built, so recovering it changes nothing there.
        @test full_collision_matrix(L, pg_built; nodes) == full_collision_matrix(L, pg_built)

        balance = repeat(rand(N_ibz) .+ 0.5, order(pg_built))
        values_built, _ = collision_eigen(full_collision_matrix(L, pg_built; nodes), balance)
        values_reordered, _ = collision_eigen(full_collision_matrix(L, pg_reordered; nodes), balance)
        @test values_built ≈ values_reordered

        # Symmetrizing a stored block pairs each entry with its transpose partner, which lives in the
        # block a group inverse away. Reading that inverse off the unfolding tensor carries the
        # relabelling; taking it from the multiplication table would index the stored block in the
        # group's own enumeration and agree only in the canonical ordering.
        symmetrized_built = symmetrized_collision_matrix(L, pg_built, balance; nodes)
        symmetrized_reordered = symmetrized_collision_matrix(L, pg_reordered, balance; nodes)
        @test size(symmetrized_built) == size(L)
        @test symmetrized_built == symmetrized_reordered

        rows_built = symmetrized_collision_matrix(L_rows, pg_built, balance; ibz_index = :row, nodes)
        rows_reordered = symmetrized_collision_matrix(L_rows, pg_reordered, balance; ibz_index = :row, nodes)
        @test rows_built == rows_reordered

        # Against the full-operator path on the same data, which needs no layout at all — so this
        # pins the relabelled stored-block arithmetic rather than only its self-consistency.
        on_full = symmetrized_collision_matrix(full_collision_matrix(L, pg_built; nodes),
                                               pg_built, balance)
        @test full_collision_matrix(symmetrized_built, pg_built; nodes) ≈ on_full
        @test full_collision_matrix(rows_built, pg_built; ibz_index = :row, nodes) ≈ on_full

        # Left unstated, the layout is assumed canonical, and the symmetrization pairs entries across
        # the wrong blocks — the same silent failure the unfolding has just below.
        @test !(symmetrized_collision_matrix(L, pg_reordered, balance) ≈ symmetrized_built)

        # Without the nodes the unfolding takes block g to carry operations(pg)[g], which the
        # reordering breaks — and breaks silently, since the result is still a matrix of the right
        # shape whose trace is unchanged.
        positional = full_collision_matrix(L, pg_reordered)
        @test !(positional ≈ L_full)
        @test tr(positional) ≈ tr(L_full)

        # The isotypic decomposition carries the same guarantee: its blocks partition the spectrum of
        # the symmetrized operator whichever ordering the point group was built from.
        symmetrized = (L_full + transpose(L_full)) / 2
        reference = sort(eigvals(Symmetric(symmetrized)))
        scale = maximum(abs, symmetrized)
        for pg in (pg_built, pg_reordered)
            table = character_table(pg)
            blocks = isotypic_blocks(symmetrized[:, 1:N_ibz], N_ibz, pg, component_ranges(table),
                                     Ludwig.basis(table); nodes)
            pooled = sort!(reduce(vcat, eigvals(Ludwig.symmetric_part(block.matrix))
                                        for block in blocks))
            @test pooled ≈ reference atol = 1e-8 * scale
        end

        # `nodes` must span the whole grid; the IBZ representatives alone cannot name the blocks.
        @test_throws DimensionMismatch full_collision_matrix(L, pg_built; nodes = nodes[1:N_ibz])
    end

    @testset "unrecoverable layouts are rejected" begin
        pg = point_group(cubic_crystal())
        ibz_nodes = [SVector(0.31 + 0.02r, 0.17 + 0.03r, 0.07 + 0.05r) for r in 1:3]
        nodes = [operations(pg)[g] * ibz_nodes[r] for g in 1:order(pg) for r in eachindex(ibz_nodes)]

        # Node counts that are not a whole number of blocks cannot carry the layout. A count that
        # divides the group order rather than being divisible by it is the one a laxer check admits.
        @test_throws DimensionMismatch block_operation_map(nodes[1:end-1], pg)
        @test_throws DimensionMismatch block_operation_map(nodes[1:24], pg)

        # A block that is not an image of block 1 at all.
        corrupted = copy(nodes)
        corrupted[4] = SVector(0.9, 0.8, 0.7)
        @test_throws ArgumentError block_operation_map(corrupted, pg)

        # A block duplicated over another: the orbit is no longer covered exactly once.
        duplicated = copy(nodes)
        duplicated[4:6] = duplicated[1:3]
        @test_throws ArgumentError block_operation_map(duplicated, pg)

        # Representatives that all sit on a common mirror cannot distinguish it from the identity.
        on_mirror = [SVector(0.3 + 0.05r, 0.2 + 0.03r, 0.0) for r in 1:3]
        degenerate = [operations(pg)[g] * on_mirror[r] for g in 1:order(pg) for r in eachindex(on_mirror)]
        @test_throws ArgumentError block_operation_map(degenerate, pg)

        @test_throws ArgumentError block_operation_map(SVector{3,Float64}[], pg)

        # The same layouts reach `full_collision_matrix` through its `nodes` argument.
        L = randn(length(nodes), 3)
        @test_throws ArgumentError full_collision_matrix(L, pg; nodes = corrupted)
        @test_throws ArgumentError full_collision_matrix(L, pg; nodes = duplicated)
    end
end
