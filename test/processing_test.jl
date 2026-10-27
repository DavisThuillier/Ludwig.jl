###
### Collision-operator processing (src/processing.jl)
###
#
# Unfolding a stored IBZ block into the full-BZ operator, the detailed-balance factors, and the
# diagonalization built on them. The point-group side of processing.jl is covered elsewhere:
# recovering the block layout from node coordinates in `crystal_symmetry_test.jl`, the isotypic
# decomposition in `representations_test.jl`.

using Test
using Ludwig
using StaticArrays
using LinearAlgebra

using Ludwig: full_collision_matrix, collision_eigen, detailed_balance_factors,
                   symmetrize_collision_matrix!, block_constancy

@isdefined(cubic_crystal) || include("crystal_fixtures.jl")

@testset "Collision operator diagonalization" begin
    # Square-lattice point group C4 (order 4) from a 90° rotation generator.
    pg = PointGroup([c4_planar()], square_primitives())
    @test order(pg) == 4

    N_ibz = 3
    N = order(pg) * N_ibz

    # Balance factors are point-group invariant ⇒ N_ibz distinct values replicated across blocks.
    balance_ibz = rand(N_ibz) .+ 0.5
    balance_full = repeat(balance_ibz, order(pg))

    # Build a physically faithful stored block: a point-group-invariant operator obeying detailed
    # balance Dᵢ·L_full[i,j] = Dⱼ·L_full[j,i]. Symmetrize a random S0 over the group orbit so it is
    # both symmetric and invariant, then unbalance it: L_full = D^{-1/2}·S0·D^{1/2}. The first N_ibz
    # columns are the stored block L.
    perms = [grid_permutation(group_multiplication_table(pg), a, N_ibz) for a in 1:order(pg)]
    R = randn(N, N)
    S0 = zeros(N, N)
    for perm in perms
        S0 .+= R[perm, perm]
    end
    S0 = (S0 + transpose(S0)) / 2
    sqrtD = sqrt.(balance_full)
    L_full = (1.0 ./ sqrtD) .* S0 .* transpose(sqrtD)   # D^{-1/2}·S0·D^{1/2}
    L = L_full[:, 1:N_ibz]        # kⱼ restricted to the IBZ representatives
    L_rows = L_full[1:N_ibz, :]   # kᵢ restricted instead

    # L_full is not symmetric, so the two stored orientations are genuinely different data and
    # neither unfolding can stand in for the other.
    @test !(L_full ≈ transpose(L_full))
    @test !(L ≈ transpose(L_rows))

    @test full_collision_matrix(L, pg) ≈ L_full   # unfolding recovers the invariant operator
    @test full_collision_matrix(L_rows, pg; ibz_index = :row) ≈ L_full

    # unfold_rows(R) = transpose(unfold_cols(transpose(R))): the same stored entries, permuted, so
    # the two agree bitwise rather than approximately.
    @test full_collision_matrix(L_rows, pg; ibz_index = :row) ==
          transpose(full_collision_matrix(transpose(L_rows), pg))
    @test full_collision_matrix(L, pg) ==
          transpose(full_collision_matrix(transpose(L), pg; ibz_index = :row))

    L_unfolded = full_collision_matrix(L, pg)
    unfolded_reference = copy(L_unfolded)
    values, vectors = collision_eigen(L_unfolded, balance_full)
    @test L_unfolded == unfolded_reference   # the operator handed in is left intact
    @test all(isreal, values)
    @test sort(values) ≈ sort(real.(eigvals(L_full)))   # balanced spectrum equals the raw L_full spectrum

    # Physical-basis eigenvectors solve L_full·ψ = λ·ψ.
    for k in 1:N
        @test L_full * vectors[:, k] ≈ values[k] * vectors[:, k]
    end

    # detailed_balance_factors composes weights with the Fermi-Dirac occupation factor.
    weights = rand(N) .+ 0.5
    energies = randn(N)
    n = fermi_dirac.(energies, 0.025)
    @test detailed_balance_factors(weights, energies, 0.025) ≈ weights .* n .* (1 .- n)

    @test_throws ArgumentError collision_eigen(L_unfolded, -balance_full)   # D must be positive

    # The stored block must hold a whole number of representatives per group element.
    @test_throws DimensionMismatch full_collision_matrix(L[:, 1:N_ibz-1], pg)
    @test_throws DimensionMismatch full_collision_matrix(L_rows[1:N_ibz-1, :], pg; ibz_index = :row)

    # A mis-declared orientation is rejected, never silently reinterpreted.
    @test_throws DimensionMismatch full_collision_matrix(L_rows, pg)
    @test_throws DimensionMismatch full_collision_matrix(L, pg; ibz_index = :row)
    @test_throws ArgumentError full_collision_matrix(L, pg; ibz_index = :columns)
end

@testset "Symmetrized collision matrix" begin
    pg = PointGroup([c4_planar()], square_primitives())
    N_ibz = 3
    N = order(pg) * N_ibz
    balance_full = repeat(rand(N_ibz) .+ 0.5, order(pg))

    # A stored block as `electron_electron` writes it: no self-scattering entry on the diagonal, and
    # no detailed balance either, since supplying that is what the pipeline is for.
    L = randn(N, N_ibz)
    for r in 1:N_ibz
        L[r, r] = 0.0
    end

    # A stored block comes back as a stored block, symmetrized without the N × N operator ever
    # existing. Unfolding it afterwards reproduces symmetrizing the unfolded operator, which is the
    # property that makes the cheaper placement legitimate.
    chained = full_collision_matrix(L, pg)
    symmetrize_collision_matrix!(chained, balance_full)
    enforce_particle_conservation!(chained, balance_full)

    stored = symmetrized_collision_matrix(L, pg, balance_full)
    @test size(stored) == size(L)
    @test full_collision_matrix(stored, pg) ≈ chained

    # A full operator is symmetrized where it stands, with no unfolding round trip, and the argument
    # survives the call.
    unfolded_input = full_collision_matrix(L, pg)
    @test symmetrized_collision_matrix(unfolded_input, pg, balance_full) == chained
    @test unfolded_input == full_collision_matrix(L, pg)

    # Detailed balance, particle conservation and point-group invariance hold together.
    L_full = full_collision_matrix(stored, pg)
    balanced = Diagonal(balance_full) * L_full
    @test balanced ≈ transpose(balanced)
    @test all(≈(0; atol = 1e-12 * maximum(abs, L_full)), sum(L_full, dims = 2))
    for a in 1:order(pg)
        perm = grid_permutation(group_multiplication_table(pg), a, N_ibz)
        @test L_full[perm, perm] ≈ L_full
    end

    # The conserved diagonal is applied as a decrement, so it comes out independent of the diagonal
    # it replaces and re-applying it is a no-op. Both branches have regressed to plain assignment
    # before, which silently zeroes the diagonal of an operator that already carries one.
    for populated in (full_collision_matrix(L, pg), copy(L))    # row branch, then column branch
        cleared = enforce_particle_conservation!(copy(populated), balance_full)
        for r in axes(populated, 2)
            populated[r, r] = 7.5 * randn()
        end
        @test enforce_particle_conservation!(populated, balance_full) ≈ cleared
        @test enforce_particle_conservation!(copy(cleared), balance_full) ≈ cleared
    end

    # Enforcing before symmetrizing does not: symmetrization holds the diagonal fixed while moving
    # the off-diagonal sums, so the conserved diagonal no longer cancels them.
    reversed = full_collision_matrix(L, pg)
    enforce_particle_conservation!(reversed, balance_full)
    symmetrize_collision_matrix!(reversed, balance_full)
    @test !all(≈(0; atol = 1e-12 * maximum(abs, reversed)), sum(reversed, dims = 2))

    # Either stored orientation describes the same operator, so both symmetrize to the same one.
    # Each keeps its own shape, and the two conservation conditions they select are distinct
    # summations that agree only once symmetrization has made detailed balance exact.
    unfolded = full_collision_matrix(L, pg)
    from_row = symmetrized_collision_matrix(unfolded[1:N_ibz, :], pg, balance_full; ibz_index = :row)
    from_column = symmetrized_collision_matrix(unfolded[:, 1:N_ibz], pg, balance_full)
    @test size(from_row) == (N_ibz, N)
    @test size(from_column) == (N, N_ibz)
    @test full_collision_matrix(from_row, pg; ibz_index = :row) ≈
          full_collision_matrix(from_column, pg)

    # `½(L + D⁻¹LᵀD)` is a projection, and the in-place step returns its argument for chaining.
    once = symmetrize_collision_matrix!(full_collision_matrix(L, pg), balance_full)
    @test symmetrize_collision_matrix!(copy(once), balance_full) ≈ once
    @test symmetrize_collision_matrix!(once, balance_full) === once

    # Balance factors that vary within a point-group orbit would buy detailed balance at the cost of
    # point-group invariance, which no downstream result reveals.
    @test block_constancy(balance_full, N_ibz) == 0
    @test_throws ArgumentError symmetrized_collision_matrix(L, pg, rand(N) .+ 0.5)

    @test_throws DimensionMismatch symmetrize_collision_matrix!(randn(N, N_ibz), balance_full)
    @test_throws DimensionMismatch symmetrize_collision_matrix!(randn(N, N), balance_full[1:end-1])
    @test_throws DimensionMismatch symmetrized_collision_matrix(L, pg, balance_full[1:end-1])
end
