###
### Quadrature pipeline (src/quadrature.jl)
###
#
# The compression that turns an isoenergy mesh into a quadrature rule — maxvol node selection with
# NNLS weights, the coarse-sampling/fine-measure split, and the IBZ and full-BZ drivers built on them.

using Test
using Ludwig
using StaticArrays
using LinearAlgebra

using Ludwig: centroids, surface_locator, locate
using Ludwig: mesh_harmonics, surface_moments, maxvol_quadrature, isoenergy_quadrature_rule
using Ludwig: distance_to_boundary, ibz_quadrature_rule, bz_quadrature_rule
using Ludwig: block_operation_map, point_group

@isdefined(cubic_crystal) || include("crystal_fixtures.jl")
@isdefined(graphene_band) || include("band_fixtures.jl")

@testset "maxvol_quadrature under-resolution warnings" begin
    # A rule can fall short of its moment target two ways, and each has its own warning: too few
    # eligible candidates to select a full basis, or a full selection whose non-negative solve is
    # clipped. The two are mutually exclusive by construction, so no rule is ever reported twice.
    U_starved = [1.0 0.0 0.0 0.0 0.0;
                 0.0 1.0 0.0 0.0 0.0;
                 0.0 0.0 1.0 0.0 0.0]
    U_ok = [1.0 0.0;
            0.0 1.0;
            1.0 1.0]

    # Fewer candidates than requested basis functions: the rule can place at most
    # #candidates nodes and warns that the measure is under-resolved.
    local starved_nodes
    @test_logs (:warn, r"eligible candidates") begin
        starved_nodes, _, _ = maxvol_quadrature(U_starved, ones(3); reject_boundary = false)
    end
    @test length(starved_nodes) < 5                 # cannot reach the requested basis count

    # Enough candidates for the requested basis: the rule builds silently, and with the constraint
    # slack the full-length selection interpolates its moments exactly.
    ok_nodes, _, ok_residual = @test_logs maxvol_quadrature(U_ok, ones(3); reject_boundary = false)
    @test 0 < length(ok_nodes) ≤ 2
    @test ok_residual < 1e-12

    # A moment target outside the non-negative cone of that same selection: the constraint binds, a
    # weight is driven to zero, and the residual is what reports it.
    _, _, clipped_residual = @test_logs (:warn, r"non-negativity constraint") maxvol_quadrature(
        U_ok, ones(3); reject_boundary = false, moments = [1.0, -1.0])
    @test clipped_residual > 1e-8

    # The same miss on a short selection stays with the eligibility warning: a starved rule misses its
    # moments because it has too few nodes, not because the constraint bound.
    @test_logs (:warn, r"eligible candidates") maxvol_quadrature(
        U_starved, ones(3); reject_boundary = false, moments = [1.0, 1.0, 1.0, 1.0, 0.0])

    # Silenced for a caller that aggregates the residuals itself.
    @test_logs maxvol_quadrature(U_ok, ones(3); reject_boundary = false, moments = [1.0, -1.0],
                                 warn_residual = false)
end

@testset "isoenergy_quadrature_rule" begin
    # Unit circle, where |v| = 2 so the isoenergy measure has total mass ∫ dS / |v| = π.
    f(p) = p[1]^2 + p[2]^2
    v(k) = SVector(2k[1], 2k[2])
    ρ(k) = 1 / norm(v(k))
    N_s = 6

    @testset "no boundary rejection" begin
        mesh = level_set(f, 1.0, (-2.0, 2.0), (-2.0, 2.0), 4; refinement_depth = 6, weld = true)
        ϕ = mesh_harmonics(mesh, ρ, N_s)

        # Without an ibz to measure against, the rule must run rather than demand boundary distances.
        nodes, weights, ε = isoenergy_quadrature_rule(mesh, v, ϕ; reject_boundary = false)
        @test length(nodes) == length(weights)
        @test 0 < length(nodes) ≤ N_s
        @test all(>(0), weights)
        @test ε < 1e-8
        # The constant harmonic is in the basis, so the weights reproduce the total mass.
        @test sum(weights) ≈ integrate_over_mesh(ρ, mesh) rtol = 1e-6
        @test sum(weights) ≈ π rtol = 1e-3

        # Boundary rejection has no skin depth to work with unless the ibz is supplied.
        @test_throws ArgumentError isoenergy_quadrature_rule(mesh, v, ϕ)
        # A moment target from a different basis is caught before it can silently inflate the residual.
        @test_throws DimensionMismatch isoenergy_quadrature_rule(mesh, v, ϕ;
            moments = zeros(N_s + 1), reject_boundary = false)
    end

    @testset "moments from a finer mesh" begin
        # Candidates and harmonics on the coarse contour; the moment target on the fine one.
        coarse = level_set(f, 1.0, (-2.0, 2.0), (-2.0, 2.0), 4; refinement_depth = 5, weld = true)
        fine   = level_set(f, 1.0, (-2.0, 2.0), (-2.0, 2.0), 4; refinement_depth = 8, weld = true)
        loc = surface_locator(coarse)
        ϕ = mesh_harmonics(coarse, ρ, N_s)

        moments, dropped = surface_moments(loc, ϕ, fine, v)
        @test dropped == 0.0

        # `dropped` counts only fine points whose interpolation came back NaN, and `locate` has no
        # distance cutoff, so it is zero whatever the two meshes are: necessary but not sufficient.
        # The located distance carries the property it stands for. A coarse mesh that had missed a
        # contour component would strand that component's fine points many cells from any coarse
        # parent, so the threshold is set at the scale of a few coarse simplices.
        @test maximum(locate(loc, c)[3] for c in centroids(fine)) < 3 * loc.cell

        nodes, weights, ε = isoenergy_quadrature_rule(coarse, v, ϕ; moments, reject_boundary = false)
        @test all(≥(0), weights)
        @test ε < 1e-8
        @test all(node ∈ centroids(coarse) for node ∈ nodes)   # nodes stay on the coarse candidate set
        @test sum(weights) ≈ integrate_over_mesh(ρ, fine) rtol = 1e-6
    end
end

@testset "ibz_quadrature_rule" begin
    # Free-electron band on the square lattice, the one case with a closed-form answer. Every
    # isoenergy contour is a circle of radius √(2mε̃) on which |v| = k/m, so ∫ dS/|v| = 2πm is
    # independent of the energy and the Gauss-Legendre energy rule is exact. The 4mm IBZ is a wedge
    # of angle 2π/|G| about Γ, so it carries 1/|G| of each contour and the rule's total mass is
    # 2·energy_window·2πm/|G| — the area of the window annulus in bare d²k, divided by the order.
    m, μ = 1.0, 0.18
    energy_window = 0.02
    N_s, N_ε = 6, 4

    crystal = Crystal(square_primitives(),
                      PointGroup([c4_planar(), mirror_x_planar()], square_primitives()))
    pg = point_group(crystal)
    @test order(pg) == 8

    dispersion(k) = dot(k, k) / 2m - μ
    velocity(k) = k / m

    # A converged sweep warns about nothing.
    nodes, weights, residuals = @test_logs(
        ibz_quadrature_rule(dispersion, velocity, crystal, energy_window, N_s, N_ε;
                            initial_depth = 4, refinement_depth = 8))

    @test length(nodes) == length(weights)
    @test length(residuals) == N_ε                   # one sheet per energy node, none of them empty
    @test all(>(0), weights)
    @test all(<(1e-8), residuals)
    @test sum(weights) ≈ 2 * energy_window * 2π * m / order(pg) rtol = 1e-4

    # Every sheet places its full complement of nodes: a weight driven to zero costs the rule a node.
    @test length(nodes) == N_s * N_ε

    @testset "under-resolved sheets are reported once, naming the levels" begin
        # Two levels short of the depth above, the arc carries ~25 segments and skin-depth rejection
        # leaves too few candidates clear of the wedge faces for the non-negative solve; two of the
        # four sheets come back clipped. They are reported together after the sweep rather than one
        # warning per sheet, so the offending energy levels stay identifiable.
        coarse_nodes, coarse_weights, coarse_residuals = @test_logs(
            (:warn, r"under-resolve the harmonic basis"),
            ibz_quadrature_rule(dispersion, velocity, crystal, energy_window, N_s, N_ε;
                                initial_depth = 4, refinement_depth = 6))

        @test length(coarse_nodes) < N_s * N_ε
        @test maximum(coarse_residuals) > 1e-2
        # Total mass is the least sensitive probe of this failure: the clipping lives in the higher
        # harmonics, so the constant mode still comes back to within a percent and a mass check alone
        # would nearly pass.
        @test isapprox(sum(coarse_weights), 2 * energy_window * 2π * m / order(pg); rtol = 1e-2)
    end

    # Skin-depth rejection is what lets the unfolding copy a node's weight to each of its images
    # without double counting: a retained node stays strictly inside the wedge, so no group element
    # fixes it and its |G| images are distinct points rather than coincident ones.
    ibz = irreducible_brillouin_zone(crystal)
    @test all(x -> distance_to_boundary(x, ibz) > 0, nodes)
    moved = [norm(M * x - x) for M in operations(pg)[2:end], x in nodes]
    @test operations(pg)[1] ≈ one(SMatrix{2,2,Float64})     # block 1 carries the identity
    @test minimum(moved) > 1e-3

    @testset "unfolded by bz_quadrature_rule" begin
        # `bz_quadrature_rule` is the IBZ rule replicated over the group, with the (2π)^D of the
        # momentum measure divided out. Both drivers march the same deterministic mesh, so the
        # identity block reproduces the IBZ nodes exactly.
        bz_nodes, bz_weights, bz_residuals =
            bz_quadrature_rule(dispersion, velocity, crystal, energy_window, N_s, N_ε;
                               initial_depth = 4, refinement_depth = 8)

        @test length(bz_nodes) == order(pg) * length(nodes)
        @test bz_nodes[1:length(nodes)] == nodes
        @test bz_weights ≈ repeat(weights, order(pg)) ./ (2π)^2
        @test bz_residuals == residuals
        @test sum(bz_weights) ≈ 2 * energy_window * 2π * m / (2π)^2 rtol = 1e-4
    end

    @testset "3D sampling mesh may not outrun the integration mesh" begin
        # The fine measure is scattered onto the coarse sampling mesh, so a coarse mesh finer than the
        # integration mesh would leave coarse vertices with no mass and break the harmonic eigensolve.
        # Rejected up front, before any marching. 2D uses a single mesh and ignores sampling_depth.
        cubic = cubic_crystal()
        @test_throws ArgumentError ibz_quadrature_rule(k -> dot(k, k) - 1.0, k -> 2k, cubic,
            0.1, 4, 2; initial_depth = 3, refinement_depth = 4, sampling_depth = 6)
    end
end

@testset "bz_quadrature_rule emits a whole number of blocks in canonical layout" begin
    # The full-BZ rule replicates the IBZ nodes block by block, positional block `g` carrying
    # `operations(pg)[g]`. Everything downstream that reads a stored IBZ collision block — the
    # unfolding, the isotypic decomposition — assumes exactly that layout, so the grid the rule
    # actually emits has to satisfy it. `block_operation_map` recovers the correspondence from the
    # coordinates and is exercised on its own in `crystal_symmetry_test.jl`.
    band = graphene_band()
    pg = point_group(band.crystal)
    @test order(pg) == 12

    # Only the layout of the emitted grid is under test, so the under-resolution warnings this
    # configuration raises are captured and ignored rather than asserted: they are a property of the
    # graphene contour at these settings, not of the block replication. Accuracy of the compressed
    # rule is gated in the `ibz_quadrature_rule` testset above.
    nodes, _, _ = @test_logs match_mode=:any bz_quadrature_rule(
        band.dispersion, band.velocity, band.crystal, 0.1, 12, 2;
        initial_depth = 5, refinement_depth = 6)
    @test length(nodes) % order(pg) == 0
    N_ibz = length(nodes) ÷ order(pg)
    @test N_ibz > 0

    block_op = block_operation_map(nodes, pg)
    ops = operations(pg)
    @test sort(block_op) == collect(1:order(pg))   # a bijection onto the group
    @test block_op[1] == 1                         # operations(pg)[1] is the identity

    # 6mm operations carry √3/2, so images land on their partners to rounding, not bit-exactly.
    @test all(norm(nodes[(g - 1) * N_ibz + r] - ops[block_op[g]] * nodes[r]) ≤ 1e-12
              for g in 1:order(pg), r in 1:N_ibz)
end
