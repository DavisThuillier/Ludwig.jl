using Ludwig
using LinearAlgebra
using StaticArrays
using Test

const FCC = [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0] / 2
const BCC = [-1.0 1.0 1.0; 1.0 -1.0 1.0; 1.0 1.0 -1.0] / 2

@testset "BrillouinZone" begin
    @testset "argument validation" begin
        @test_throws ArgumentError Ludwig.BrillouinZone(SMatrix{1, 1, Float64, 1}(1.0))
        @test_throws ArgumentError Ludwig.BrillouinZone(
            SMatrix{4, 4, Float64, 16}(2π * I(4)),
        )
    end

    @testset "3D combinatorics" begin
        # Real lattice → expected (V, F, E) of the BZ (Wigner-Seitz of the
        # reciprocal lattice).
        cases = (
            (name = "cubic",
             rlv  = SMatrix{3, 3, Float64, 9}(2π * I(3)),
             vfe  = (8, 6, 12)),
            (name = "FCC real (BZ = truncated octahedron)",
             rlv  = SMatrix{3, 3, Float64, 9}(Ludwig.reciprocal_lattice_vectors(FCC)),
             vfe  = (24, 14, 36)),
            (name = "BCC real (BZ = rhombic dodecahedron)",
             rlv  = SMatrix{3, 3, Float64, 9}(Ludwig.reciprocal_lattice_vectors(BCC)),
             vfe  = (14, 12, 24)),
        )
        for case in cases
            @testset "$(case.name)" begin
                bz = Ludwig.BrillouinZone(case.rlv)
                V, F, E = length(bz.vertices), length(bz.facets), length(bz.ridges)
                @test (V, F, E) == case.vfe
                @test V - E + F == 2  # Euler characteristic for a convex polyhedron
                @test length(bz.normals) == F
                @test length(bz.offsets) == F
            end
        end
    end

    @testset "2D combinatorics" begin
        cases = (
            (name = "square",
             rlv  = SMatrix{2, 2, Float64, 4}(2π * I(2)),
             vf   = (4, 4)),
            (name = "hexagonal",
             rlv  = SMatrix{2, 2, Float64, 4}(Ludwig.reciprocal_lattice_vectors(
                        hcat(SVector(1.0, 0.0), SVector(0.5, sqrt(3)/2)))),
             vf   = (6, 6)),
            (name = "oblique",
             rlv  = SMatrix{2, 2, Float64, 4}(Ludwig.reciprocal_lattice_vectors(
                        hcat(SVector(1.0, 0.0), SVector(0.3, 0.7)))),
             vf   = (6, 6)),
            (name = "rectangular 1:2",
             rlv  = SMatrix{2, 2, Float64, 4}(Ludwig.reciprocal_lattice_vectors(
                        hcat(SVector(1.0, 0.0), SVector(0.0, 2.0)))),
             vf   = (4, 4)),
        )
        for case in cases
            @testset "$(case.name)" begin
                bz = Ludwig.BrillouinZone(case.rlv)
                @test (length(bz.vertices), length(bz.facets)) == case.vf
                @test isempty(bz.ridges)  # 3D-only field
                @test length(bz.normals) == length(bz.facets)
                @test length(bz.offsets) == length(bz.facets)
            end
        end
    end

    @testset "geometric invariants" begin
        rlvs = (
            SMatrix{3, 3, Float64, 9}(2π * I(3)),
            SMatrix{3, 3, Float64, 9}(Ludwig.reciprocal_lattice_vectors(FCC)),
            SMatrix{3, 3, Float64, 9}(Ludwig.reciprocal_lattice_vectors(BCC)),
        )
        for rlv in rlvs
            bz = Ludwig.BrillouinZone(rlv)
            scale = maximum(bz.offsets)

            # Every vertex sits on its incident facets and inside every other.
            max_excess = maximum(
                dot(n, v) - c
                for v in bz.vertices for (n, c) in zip(bz.normals, bz.offsets)
            )
            @test max_excess ≤ 1e-12 * scale

            # Every facet is anchored by at least D vertices.
            @test all(length(f.vertices) ≥ 3 for f in bz.facets)

            # Outward normals point away from origin.
            @test all(c > 0 for c in bz.offsets)

            # Inversion symmetry: every facet plane has its antipode.
            for (n, c) in zip(bz.normals, bz.offsets)
                @test any(
                    norm(n + n2) < 1e-10 && abs(c - c2) < 1e-10 * scale
                    for (n2, c2) in zip(bz.normals, bz.offsets)
                )
            end

            # Each ridge connects two distinct vertices.
            for (i, j) in bz.ridges
                @test 1 ≤ i < j ≤ length(bz.vertices)
            end
        end
    end

    @testset "type inference" begin
        rlv = SMatrix{2, 2, Float64, 4}(2π * I(2))
        bz = Ludwig.BrillouinZone(rlv)
        @test bz isa Ludwig.BrillouinZone{2, Float64}

        rlv = SMatrix{3, 3, Float64, 9}(2π * I(3))
        bz = Ludwig.BrillouinZone(rlv)
        @test bz isa Ludwig.BrillouinZone{3, Float64}
    end
end
