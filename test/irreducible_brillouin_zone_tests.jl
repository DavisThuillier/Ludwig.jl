using Ludwig
using LinearAlgebra
using StaticArrays
using Test
using Random

const SC  = Matrix{Float64}(I, 3, 3)
const FCC_real = [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0] / 2
const BCC_real = [-1.0 1.0 1.0; 1.0 -1.0 1.0; 1.0 1.0 -1.0] / 2

# Generators of Oh (full cubic point group, order 48): 4-fold about z, 3-fold
# along (1,1,1) (cyclic permutation), and inversion.
const OH_GENS = [
    [0.0 -1.0  0.0; 1.0 0.0 0.0; 0.0 0.0 1.0],   # C4z
    [0.0  0.0  1.0; 1.0 0.0 0.0; 0.0 1.0 0.0],   # C3 along (1,1,1)
    -Matrix{Float64}(I, 3, 3),                   # inversion
]

# Generators of C4v (square, order 8): C4 + mirror through x-axis.
const C4V_GENS = [
    [0.0 -1.0; 1.0 0.0],
    [1.0  0.0; 0.0 -1.0],
]

# Generators of C6v (hexagonal, order 12): C6 + mirror through x-axis.
const C6V_GENS = [
    [0.5 -sqrt(3)/2; sqrt(3)/2 0.5],
    [1.0  0.0; 0.0 -1.0],
]

const HEX_real = hcat(SVector(1.0, 0.0), SVector(-0.5, sqrt(3)/2))

@testset "IrreducibleBrillouinZone" begin
    @testset "volume = vol(BZ) / |G|" begin
        cases_3d = (
            (name = "simple cubic + Oh",   A = SC,       gens = OH_GENS),
            (name = "FCC + Oh",            A = FCC_real, gens = OH_GENS),
            (name = "BCC + Oh",            A = BCC_real, gens = OH_GENS),
        )
        for case in cases_3d
            @testset "$(case.name)" begin
                lat = Lattice(case.A)
                pg = PointGroup(case.gens, lat)
                ibz = IrreducibleBrillouinZone(lat, pg)
                vbz  = Ludwig.volume(lat.brillouin_zone)
                vibz = Ludwig.volume(ibz)
                @test vibz ≈ vbz / length(pg.operations) rtol = 1e-10
                @test vibz > 0
            end
        end

        cases_2d = (
            (name = "square + C4v",      A = Matrix{Float64}(I, 2, 2), gens = C4V_GENS),
            (name = "hexagonal + C6v",   A = HEX_real,                 gens = C6V_GENS),
        )
        for case in cases_2d
            @testset "$(case.name)" begin
                lat = Lattice(case.A)
                pg = PointGroup(case.gens, lat)
                ibz = IrreducibleBrillouinZone(lat, pg)
                vbz  = Ludwig.volume(lat.brillouin_zone)
                vibz = Ludwig.volume(ibz)
                @test vibz ≈ vbz / length(pg.operations) rtol = 1e-10
                @test vibz > 0
            end
        end
    end

    @testset "geometric invariants" begin
        for (A, gens, D) in (
            (SC,        OH_GENS,  3),
            (FCC_real,  OH_GENS,  3),
            (BCC_real,  OH_GENS,  3),
            (Matrix{Float64}(I, 2, 2), C4V_GENS, 2),
            (HEX_real,                 C6V_GENS, 2),
        )
            lat = Lattice(A)
            pg = PointGroup(gens, lat)
            ibz = IrreducibleBrillouinZone(lat, pg)
            scale = maximum(abs, ibz.offsets) + maximum(norm, ibz.vertices)

            # Every vertex satisfies n·v ≤ c for every facet.
            max_excess = maximum(
                dot(n, v) - c
                for v in ibz.vertices
                for (n, c) in zip(ibz.normals, ibz.offsets)
            )
            @test max_excess ≤ 1e-10 * scale

            # Every facet anchors at least D vertices.
            @test all(length(f.vertices) ≥ D for f in ibz.facets)

            # Field consistency.
            @test length(ibz.normals) == length(ibz.facets)
            @test length(ibz.offsets) == length(ibz.facets)

            # Outward normals: every facet's offset is non-negative (IBZ contains Γ).
            @test all(c ≥ -1e-12 * scale for c in ibz.offsets)

            if D == 3
                V, F, E = length(ibz.vertices), length(ibz.facets), length(ibz.ridges)
                @test V - E + F == 2  # Euler characteristic
                for (i, j) in ibz.ridges
                    @test 1 ≤ i < j ≤ V
                end
            else
                @test isempty(ibz.ridges)
            end
        end
    end

    @testset "Γ is an IBZ vertex" begin
        # Voronoi-bisector construction: bisectors pass through origin, so Γ is a
        # vertex whenever the point group is nontrivial.
        lat = Lattice(SC)
        pg = PointGroup(OH_GENS, lat)
        ibz = IrreducibleBrillouinZone(lat, pg)
        @test any(v -> norm(v) < 1e-12, ibz.vertices)
    end

    @testset "orbit cover" begin
        # Random points in the BZ are reachable from the IBZ by some g ∈ G.
        Random.seed!(0xC0FFEE)
        lat = Lattice(FCC_real)
        pg = PointGroup(OH_GENS, lat)
        ibz = IrreducibleBrillouinZone(lat, pg)
        bz = lat.brillouin_zone
        bound = 2 * maximum(norm, bz.vertices)
        scale = maximum(abs, ibz.offsets) + bound
        atol = 1e-10 * scale

        n_inside_bz = 0
        n_covered = 0
        while n_inside_bz < 200
            k = SVector{3, Float64}(2 * bound * (rand(3) .- 0.5))
            Ludwig.contains(bz, k) || continue
            n_inside_bz += 1
            covered = any(pg.operations) do g
                kp = SVector{3}(g * k)
                all(dot(n, kp) ≤ c + atol for (n, c) in zip(ibz.normals, ibz.offsets))
            end
            n_covered += covered
        end
        @test n_covered == n_inside_bz
    end

    @testset "trivial group → IBZ ≡ BZ" begin
        lat = Lattice(SC)
        pg = PointGroup([Matrix{Float64}(I, 3, 3)], lat)
        ibz = IrreducibleBrillouinZone(lat, pg)
        bz = lat.brillouin_zone
        @test length(ibz.vertices) == length(bz.vertices)
        @test length(ibz.facets)   == length(bz.facets)
        @test length(ibz.ridges)   == length(bz.ridges)
        @test Ludwig.volume(ibz) ≈ Ludwig.volume(bz) rtol = 1e-12
    end

    @testset "inversion only → vol(BZ)/2" begin
        lat = Lattice(SC)
        pg = PointGroup([-Matrix{Float64}(I, 3, 3)], lat)
        ibz = IrreducibleBrillouinZone(lat, pg)
        @test Ludwig.volume(ibz) ≈ Ludwig.volume(lat.brillouin_zone) / 2 rtol = 1e-12
    end

    @testset "square + C4v → Γ-X-M triangle" begin
        lat = Lattice(Matrix{Float64}(I, 2, 2))
        pg = PointGroup(C4V_GENS, lat)
        ibz = IrreducibleBrillouinZone(lat, pg)
        @test length(ibz.vertices) == 3
        @test length(ibz.facets)   == 3
    end

    @testset "type inference" begin
        lat3 = Lattice(SC)
        pg3 = PointGroup(OH_GENS, lat3)
        @test (@inferred IrreducibleBrillouinZone(lat3, pg3)) isa
              IrreducibleBrillouinZone{3, Float64}

        lat2 = Lattice(Matrix{Float64}(I, 2, 2))
        pg2 = PointGroup(C4V_GENS, lat2)
        @test (@inferred IrreducibleBrillouinZone(lat2, pg2)) isa
              IrreducibleBrillouinZone{2, Float64}
    end
end
