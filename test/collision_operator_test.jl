###
### Electron-electron collision integral (src/collision_operator.jl)
###

using Test
using Ludwig
using StaticArrays
using LinearAlgebra

@isdefined(graphene_band) || include("band_fixtures.jl")

@testset "Electron-electron collision integral" begin
    Weff_squared(k1, k2, k3, k4) = 1.0

    @testset "no regression on a periodic band" begin
        band = graphene_band()
        ki = SVector(4.0738, 0.0)          # on the graphene Fermi pocket near the Dirac point (4π/3, 0)
        kj = SVector(4.3064, 0.05)

        Lij = electron_electron(ki, kj, 1.0, band.crystal, band.dispersion, band.velocity, 0.025,
                                Weff_squared; initial_depth = 3, refinement_depth = 6)
        # Baseline computed on the pre-fix (develop) code. Recentring the marching cell on Γ leaves a
        # lattice-periodic band's collision element unchanged to ~13 significant figures.
        @test isapprox(Lij, -1.467258450459861e-3; rtol = 1e-9)
    end

    @testset "translation invariance of a Γ-centered continuum pocket" begin
        # Square lattice with rlv = I: Cartesian momenta equal fractional coords and the marching cell
        # is exactly [-1/2, 1/2]^2. The monotonic radial band has a single Fermi circle |k - c| = R, so
        # a rigid shift of the whole pocket (band + both incoming nodes) may not change the element — a
        # property that only holds once the cell is centred on Γ.
        lat = Lattice(SMatrix{2,2,Float64}(2π, 0.0, 0.0, 2π))
        R = 0.1
        T = 0.02
        function element(c)
            dispersion(k) = norm(k - c) - R
            function velocity(k)
                d = k - c
                n = norm(d)
                n == 0.0 ? SVector(0.0, 0.0) : d / n
            end
            ki = c + SVector(R, 0.0)                     # on the pocket, ε = 0
            kj = c + R * SVector(cos(2.0), sin(2.0))     # on the pocket, ε = 0
            electron_electron(ki, kj, 1.0, lat, dispersion, velocity, T, Weff_squared;
                              initial_depth = 4, refinement_depth = 7)
        end

        base    = element(SVector(0.0, 0.0))
        shifted = element(SVector(0.13, -0.09))          # pocket stays interior to the cell
        @test abs(base) > 1e-4                            # a genuine, non-vacuous element
        @test isapprox(shifted, base; rtol = 1e-3)
    end
end
