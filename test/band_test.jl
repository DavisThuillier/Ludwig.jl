using Ludwig
using Test
using Interpolations
using LinearAlgebra
using Random
using StaticArrays

###
### 2x2 Bloch Hamiltonian on a square lattice with primitive a = 1.
### Periodic in 2π in each component of k. The mass m is large enough
### that the diagonal never vanishes, leaving a finite gap between bands.
###

function H(k)
    s1, s2 = sin(k[1]), sin(k[2])
    c1, c2 = cos(k[1]), cos(k[2])
    m = 2.5
    d = m + c1 + c2
    return [d                s1 - im*s2;
            s1 + im*s2       -d        ]
end

l  = Lattice([1.0 0.0; 0.0 1.0])
hb = HamiltonianBand(H, 1)  # lower band

###
### Build a periodic cubic-spline interpolation of the lower band on a
### uniform grid of fractional reciprocal coordinates [0, 1)^2. For this
### lattice k_cartesian = 2π * k_fractional.
###

N  = 256
xs = range(0.0, stop = 1.0, length = N + 1)  # both endpoints; last == first by periodicity
samples = [hb([2π*kx, 2π*ky]) for kx in xs, ky in xs]

itp = Interpolations.scale(
    interpolate(samples, BSpline(Cubic(Periodic(OnGrid())))),
    xs, xs,
)
ib = InterpolatedBand(itp, l)

###
### Draw probes from a uniform distribution on [0, 1)^2 in fractional
### reciprocal coordinates. The chance of landing exactly on a sample
### node (a measure-zero set) is zero, so every probe exercises the
### spline rather than a node value.
###

rng = MersenneTwister(0x6c75646c)
n_probes = 100
probes = [2π * rand(rng, 2) for _ in 1:n_probes]

@testset "Energy agreement" begin
    for k in probes
        @test isapprox(hb(k), ib(k); atol = 1e-4)
    end
end

@testset "Velocity agreement" begin
    for k in probes
        v_h = band_velocity(hb, k)
        v_i = band_velocity(ib, k)
        @test isapprox(v_h, v_i; atol = 1e-3)
    end
end

###
### Repeat on an oblique lattice so that `invrlv` is non-symmetric and
### the gradient chain rule `∇_k ε = invrlv^T · ∇_f itp` is genuinely
### exercised. (On the square lattice above, invrlv = (1/2π) I and the
### transpose is a no-op.) The Hamiltonian is defined through fractional
### reciprocal coordinates so that it is truly periodic on the oblique BZ.
###

l_obl = Lattice([1.0 0.3; 0.0 1.1])
invrlv_obl = inv(reciprocal_lattice_vectors(l_obl))

function H_obl(k)
    f1, f2 = invrlv_obl * k
    s1, s2 = sin(2π*f1), sin(2π*f2)
    c1, c2 = cos(2π*f1), cos(2π*f2)
    m = 2.5
    d = m + c1 + c2
    return [d                s1 - im*s2;
            s1 + im*s2       -d        ]
end
hb_obl = HamiltonianBand(H_obl, 1)

rlv_obl = reciprocal_lattice_vectors(l_obl)
samples_obl = [hb_obl(rlv_obl * [fx, fy]) for fx in xs, fy in xs]
itp_obl = Interpolations.scale(
    interpolate(samples_obl, BSpline(Cubic(Periodic(OnGrid())))),
    xs, xs,
)
ib_obl = InterpolatedBand(itp_obl, l_obl)

probes_obl = [rlv_obl * rand(rng, 2) for _ in 1:n_probes]

@testset "Energy agreement (oblique lattice)" begin
    for k in probes_obl
        @test isapprox(hb_obl(k), ib_obl(k); atol = 1e-4)
    end
end

@testset "Velocity agreement (oblique lattice)" begin
    for k in probes_obl
        v_h = band_velocity(hb_obl, k)
        v_i = band_velocity(ib_obl, k)
        @test isapprox(v_h, v_i; atol = 5e-3)
    end
end

###
### IBZInterpolatedBand: sample on the IBZ bounding rectangle of the square
### lattice ([0, 0.5]² in fractional reciprocal coords). The reference H(k)
### is D₄-symmetric in addition to being BZ-periodic, so any Cartesian k in
### the full BZ folds to an IBZ image whose energy and velocity must match
### the direct H(k) reference.
###

sentinel = 1e3
xs_ibz = range(0.0, stop = 0.5, length = N + 1)
ys_ibz = range(0.0, stop = 0.5, length = N + 1)
samples_ibz = [hb([2π*fx, 2π*fy]) for fx in xs_ibz, fy in ys_ibz]
itp_ibz = Interpolations.scale(
    interpolate(samples_ibz, BSpline(Cubic(Line(OnGrid())))),
    xs_ibz, ys_ibz,
)
ibz_band = IBZInterpolatedBand(itp_ibz, l; sentinel = sentinel)

# Probes drawn uniformly from the full BZ [-π, π)²; both translation
# folding and point-group folding are exercised.
bz_probes = [SVector(2π*(rand(rng) - 0.5), 2π*(rand(rng) - 0.5)) for _ in 1:n_probes]

@testset "IBZInterpolatedBand: full-BZ energy" begin
    for k in bz_probes
        @test isapprox(hb(k), ibz_band(k); atol = 1e-3)
    end
end

@testset "IBZInterpolatedBand: full-BZ velocity" begin
    for k in bz_probes
        v_h = band_velocity(hb, k)
        v_i = band_velocity(ibz_band, k)
        @test isapprox(v_h, v_i; atol = 5e-3)
    end
end

# Distant translation: the same k offset by a reciprocal lattice vector must
# give the identical band value (catches a missing or wrong map_to_bz step).
@testset "IBZInterpolatedBand: translation invariance" begin
    rlv_sq = reciprocal_lattice_vectors(l)
    for k in bz_probes[1:20]
        for shift in (rlv_sq * [1, 0], rlv_sq * [-1, 0], rlv_sq * [0, 1], rlv_sq * [2, -3])
            @test isapprox(ibz_band(k), ibz_band(k + shift); atol = 1e-9)
        end
    end
end

###
### Out-of-support: sample only [0, 0.2]² in fractional coords. Probes
### whose IBZ image has |f| > 0.2 should hit the sentinel.
###

xs_small = range(0.0, stop = 0.2, length = N + 1)
ys_small = range(0.0, stop = 0.2, length = N + 1)
samples_small = [hb([2π*fx, 2π*fy]) for fx in xs_small, fy in ys_small]
itp_small = Interpolations.scale(
    interpolate(samples_small, BSpline(Cubic(Line(OnGrid())))),
    xs_small, ys_small,
)
ibz_band_small = IBZInterpolatedBand(itp_small, l; sentinel = sentinel)

@testset "IBZInterpolatedBand: out-of-support energy" begin
    # Pick fractional points strictly inside the IBZ wedge (0 ≤ f₂ ≤ f₁ ≤ 0.5)
    # but outside the support box [0, 0.2]².
    for (fx, fy) in ((0.4, 0.1), (0.3, 0.25), (0.45, 0.05))
        k = SVector(2π*fx, 2π*fy)
        @test ibz_band_small(k) == sentinel
    end
end

@testset "IBZInterpolatedBand: out-of-support velocity" begin
    for (fx, fy) in ((0.4, 0.1), (0.3, 0.25), (0.45, 0.05))
        k = SVector(2π*fx, 2π*fy)
        @test band_velocity(ibz_band_small, k) == SVector(0.0, 0.0)
    end
end

###
### Convenience constructors: pass a callable + lattice + resolution and
### get a fully-built band, no manual interpolation step required.
###

@testset "InterpolatedBand convenience constructor" begin
    ib_auto = InterpolatedBand(hb, l, N)
    for k in probes
        @test isapprox(hb(k), ib_auto(k); atol = 1e-4)
        @test isapprox(band_velocity(hb, k), band_velocity(ib_auto, k); atol = 1e-3)
    end
end

@testset "IBZInterpolatedBand convenience constructor" begin
    ibz_auto = IBZInterpolatedBand(hb, l, N; sentinel = sentinel)
    for k in bz_probes
        @test isapprox(hb(k), ibz_auto(k); atol = 1e-3)
        @test isapprox(band_velocity(hb, k), band_velocity(ibz_auto, k); atol = 5e-3)
    end
end

@testset "InterpolatedBand range constructor" begin
    xs_full = range(0.0, stop = 1.0, length = N + 1)
    ib_range = InterpolatedBand(hb, l, xs_full)
    for k in probes
        @test isapprox(hb(k), ib_range(k); atol = 1e-4)
    end
end

@testset "IBZInterpolatedBand range constructor" begin
    # Custom sub-rectangle [0, 0.2]² inside the IBZ wedge: in-support points
    # match the reference; points whose IBZ image is outside the rectangle
    # hit the sentinel.
    xs_custom = range(0.0, stop = 0.2, length = N + 1)
    ibz_custom = IBZInterpolatedBand(hb, l, xs_custom; sentinel = sentinel)
    for _ in 1:n_probes
        # In-support: pick fx ∈ [0, 0.2], fy ∈ [0, fx] (inside IBZ wedge)
        fx = 0.2 * rand(rng)
        fy = fx * rand(rng)
        k = SVector(2π*fx, 2π*fy)
        @test isapprox(hb(k), ibz_custom(k); atol = 1e-3)
    end
    for (fx, fy) in ((0.4, 0.1), (0.3, 0.25), (0.45, 0.05))
        k = SVector(2π*fx, 2π*fy)
        @test ibz_custom(k) == sentinel
    end
end
