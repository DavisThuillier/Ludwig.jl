using Ludwig
using Test
using Interpolations
using LinearAlgebra
using Random

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
