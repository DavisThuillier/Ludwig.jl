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
