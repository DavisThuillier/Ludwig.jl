###
### Reusable band-structure test fixtures
###
#
# Materials for the end-to-end collision and quadrature tests: a `Crystal` together with the
# `dispersion` and `velocity` callbacks the collision integral and the quadrature pipeline expect.
# Momenta are Cartesian in units of the inverse lattice spacing, so these are the same callbacks the
# material files under `scripts/materials` provide; the tests deliberately do not depend on those
# files.
#
# Every fixture is a function, so nothing here is a top-level `const` whose redefinition would warn.
# Consumers guard the include (`@isdefined(graphene_band) || include("band_fixtures.jl")`) all the
# same: `Pkg.test` runs with `--warn-overwrite=yes`, under which a repeated include warns once per
# method it defines.

using Ludwig
using Ludwig: PointGroup
using StaticArrays

###
### Graphene (planar 6mm, order 12)
###

"""
    graphene_band(; hopping = 2.8, fermi = 0.1 * hopping) -> (; crystal, dispersion, velocity)

Nearest-neighbour tight-binding graphene: the hexagonal `Crystal{2}` carrying the planar 6mm point
group, the conduction band measured from `fermi`, and its velocity. Both callbacks take Cartesian
momenta in units of `1/a`.
"""
function graphene_band(; hopping::Float64 = 2.8, fermi::Float64 = 0.1 * hopping)
    primitives = SMatrix{2,2,Float64}(1.0, 0.0, 0.5, sqrt(3)/2)
    generators = [SMatrix{2,2,Float64}(0.5, sqrt(3)/2, -sqrt(3)/2, 0.5),   # C6 about z
                  SMatrix{2,2,Float64}(1.0, 0.0, 0.0, -1.0)]               # mirror across x
    crystal = Crystal(primitives, PointGroup(generators, primitives))

    # |f(k)|² for the honeycomb structure factor, which vanishes at the Dirac points.
    structure_factor(k) =
        3 + 2 * (cos(k[1]) + cos(k[1]/2 + sqrt(3)/2 * k[2]) + cos(k[1]/2 - sqrt(3)/2 * k[2]))

    dispersion(k) = hopping * sqrt(max(0.0, structure_factor(k))) - fermi

    function velocity(k)
        a = k[1]/2 + sqrt(3)/2 * k[2]
        b = k[1]/2 - sqrt(3)/2 * k[2]
        g = structure_factor(k)
        g <= 0.0 && return SVector(0.0, 0.0)   # the Dirac point, where the gradient is undefined
        hopping * SVector(-2*sin(k[1]) - sin(a) - sin(b), sqrt(3) * (sin(b) - sin(a))) / (2 * sqrt(g))
    end

    return (; crystal, dispersion, velocity)
end
