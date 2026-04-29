###
### Band wrappers
###

"""
    HamiltonianBand(H::F, n::Int) where {F}

Wrap a matrix-valued Hamiltonian function `H(k)` for the `n`-th band.

`H` must return a square Hermitian (or real symmetric) matrix for each momentum
`k::AbstractVector`. The band energy is the `n`-th eigenvalue (in ascending order);
the group velocity is computed by the Hellmann-Feynman theorem using central finite
differences to differentiate `H`.

# Fields
- `H::F`: callable returning a Hermitian (or real symmetric) matrix as a function of
  momentum.
- `n::Int`: index of the band to extract (1-based, in ascending eigenvalue order).

# Examples
```julia
function H(k)
    t = 1.0
    [0.0+0im  t*(exp(im*k[1]) + exp(im*k[2]));
     t*(exp(-im*k[1]) + exp(-im*k[2]))  0.0+0im]
end
band = HamiltonianBand(H, 1)  # lower band
```

See also [`InterpolatedBand`](@ref), [`band_velocity`](@ref), [`bz_mesh`](@ref),
[`ibz_mesh`](@ref).
"""
struct HamiltonianBand{F}
    H::F
    n::Int
end

(b::HamiltonianBand)(k) = real(eigvals(Hermitian(b.H(k)))[b.n])

"""
    InterpolatedBand(itp::I, invrlv::M) where {I, M}
    InterpolatedBand(itp::I, l::AbstractLattice) where {I}

Wrap a 2D dispersion interpolation defined on the unit cell ``[0,1)^2`` of
reciprocal-lattice coordinates.

`itp` is a callable `(rlb1, rlb2) -> ε` such as one returned by
`Interpolations.scale(interpolate(...))`, evaluated at fractional reciprocal
coordinates. `invrlv` is the inverse of the reciprocal-lattice-vector matrix:
given a Cartesian momentum `k`, the band evaluates `itp` at
`mod.(invrlv * k, 1.0)`. When constructed from an `AbstractLattice`,
`invrlv = inv(reciprocal_lattice_vectors(l))` is computed automatically.

The group velocity is obtained from `Interpolations.gradient` on the spline
basis (analytic derivatives, not autodiff) and transformed back to Cartesian
momentum coordinates by ``\\nabla_k \\varepsilon = \\mathrm{invrlv}^\\top
\\nabla_f \\mathrm{itp}``, where ``f = \\mathrm{invrlv}\\,k`` are the
fractional reciprocal coordinates.

# Fields
- `itp::I`: callable spline interpolation in fractional reciprocal coordinates.
- `invrlv::M`: inverse of the reciprocal-lattice-vector matrix.

See also [`HamiltonianBand`](@ref), [`band_velocity`](@ref).
"""
struct InterpolatedBand{I, M}
    itp::I
    invrlv::M
end

InterpolatedBand(itp, l::AbstractLattice) =
    InterpolatedBand(itp, inv(reciprocal_lattice_vectors(l)))

function (b::InterpolatedBand)(k)
    rlb = mod.(b.invrlv * k, 1.0)
    return b.itp(rlb[1], rlb[2])
end

"""
    band_velocity(band, k)

Return the group velocity of `band` at Cartesian momentum `k`.

Dispatches on band type:
- a plain `Function` is differentiated with `ForwardDiff.gradient`;
- a [`HamiltonianBand`](@ref) uses the Hellmann-Feynman theorem with central
  finite differences applied to the Hamiltonian matrix;
- an [`InterpolatedBand`](@ref) uses `Interpolations.gradient` on the spline
  basis and transforms the result back to Cartesian coordinates;
- an [`IBZInterpolatedBand`](@ref) folds `k` into the IBZ via the lattice
  point group and returns ``O^\\top \\mathrm{invrlv}^\\top \\nabla_f
  \\mathrm{itp}`` inside the support box (zero outside).
"""
band_velocity(ε::Function, k) = gradient(ε, k)

function band_velocity(b::HamiltonianBand, k)
    F   = eigen(Hermitian(b.H(k)))
    ψ   = F.vectors[:, b.n]
    δ   = sqrt(eps(Float64))
    e1  = SVector{2,Float64}(1, 0)
    e2  = SVector{2,Float64}(0, 1)
    dHx = (b.H(k + δ*e1) - b.H(k - δ*e1)) / (2δ)
    dHy = (b.H(k + δ*e2) - b.H(k - δ*e2)) / (2δ)
    return SVector{2,Float64}(real(ψ'*dHx*ψ), real(ψ'*dHy*ψ))
end

function band_velocity(b::InterpolatedBand, k)
    rlb = mod.(b.invrlv * k, 1.0)
    return b.invrlv' * Interpolations.gradient(b.itp, rlb[1], rlb[2])
end

"""
    IBZInterpolatedBand(itp::I, rlv::SMatrix{2,2,Float64,4}, invrlv::SMatrix{2,2,Float64,4},
                        bz::Vector{SVector{2,Float64}},
                        ibz::Vector{SVector{2,Float64}},
                        group_ops::Vector{SMatrix{2,2,Float64,4}},
                        sentinel::Float64) where {I}
    IBZInterpolatedBand(itp::I, l::AbstractLattice; sentinel::Real=1e3) where {I}

Wrap a 2D dispersion interpolation that is sampled over (a sub-region of) the
irreducible Brillouin zone of `l`, and represents the band over the *full*
Brillouin zone via the lattice point-group symmetry.

`itp` is a callable `(rlb1, rlb2) -> ε` (e.g. from `Interpolations.scale(...)`)
defined on a rectangle in fractional reciprocal coordinates that contains
the IBZ. At evaluation a Cartesian momentum `k` is folded to the first BZ
via [`map_to_bz`](@ref), then the point-group operation
``O \\in G(l)`` whose image ``O \\cdot k_{bz}`` lies in the IBZ polygon
is selected. The IBZ image is converted to fractional coordinates and
looked up in the interpolation: if it lies in the interpolation support box
the spline value is returned, otherwise `sentinel`.

The default sentinel is far above any realistic chemical potential, so the
energy-conservation cutoff inside the electron-electron kernel naturally
suppresses out-of-support scattering momenta without any caller-side change.

The group velocity transforms as
``\\nabla_k \\varepsilon = O^\\top \\,\\mathrm{invrlv}^\\top\\, \\nabla_f
\\mathrm{itp}`` inside the support and is the zero vector outside.

# Fields
- `itp::I`: callable spline interpolation in fractional reciprocal coordinates.
- `rlv::SMatrix{2,2,Float64,4}`: reciprocal-lattice-vector matrix.
- `invrlv::SMatrix{2,2,Float64,4}`: inverse of `rlv`.
- `bz::Vector{SVector{2,Float64}}`: vertices of the first Brillouin zone.
- `ibz::Vector{SVector{2,Float64}}`: vertices of the irreducible Brillouin zone.
- `group_ops::Vector{SMatrix{2,2,Float64,4}}`: 2×2 matrix representations of every
  point-group element.
- `sentinel::Float64`: value returned when a folded momentum lies outside the
  interpolation support box.

See also [`InterpolatedBand`](@ref), [`band_velocity`](@ref).
"""
struct IBZInterpolatedBand{I}
    itp::I
    rlv::SMatrix{2,2,Float64,4}
    invrlv::SMatrix{2,2,Float64,4}
    bz::Vector{SVector{2,Float64}}
    ibz::Vector{SVector{2,Float64}}
    group_ops::Vector{SMatrix{2,2,Float64,4}}
    sentinel::Float64
end

function IBZInterpolatedBand(itp, l::AbstractLattice; sentinel::Real=1e3)
    rlv = reciprocal_lattice_vectors(l)
    G   = point_group(l)
    ops = [SMatrix{2,2,Float64}(get_matrix_representation(g)) for g in G.elements]
    return IBZInterpolatedBand(itp, rlv, inv(rlv), get_bz(l), get_ibz(l),
                               ops, Float64(sentinel))
end

"""
    fold_to_ibz(b::IBZInterpolatedBand, k)

Fold Cartesian momentum `k` into the irreducible Brillouin zone of `b`.

Returns `(kibz, O)` where `kibz = O * kbz` lies in the IBZ polygon and `kbz` is the first-BZ
image of `k`. Falls back to the identity operation when no exact image of `kbz` lies inside
the IBZ polygon (numerical edge case at the IBZ boundary).
"""
function fold_to_ibz(b::IBZInterpolatedBand, k)
    kbz = map_to_bz(k, b.bz, b.rlv, b.invrlv)
    for O in b.group_ops
        kibz = O * kbz
        in_polygon(kibz, b.ibz) && return SVector{2,Float64}(kibz), O
    end
    # Numerical edge of IBZ: any image is symmetry-equivalent, fall back to identity.
    return SVector{2,Float64}(kbz), SMatrix{2,2,Float64}(1.0, 0.0, 0.0, 1.0)
end

function (b::IBZInterpolatedBand)(k)
    kibz, _ = fold_to_ibz(b, k)
    f = b.invrlv * kibz
    (xlo, xhi), (ylo, yhi) = Interpolations.bounds(b.itp)
    return (xlo ≤ f[1] ≤ xhi && ylo ≤ f[2] ≤ yhi) ? b.itp(f[1], f[2]) : b.sentinel
end

function band_velocity(b::IBZInterpolatedBand, k)
    kibz, O = fold_to_ibz(b, k)
    f = b.invrlv * kibz
    (xlo, xhi), (ylo, yhi) = Interpolations.bounds(b.itp)
    if xlo ≤ f[1] ≤ xhi && ylo ≤ f[2] ≤ yhi
        return O' * (b.invrlv' * Interpolations.gradient(b.itp, f[1], f[2]))
    else
        return SVector{2,Float64}(0.0, 0.0)
    end
end

"""
    InterpolatedBand(band, l::AbstractLattice, xs::AbstractRange, ys::AbstractRange=xs)
    InterpolatedBand(band, l::AbstractLattice, n::Integer)

Build a periodic cubic-spline `InterpolatedBand` from a callable `band`.

`band` is any callable mapping a Cartesian momentum to an energy
(e.g. a [`HamiltonianBand`](@ref) or a closure). Samples are taken at
`band(rlv * [fx, fy])` for each `(fx, fy)` in the outer product `xs × ys`,
where `rlv = reciprocal_lattice_vectors(l)`. The spline uses a periodic
cubic B-spline, so `band` must be 1-periodic in each fractional axis (i.e.
BZ-periodic) and `xs`, `ys` should span exactly one period of `band`. The
integer-resolution form is shorthand for `xs = ys = range(0, 1, length=n+1)`.
"""
function InterpolatedBand(band, l::AbstractLattice, xs::AbstractRange,
                          ys::AbstractRange=xs)
    rlv = reciprocal_lattice_vectors(l)
    samples = [band(rlv * [fx, fy]) for fx in xs, fy in ys]
    itp = Interpolations.scale(
        Interpolations.interpolate(samples, BSpline(Cubic(Periodic(OnGrid())))),
        xs, ys,
    )
    return InterpolatedBand(itp, l)
end

InterpolatedBand(band, l::AbstractLattice, n::Integer) =
    InterpolatedBand(band, l, range(0.0, stop=1.0, length=n + 1))

"""
    IBZInterpolatedBand(band, l::AbstractLattice, xs::AbstractRange, ys::AbstractRange=xs;
                        sentinel=1e3)
    IBZInterpolatedBand(band, l::AbstractLattice, n::Integer; sentinel=1e3)

Build an `IBZInterpolatedBand` from a callable `band`.

Samples are taken at `band(rlv * [fx, fy])` for each `(fx, fy)` in `xs × ys`,
where `xs`, `ys` are ranges in fractional reciprocal coordinates that should
contain (a region of interest within) the IBZ. The spline uses a non-periodic
cubic B-spline with Line boundary conditions. Since `band` is assumed to
respect the lattice point-group symmetry, grid points outside the IBZ but
inside the sampled rectangle automatically agree with their IBZ images.

The integer-resolution form picks `xs`, `ys` to be the IBZ bounding rectangle
(read from [`get_ibz`](@ref)) with `n+1` samples per axis.
"""
function IBZInterpolatedBand(band, l::AbstractLattice, xs::AbstractRange,
                             ys::AbstractRange=xs; sentinel::Real=1e3)
    rlv = reciprocal_lattice_vectors(l)
    samples = [band(rlv * [fx, fy]) for fx in xs, fy in ys]
    itp = Interpolations.scale(
        Interpolations.interpolate(samples, BSpline(Cubic(Line(OnGrid())))),
        xs, ys,
    )
    return IBZInterpolatedBand(itp, l; sentinel=sentinel)
end

function IBZInterpolatedBand(band, l::AbstractLattice, n::Integer; sentinel::Real=1e3)
    ibz_frac = [inv(reciprocal_lattice_vectors(l)) * v for v in get_ibz(l)]
    xlo, xhi = extrema(v[1] for v in ibz_frac)
    ylo, yhi = extrema(v[2] for v in ibz_frac)
    xs = range(xlo, stop=xhi, length=n + 1)
    ys = range(ylo, stop=yhi, length=n + 1)
    return IBZInterpolatedBand(band, l, xs, ys; sentinel=sentinel)
end
