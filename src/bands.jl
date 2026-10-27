###
### Band wrappers
###

"""
    HamiltonianBand(H::F, n::Int) where {F}

Wrap a matrix-valued Hamiltonian function `H(k)` for the `n`-th band.

`H` must return a square Hermitian (or real symmetric) matrix for each momentum
`k::AbstractVector`, with both triangles filled in. The band energy is the `n`-th
eigenvalue (in ascending order).

If `H` returns an `SMatrix`, every diagonalization runs on the stack by Jacobi rotations
(see [`hermitian_eigen`](@ref)), so evaluating the band allocates nothing. This is the form 
to use inside hot loops. Any other matrix is diagonalized by LAPACK.

!!! warning
    Selecting eigenvalues by ascending-order index is discontinuous across band
    crossings or near-degeneracies: when two bands cross or come within numerical
    noise of each other, the `n`-th eigenvalue can swap between the two physical
    bands as `k` varies, producing a non-smooth dispersion. Use only on bands that
    are well separated throughout the momentum region of interest, or apply a
    band-tracking strategy externally.

# Fields
- `H::F`: callable returning a Hermitian matrix as a function of momentum.
- `n::Int`: index of the band to extract (1-based, in ascending eigenvalue order).
"""
struct HamiltonianBand{F}
    H::F
    n::Int
end

(b::HamiltonianBand)(k) = hermitian_eigvals(b.H(k))[b.n]

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

# Fields
- `itp::I`: callable spline interpolation in fractional reciprocal coordinates.
- `invrlv::M`: inverse of the reciprocal-lattice-vector matrix.
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

band_velocity(b::HamiltonianBand, k) = bandstate(b, k)[2]

function band_velocity(b::InterpolatedBand, k)
    rlb = mod.(b.invrlv * k, 1.0)
    return b.invrlv' * Interpolations.gradient(b.itp, rlb[1], rlb[2])
end

"""
    bandstate(band, k)

Return the energy, group velocity, and state of `band` at Cartesian momentum `k` as the
tuple `(e, v, ψ)`.
"""
bandstate(band, k) = (band(k), band_velocity(band, k), nothing)

function bandstate(b::HamiltonianBand, k)
    F   = hermitian_eigen(b.H(k))
    ψ   = F.vectors[:, b.n]
    δ   = sqrt(eps(Float64))
    e1  = SVector{2,Float64}(1, 0)
    e2  = SVector{2,Float64}(0, 1)
    dHx = (b.H(k + δ*e1) - b.H(k - δ*e1)) / (2δ)
    dHy = (b.H(k + δ*e2) - b.H(k - δ*e2)) / (2δ)
    v   = SVector{2,Float64}(real(ψ'*dHx*ψ), real(ψ'*dHy*ψ))
    return F.values[b.n], v, ψ
end

###
### Construction from a callable
###

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
