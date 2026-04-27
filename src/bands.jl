###
### Band wrappers
###

"""
    HamiltonianBand(H, n)

Wrap a matrix-valued Hamiltonian function `H(k)` for the `n`-th band.

`H` must return a square Hermitian (or real symmetric) matrix for each momentum
`k::AbstractVector`. The band energy is the `n`-th eigenvalue (in ascending order);
the group velocity is computed by the Hellmann-Feynman theorem using central finite
differences to differentiate `H`.

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
    InterpolatedBand(itp, invrlv)
    InterpolatedBand(itp, l::AbstractLattice)

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
momentum coordinates by `invrlv`.

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
  basis and transforms the result back to Cartesian coordinates.
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
    return b.invrlv * Interpolations.gradient(b.itp, rlb[1], rlb[2])
end
