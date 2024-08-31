export f0, map_to_first_bz, get_bands

"""
        f0(E, T)

Return the value of the Fermi-Dirac distribution for energy `E` and temperature `T`.

```math
    f^{(0)}(\\varepsilon) = \\frac{1}{1 + e^{\\varepsilon/k_B T}}
```
"""
f0(E::Float64, T::Float64) = 1 / (exp(E/T) + 1)

"""
    map_to_first_bz(k)
Map a vector `k` to the ``d``-dimensional centered unit cube where ``d`` is the dimension of `k`.
"""
map_to_first_bz(k) = mod.(k .+ 0.5, 1.0) .- 0.5

"""
        get_bands(H, N)

Return an interpolation of the eigenvalues of `H` on a square grid [-0.5, 0.5].

It is assumed that `H` is a function of a vector of length 2 and returns a square matrix.
`N` is the number of points between -0.5 and 0.5 used for interpolation.
"""
function get_bands(H::Function, N::Int)
    n_bands = size(H([0.0, 0.0]))[1] # Number of bands
    E = Array{Float64}(undef, N, N, n_bands)
    x = LinRange(-0.5, 0.5, N)
    for i in 1:N, j in 1:N
        E[i, j, :] .= eigvals(H([x[i], x[j]]))# Get eigenvalues (bands) of each k-point
    end

    sitps = Vector{ScaledInterpolation}(undef, n_bands)
    for i in 1:n_bands
        itp = interpolate(E[:,:,i], BSpline(Cubic(Line(OnGrid()))))
        sitps[i] = scale(itp, -0.5:1/(N-1):0.5, -0.5:1/(N-1):0.5)
    end

    return sitps
end