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