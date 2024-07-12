module Ludwig

    const G0::Float64 = 7.748091729e-5 # Conductance quantum in Siemens
    const kb::Float64 = 8.6173e-5 # Boltzmann constant in eV / K
    const hbar::Float64 = 6.582119569e-16 # In eV.s 
    const e_charge::Float64 = 1.60218e-19 # C
    export G0, kb, hbar, e_charge

    export f0
    """
        f0(E, T)

    Return the value of the Fermi-Dirac distribution for energy E and temperature T.
    """
    f0(E::Float64, T::Float64) = 1 / (exp(E/T) + 1)

    import StaticArrays: SVector, MVector
    using LinearAlgebra
    using ForwardDiff
    import DataStructures: OrderedDict
    using ProgressBars
    using Interpolations
    using IterativeSolvers

    include("./mesh/marching_squares.jl")
    include("./mesh/mesh.jl")
    include("./integration.jl")
    include("./properties.jl")

    function band_weight(v, n)
        weights = Vector{Float64}(undef, n)
        ℓ = length(v) ÷ n
        for i in eachindex(weights)
            weights[i] = norm(@view v[(i - 1) * ℓ + 1: i * ℓ])
        end
    
        return weights / sum(weights) # Normalize the weights
    end

    function get_bands(H::Function, N::Int)
        n_bands = size(H([0.0, 0.0]))[1] # Number of bands
        E = Array{Float64}(undef, N, N, n_bands)
        x = LinRange(-0.5, 0.5, N)
        for i in 1:N, j in 1:N
            E[i, j, :] .= eigvals(H([x[i], x[j]]))# Get eigenvalues (bands) of each k-point
        end
    
        sitps = Vector{ScaledInterpolation}(undef, length(bands))
        for i in 1:n_bands
            itp = interpolate(E[:,:,i], BSpline(Cubic(Line(OnGrid()))))
            sitps[i] = scale(itp, -0.5:1/(N-1):0.5, -0.5:1/(N-1):0.5)
        end
    
        return sitps
    end

end # module Ludwig
