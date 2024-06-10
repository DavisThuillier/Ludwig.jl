module Ludwig

    const G0::Float64 = 7.748091729e-5 # Conductance quantum in Siemens
    const kb::Float64 = 8.6173e-5 # Boltzmann constant in eV / K
    const hbar::Float64 = 6.582119569e-16 # In eV.s 
    export G0, kb, hbar

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

    function get_n(h::Function, N::Int = 2000)
        x = LinRange(0.0, 0.5, N)
        E = map(x -> h([x[1], x[2]]), collect(Iterators.product(x, x)))

        dn = 0.0
        c = find_contour(x, x, E, 0.0)
        for isoline in c.isolines
            for i in eachindex(isoline.points)
                i == length(isoline.points) && continue
                ds = norm(isoline.points[i + 1] - isoline.points[i])
                k = (isoline.points[i + 1] + isoline.points[i]) / 2
                dn += ds / norm(ForwardDiff.gradient(h, k))
            end
        end

        return 4 * dn
    end

    function band_weight(v, n)
        weights = Vector{Float64}(undef, n)
        ℓ = length(v) ÷ n
        for i in eachindex(weights)
            weights[i] = norm(@view v[(i - 1) * ℓ + 1: i * ℓ])
        end
    
        return weights / sum(weights) # Normalize the weights
    end

end # module Ludwig
